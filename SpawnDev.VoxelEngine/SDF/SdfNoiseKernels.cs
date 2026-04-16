using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.SDF
{
    /// <summary>
    /// GPU kernels for SDF terrain evaluation via layered noise.
    ///
    /// Architecture (NMS "Uber Noise" inspired):
    /// - Domain-warped fractional Brownian motion (fBm)
    /// - 7 composable noise layers with per-biome weighting
    /// - CSG boolean operations for caves and overhangs
    /// - Output: 16-bit fixed-point SDF values
    ///
    /// Dispatch: one thread per voxel (32x32x32 = 32,768 threads).
    /// Embarrassingly parallel - no data dependencies between voxels.
    /// </summary>
    public static class SdfNoiseKernels
    {
        // ===================================================================
        // Noise primitives (GPU-compatible, no Math.F or System calls)
        // ===================================================================

        /// <summary>
        /// GPU-compatible hash function for 3D integer coordinates.
        /// Returns pseudo-random float in [0, 1).
        /// </summary>
        private static float Hash3D(int x, int y, int z)
        {
            int h = x * 374761393 + y * 668265263 + z * 1274126177;
            h = (h ^ (h >> 13)) * 1274126177;
            h = h ^ (h >> 16);
            // Map to [0, 1) - use unsigned interpretation
            return (h & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        }

        /// <summary>
        /// Smooth interpolation (smoothstep / Hermite).
        /// </summary>
        private static float Fade(float t)
        {
            return t * t * t * (t * (t * 6f - 15f) + 10f);
        }

        /// <summary>
        /// Linear interpolation.
        /// </summary>
        private static float Lerp(float a, float b, float t)
        {
            return a + (b - a) * t;
        }

        /// <summary>
        /// Value noise at 3D position. Returns [-1, 1].
        /// Simple, fast, no gradient table needed - works on all ILGPU backends.
        /// </summary>
        private static float ValueNoise3D(float x, float y, float z)
        {
            int ix = x >= 0 ? (int)x : (int)x - 1;
            int iy = y >= 0 ? (int)y : (int)y - 1;
            int iz = z >= 0 ? (int)z : (int)z - 1;

            float fx = x - ix;
            float fy = y - iy;
            float fz = z - iz;

            float u = Fade(fx);
            float v = Fade(fy);
            float w = Fade(fz);

            // 8 corner values
            float c000 = Hash3D(ix, iy, iz);
            float c100 = Hash3D(ix + 1, iy, iz);
            float c010 = Hash3D(ix, iy + 1, iz);
            float c110 = Hash3D(ix + 1, iy + 1, iz);
            float c001 = Hash3D(ix, iy, iz + 1);
            float c101 = Hash3D(ix + 1, iy, iz + 1);
            float c011 = Hash3D(ix, iy + 1, iz + 1);
            float c111 = Hash3D(ix + 1, iy + 1, iz + 1);

            // Trilinear interpolation
            float x00 = Lerp(c000, c100, u);
            float x10 = Lerp(c010, c110, u);
            float x01 = Lerp(c001, c101, u);
            float x11 = Lerp(c011, c111, u);
            float y0 = Lerp(x00, x10, v);
            float y1 = Lerp(x01, x11, v);

            return Lerp(y0, y1, w) * 2f - 1f; // map [0,1] to [-1,1]
        }

        /// <summary>
        /// Fractional Brownian motion (fBm) - layered noise at multiple frequencies.
        /// </summary>
        /// <param name="x">World X position.</param>
        /// <param name="y">World Y position.</param>
        /// <param name="z">World Z position.</param>
        /// <param name="octaves">Number of noise layers (1-8).</param>
        /// <param name="frequency">Base sampling frequency.</param>
        /// <param name="persistence">Amplitude decay per octave (typically 0.5).</param>
        /// <param name="lacunarity">Frequency multiplier per octave (typically 2.0).</param>
        private static float FBM(float x, float y, float z,
            int octaves, float frequency, float persistence, float lacunarity)
        {
            float sum = 0f;
            float amplitude = 1f;
            float maxAmplitude = 0f;
            float freq = frequency;

            for (int i = 0; i < octaves && i < 8; i++)
            {
                sum += ValueNoise3D(x * freq, y * freq, z * freq) * amplitude;
                maxAmplitude += amplitude;
                amplitude *= persistence;
                freq *= lacunarity;
            }

            return sum / maxAmplitude; // normalize to [-1, 1]
        }

        /// <summary>
        /// Domain-warped fBm (NMS "Uber Noise" approach).
        /// Each sample point is offset by another fBm evaluation before the main fBm.
        /// Creates twisted, organic-looking terrain features.
        /// </summary>
        private static float DomainWarpedFBM(float x, float y, float z,
            int octaves, float frequency, float persistence, float lacunarity,
            float warpStrength)
        {
            // Warp offsets sampled at different positions to avoid correlation
            float wx = FBM(x + 0f, y + 0f, z + 0f, 3, frequency * 0.5f, 0.5f, 2f) * warpStrength;
            float wy = FBM(x + 5.2f, y + 1.3f, z + 2.8f, 3, frequency * 0.5f, 0.5f, 2f) * warpStrength;
            float wz = FBM(x + 9.1f, y + 4.7f, z + 7.3f, 3, frequency * 0.5f, 0.5f, 2f) * warpStrength;

            return FBM(x + wx, y + wy, z + wz, octaves, frequency, persistence, lacunarity);
        }

        // ===================================================================
        // SDF Boolean Operations (Inigo Quilez formulas)
        // ===================================================================

        /// <summary>Union: merge two solids. min(a, b).</summary>
        private static float SdfUnion(float a, float b) => a < b ? a : b;

        /// <summary>Subtraction: carve a from b. max(-a, b).</summary>
        private static float SdfSubtract(float a, float b) => (-a) > b ? (-a) : b;

        /// <summary>Intersection: keep only overlap. max(a, b).</summary>
        private static float SdfIntersect(float a, float b) => a > b ? a : b;

        /// <summary>
        /// Smooth union: merge with rounded blend.
        /// k controls blend radius (larger k = smoother transition).
        /// </summary>
        private static float SdfSmoothUnion(float a, float b, float k)
        {
            float h = a - b;
            if (h < 0) h = -h;
            h = 0.5f + 0.5f * (b - a) / k;
            if (h < 0) h = 0;
            if (h > 1) h = 1;
            return Lerp(b, a, h) - k * h * (1f - h);
        }

        /// <summary>
        /// Smooth subtraction: carve with rounded edges.
        /// </summary>
        private static float SdfSmoothSubtract(float a, float b, float k)
        {
            return -SdfSmoothUnion(-a, b, k);
        }

        // ===================================================================
        // Main SDF Evaluation Kernel
        // ===================================================================

        /// <summary>
        /// Evaluate SDF for a chunk of voxels. One thread per voxel.
        ///
        /// Produces terrain with:
        /// - Base elevation from domain-warped fBm
        /// - Cave systems carved via 3D noise subtraction
        /// - Smooth blending between features
        ///
        /// Output: 16-bit fixed-point SDF values (positive = solid, negative = air).
        /// </summary>
        public static void EvaluateSdfKernel(
            Index3D index,
            ArrayView<short> sdfOutput,
            float chunkWorldX,
            float chunkWorldY,
            float chunkWorldZ,
            float voxelSize,
            int chunkSize,
            int seed)
        {
            int x = index.X;
            int y = index.Y;
            int z = index.Z;

            if (x >= chunkSize || y >= chunkSize || z >= chunkSize) return;

            // World-space position of this voxel
            float wx = chunkWorldX + x * voxelSize;
            float wy = chunkWorldY + y * voxelSize;
            float wz = chunkWorldZ + z * voxelSize;

            // Seed offset to make each world unique
            float sx = wx + seed * 0.31415f;
            float sz = wz + seed * 0.27183f;

            // === Layer 1: Base terrain (ground plane + height variation) ===
            // Positive below surface, negative above
            float baseHeight = 64f; // base ground level
            float heightNoise = DomainWarpedFBM(sx, 0, sz, 6, 0.005f, 0.5f, 2f, 20f) * 40f;
            float terrain = (baseHeight + heightNoise) - wy;

            // === Layer 2: Hills ===
            float hills = FBM(sx, 0, sz, 4, 0.02f, 0.5f, 2f) * 15f;
            terrain += hills;

            // === Layer 3: Mountain peaks ===
            float mountainMask = FBM(sx, 0, sz, 2, 0.003f, 0.5f, 2f);
            if (mountainMask > 0.3f)
            {
                float mountainHeight = (mountainMask - 0.3f) * 80f;
                float mountainNoise = FBM(sx, wy * 0.5f, sz, 4, 0.01f, 0.6f, 2.2f) * 10f;
                terrain += mountainHeight + mountainNoise;
            }

            // === Layer 4: Fine surface detail ===
            float detail = FBM(sx, wy, sz, 3, 0.1f, 0.5f, 2f) * 2f;
            terrain += detail;

            // === Layer 5: Cave system (3D noise subtraction) ===
            float caveDensity = FBM(sx * 0.8f, wy * 0.8f, sz * 0.8f, 4, 0.03f, 0.5f, 2f);
            float caveRadius = 4f; // cave passage radius in world units
            if (caveDensity > 0.4f)
            {
                float caveStrength = (caveDensity - 0.4f) * caveRadius * 10f;
                terrain = SdfSmoothSubtract(caveStrength, terrain, 2f);
            }

            // === Layer 6: Overhangs (horizontal 3D noise near surface) ===
            float nearSurface = terrain; // how close to surface
            if (nearSurface > -10f && nearSurface < 10f)
            {
                float overhang = FBM(sx, wy * 2f, sz, 3, 0.05f, 0.5f, 2f) * 3f;
                terrain += overhang;
            }

            // Convert to 16-bit fixed point and clamp
            float clamped = terrain;
            if (clamped > 127f) clamped = 127f;
            if (clamped < -128f) clamped = -128f;
            short fixedPoint = (short)(clamped * SdfChunk.FixedPointScale);

            sdfOutput[x + z * chunkSize + y * chunkSize * chunkSize] = fixedPoint;
        }

        /// <summary>
        /// Modify SDF values in a sphere (terrain deformation tool).
        /// mode: 0 = subtract (dig), 1 = add (fill)
        ///
        /// Dispatch: one thread per voxel in affected region.
        /// </summary>
        public static void ModifySdfSphereKernel(
            Index3D index,
            ArrayView<short> sdfValues,
            float centerX, float centerY, float centerZ,
            float radius,
            int mode,
            float chunkWorldX, float chunkWorldY, float chunkWorldZ,
            float voxelSize,
            int chunkSize)
        {
            int x = index.X;
            int y = index.Y;
            int z = index.Z;

            if (x >= chunkSize || y >= chunkSize || z >= chunkSize) return;

            float wx = chunkWorldX + x * voxelSize;
            float wy = chunkWorldY + y * voxelSize;
            float wz = chunkWorldZ + z * voxelSize;

            // SDF of sphere: distance from center minus radius
            float dx = wx - centerX;
            float dy = wy - centerY;
            float dz = wz - centerZ;
            float dist = dx * dx + dy * dy + dz * dz;

            // Fast sqrt approximation for GPU
            // Only process voxels near the sphere
            if (dist > (radius + 4f) * (radius + 4f)) return;

            // Actual distance
            float sqrtDist = dist;
            // Newton-Raphson sqrt (3 iterations for good precision)
            if (sqrtDist > 0)
            {
                float guess = sqrtDist * 0.5f;
                guess = 0.5f * (guess + sqrtDist / guess);
                guess = 0.5f * (guess + sqrtDist / guess);
                guess = 0.5f * (guess + sqrtDist / guess);
                sqrtDist = guess;
            }

            float sphereSdf = sqrtDist - radius;

            int idx = x + z * chunkSize + y * chunkSize * chunkSize;
            float currentSdf = sdfValues[idx] / SdfChunk.FixedPointScale;

            float newSdf;
            if (mode == 0)
            {
                // Subtract (dig): carve sphere from terrain
                newSdf = SdfSmoothSubtract(-sphereSdf, currentSdf, 1f);
            }
            else
            {
                // Add (fill): merge sphere into terrain
                newSdf = SdfSmoothUnion(sphereSdf, currentSdf, 1f);
            }

            float clamped = newSdf;
            if (clamped > 127f) clamped = 127f;
            if (clamped < -128f) clamped = -128f;
            sdfValues[idx] = (short)(clamped * SdfChunk.FixedPointScale);
        }
    }
}
