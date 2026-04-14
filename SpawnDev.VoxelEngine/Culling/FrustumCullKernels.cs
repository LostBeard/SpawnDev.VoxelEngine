using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Culling
{
    /// <summary>
    /// GPU kernel for frustum and fog culling of chunks.
    /// One thread per chunk - tests AABB against 6 frustum planes + cylindrical fog distance.
    /// Surviving chunks are written to a compacted output buffer via atomic counter.
    ///
    /// Run this FIRST in the culling pipeline (cheapest, eliminates most chunks).
    /// </summary>
    public static class FrustumCullKernels
    {
        /// <summary>
        /// Frustum + fog culling kernel.
        /// Tests each chunk's AABB against 6 frustum planes and cylindrical fog distance.
        /// Uses P-vertex/N-vertex optimization (2 corner tests per plane instead of 8).
        ///
        /// Input:
        ///   chunkCenters - [chunkCount * 3] floats: (cx, cy, cz) per chunk
        ///   chunkHalfSizes - [3] floats: half-size of chunk AABB (shared for uniform chunks)
        ///   frustumPlanes - [24] floats: 6 planes * (A, B, C, D)
        ///   fogParams - [4] floats: (cameraX, cameraY, cameraZ, fogDistSq)
        ///
        /// Output:
        ///   visibleIndices - compacted list of surviving chunk indices
        ///   visibleCount - atomic counter of surviving chunks
        /// </summary>
        public static void FrustumFogCullKernel(
            Index1D index,
            ArrayView<float> chunkCenters,
            ArrayView<float> chunkHalfSizes,
            ArrayView<float> frustumPlanes,
            ArrayView<float> fogParams,
            ArrayView<int> visibleIndices,
            ArrayView<int> visibleCount,
            int chunkCount)
        {
            if (index >= chunkCount) return;

            float cx = chunkCenters[index * 3 + 0];
            float cy = chunkCenters[index * 3 + 1];
            float cz = chunkCenters[index * 3 + 2];

            float hx = chunkHalfSizes[0];
            float hy = chunkHalfSizes[1];
            float hz = chunkHalfSizes[2];

            // Fog cull first (cheapest test)
            float camX = fogParams[0];
            float camY = fogParams[1];
            float camZ = fogParams[2];
            float fogDistSq = fogParams[3];

            // Cylindrical fog: horizontal distance (XZ) + separate vertical check
            float dx = cx - camX;
            float dz = cz - camZ;
            float horizontalDistSq = dx * dx + dz * dz;

            if (horizontalDistSq > fogDistSq) return; // Beyond fog distance

            // Vertical check: compare dy^2 against fogDistSq (avoids sqrt in kernel)
            float verticalDist = cy - camY;
            if (verticalDist * verticalDist > fogDistSq) return; // Above/below fog cylinder

            // Frustum cull: P-vertex test against 6 planes
            for (int i = 0; i < 6; i++)
            {
                float a = frustumPlanes[i * 4 + 0];
                float b = frustumPlanes[i * 4 + 1];
                float c = frustumPlanes[i * 4 + 2];
                float d = frustumPlanes[i * 4 + 3];

                // P-vertex: the corner of the AABB most in the direction of the plane normal
                float px = (a >= 0) ? (cx + hx) : (cx - hx);
                float py = (b >= 0) ? (cy + hy) : (cy - hy);
                float pz = (c >= 0) ? (cz + hz) : (cz - hz);

                // If P-vertex is outside this plane, the entire AABB is outside
                if (a * px + b * py + c * pz + d < 0)
                    return; // Culled - fixed in ILGPU v4.9.2-rc.4 (return-in-loop codegen)
            }

            // Survived all tests - add to visible list
            int slot = Atomic.Add(ref visibleCount[0], 1);
            if (slot < visibleIndices.IntLength)
            {
                visibleIndices[slot] = index;
            }
            else
            {
                Atomic.Add(ref visibleCount[0], -1); // Rollback
            }
        }
    }

    /// <summary>
    /// CPU reference implementation and utilities for frustum culling.
    /// </summary>
    public static class FrustumCullCpuReference
    {
        /// <summary>
        /// Extract 6 frustum planes from a combined view-projection matrix.
        /// Gribb-Hartmann method. Returns flat array of 24 floats (6 planes x ABCD).
        /// Matrix is column-major (as used by GPU - GpuMatrix4x4 handles this).
        ///
        /// For row-major .NET Matrix4x4, transpose first or use the row-extraction variant.
        /// </summary>
        public static float[] ExtractFrustumPlanes(float[] viewProjRowMajor)
        {
            // .NET Matrix4x4 is row-major with post-multiply convention (v * M).
            // Gribb-Hartmann for post-multiply extracts from COLUMNS of the matrix
            // (equivalent to rows of the transpose).
            //
            // Row-major layout: m[row*4+col]
            // Column 0: m[0], m[4], m[8], m[12]
            // Column 1: m[1], m[5], m[9], m[13]
            // Column 2: m[2], m[6], m[10], m[14]
            // Column 3: m[3], m[7], m[11], m[15]

            float[] m = viewProjRowMajor;
            var planes = new float[24];

            // Left: col3 + col0
            SetPlane(planes, 0, m[3] + m[0], m[7] + m[4], m[11] + m[8], m[15] + m[12]);
            // Right: col3 - col0
            SetPlane(planes, 1, m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]);
            // Bottom: col3 + col1
            SetPlane(planes, 2, m[3] + m[1], m[7] + m[5], m[11] + m[9], m[15] + m[13]);
            // Top: col3 - col1
            SetPlane(planes, 3, m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]);
            // Near: col3 + col2
            SetPlane(planes, 4, m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]);
            // Far: col3 - col2
            SetPlane(planes, 5, m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]);

            return planes;
        }

        private static void SetPlane(float[] planes, int index, float a, float b, float c, float d)
        {
            // Normalize the plane
            float len = MathF.Sqrt(a * a + b * b + c * c);
            if (len > 0.0001f)
            {
                float invLen = 1.0f / len;
                a *= invLen; b *= invLen; c *= invLen; d *= invLen;
            }
            planes[index * 4 + 0] = a;
            planes[index * 4 + 1] = b;
            planes[index * 4 + 2] = c;
            planes[index * 4 + 3] = d;
        }

        /// <summary>
        /// CPU reference frustum + fog culling. Returns list of visible chunk indices.
        /// Must produce the same set (though not necessarily same order) as the GPU kernel.
        /// </summary>
        public static List<int> CullChunks(
            float[] chunkCenters, float halfX, float halfY, float halfZ,
            float[] frustumPlanes, float cameraX, float cameraY, float cameraZ, float fogDistSq)
        {
            var visible = new List<int>();
            int chunkCount = chunkCenters.Length / 3;

            for (int i = 0; i < chunkCount; i++)
            {
                float cx = chunkCenters[i * 3 + 0];
                float cy = chunkCenters[i * 3 + 1];
                float cz = chunkCenters[i * 3 + 2];

                // Fog cull (cylindrical) - squared comparisons, no sqrt
                float dx = cx - cameraX;
                float dz = cz - cameraZ;
                if (dx * dx + dz * dz > fogDistSq) continue;
                float dy = cy - cameraY;
                if (dy * dy > fogDistSq) continue;

                // Frustum cull
                bool culled = false;
                for (int p = 0; p < 6; p++)
                {
                    float a = frustumPlanes[p * 4 + 0];
                    float b = frustumPlanes[p * 4 + 1];
                    float c = frustumPlanes[p * 4 + 2];
                    float d = frustumPlanes[p * 4 + 3];

                    float px = (a >= 0) ? (cx + halfX) : (cx - halfX);
                    float py = (b >= 0) ? (cy + halfY) : (cy - halfY);
                    float pz = (c >= 0) ? (cz + halfZ) : (cz - halfZ);

                    if (a * px + b * py + c * pz + d < 0)
                    {
                        culled = true;
                        break;
                    }
                }

                if (!culled) visible.Add(i);
            }

            return visible;
        }

        /// <summary>
        /// Build a perspective view-projection matrix for testing.
        /// Uses System.Numerics.Matrix4x4 for correctness.
        /// Returns row-major 16-float array for frustum plane extraction.
        /// </summary>
        public static float[] BuildTestViewProj(
            float cameraX, float cameraY, float cameraZ,
            float lookAtX, float lookAtY, float lookAtZ,
            float fovRadians, float aspect, float near, float far)
        {
            var cameraPos = new System.Numerics.Vector3(cameraX, cameraY, cameraZ);
            var lookAt = new System.Numerics.Vector3(lookAtX, lookAtY, lookAtZ);
            var up = System.Numerics.Vector3.UnitY;

            var view = System.Numerics.Matrix4x4.CreateLookAt(cameraPos, lookAt, up);
            var proj = System.Numerics.Matrix4x4.CreatePerspectiveFieldOfView(fovRadians, aspect, near, far);
            var vp = view * proj;

            // Extract row-major floats
            return new float[]
            {
                vp.M11, vp.M12, vp.M13, vp.M14,
                vp.M21, vp.M22, vp.M23, vp.M24,
                vp.M31, vp.M32, vp.M33, vp.M34,
                vp.M41, vp.M42, vp.M43, vp.M44,
            };
        }
    }
}
