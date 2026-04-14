using System.Numerics;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Physics
{
    /// <summary>
    /// Sphere-voxel intersection query.
    /// Finds all blocks within a sphere radius from a center point.
    ///
    /// Used for: explosion damage, area-of-effect abilities, proximity detection,
    /// structural integrity checks (find all connected blocks within radius).
    ///
    /// CPU reference for correctness, GPU kernel for batch/large-radius queries.
    /// </summary>
    public static class VoxelSphereQuery
    {
        /// <summary>
        /// Result entry: one block within the sphere.
        /// </summary>
        public struct SphereHit
        {
            public int X, Y, Z;
            public int BlockType;
            public int DamageLevel;
            public float Distance;
        }

        /// <summary>
        /// Find all non-air blocks within radius of center point (CPU reference).
        /// Returns blocks sorted by distance from center.
        /// </summary>
        /// <param name="blocks">Section voxel data (PackedBlock format).</param>
        /// <param name="sizeXZ">Section size XZ.</param>
        /// <param name="sizeY">Section size Y.</param>
        /// <param name="center">Sphere center in section-local space.</param>
        /// <param name="radius">Sphere radius in voxel units.</param>
        /// <param name="blockFilter">Optional filter. Null = all non-air.</param>
        public static List<SphereHit> Query(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            Vector3 center, float radius,
            Func<int, bool>? blockFilter = null)
        {
            float radiusSq = radius * radius;
            var results = new List<SphereHit>();

            // Bounding box of the sphere (clamped to section bounds)
            int minX = Math.Max(0, (int)MathF.Floor(center.X - radius));
            int minY = Math.Max(0, (int)MathF.Floor(center.Y - radius));
            int minZ = Math.Max(0, (int)MathF.Floor(center.Z - radius));
            int maxX = Math.Min(sizeXZ - 1, (int)MathF.Floor(center.X + radius));
            int maxY = Math.Min(sizeY - 1, (int)MathF.Floor(center.Y + radius));
            int maxZ = Math.Min(sizeXZ - 1, (int)MathF.Floor(center.Z + radius));

            for (int y = minY; y <= maxY; y++)
            {
                for (int z = minZ; z <= maxZ; z++)
                {
                    for (int x = minX; x <= maxX; x++)
                    {
                        // Distance from block center to sphere center
                        float dx = (x + 0.5f) - center.X;
                        float dy = (y + 0.5f) - center.Y;
                        float dz = (z + 0.5f) - center.Z;
                        float distSq = dx * dx + dy * dy + dz * dz;

                        if (distSq > radiusSq) continue;

                        int packed = blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ];
                        if (PackedBlock.IsAir(packed)) continue;
                        if (blockFilter != null && !blockFilter(packed)) continue;

                        results.Add(new SphereHit
                        {
                            X = x, Y = y, Z = z,
                            BlockType = PackedBlock.GetType(packed),
                            DamageLevel = PackedBlock.GetDamage(packed),
                            Distance = MathF.Sqrt(distSq),
                        });
                    }
                }
            }

            results.Sort((a, b) => a.Distance.CompareTo(b.Distance));
            return results;
        }

        /// <summary>
        /// Count non-air blocks within radius (faster than full query when you just need the count).
        /// </summary>
        public static int Count(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            Vector3 center, float radius,
            Func<int, bool>? blockFilter = null)
        {
            float radiusSq = radius * radius;
            int count = 0;

            int minX = Math.Max(0, (int)MathF.Floor(center.X - radius));
            int minY = Math.Max(0, (int)MathF.Floor(center.Y - radius));
            int minZ = Math.Max(0, (int)MathF.Floor(center.Z - radius));
            int maxX = Math.Min(sizeXZ - 1, (int)MathF.Floor(center.X + radius));
            int maxY = Math.Min(sizeY - 1, (int)MathF.Floor(center.Y + radius));
            int maxZ = Math.Min(sizeXZ - 1, (int)MathF.Floor(center.Z + radius));

            for (int y = minY; y <= maxY; y++)
                for (int z = minZ; z <= maxZ; z++)
                    for (int x = minX; x <= maxX; x++)
                    {
                        float dx = (x + 0.5f) - center.X;
                        float dy = (y + 0.5f) - center.Y;
                        float dz = (z + 0.5f) - center.Z;
                        if (dx * dx + dy * dy + dz * dz > radiusSq) continue;

                        int packed = blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ];
                        if (PackedBlock.IsAir(packed)) continue;
                        if (blockFilter != null && !blockFilter(packed)) continue;
                        count++;
                    }

            return count;
        }

        /// <summary>
        /// GPU kernel: count blocks within sphere. One thread per block in bounding box.
        /// Result written to atomic counter.
        /// </summary>
        public static void SphereCountKernel(
            Index1D index,
            ArrayView<int> blocks,
            ArrayView<int> counter,
            float centerX, float centerY, float centerZ,
            float radiusSq,
            int sizeXZ, int sizeY,
            int minX, int minY, int minZ,
            int rangeX, int rangeY, int rangeZ)
        {
            // Decode 1D index to 3D position within bounding box
            int rx = index % rangeX;
            int rz = (index / rangeX) % rangeZ;
            int ry = index / (rangeX * rangeZ);

            if (ry >= rangeY) return;

            int x = minX + rx;
            int y = minY + ry;
            int z = minZ + rz;

            if (x >= sizeXZ || y >= sizeY || z >= sizeXZ) return;

            float dx = (x + 0.5f) - centerX;
            float dy = (y + 0.5f) - centerY;
            float dz = (z + 0.5f) - centerZ;
            if (dx * dx + dy * dy + dz * dz > radiusSq) return;

            int packed = blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ];
            if ((packed & 0xFFF) == 0) return; // air

            Atomic.Add(ref counter[0], 1);
        }
    }
}
