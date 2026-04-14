using System.Numerics;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Destruction
{
    /// <summary>
    /// GPU-accelerated explosion/destruction system.
    ///
    /// DestroyBlocksInSphere: marks all blocks within radius as air (type 0).
    /// GPU kernel: one thread per block in the bounding box, distance check.
    /// Returns list of affected section coordinates for re-meshing.
    ///
    /// Destruction is persistent (DayZ style) - blocks don't respawn.
    /// Structural integrity check (Physics/StructuralIntegrity) runs after
    /// to find unsupported blocks that should collapse.
    /// </summary>
    public static class ExplosionKernels
    {
        /// <summary>
        /// CPU reference: destroy all blocks within sphere radius.
        /// Modifies block data in place. Returns count of blocks destroyed.
        /// </summary>
        public static int DestroyInSphere(
            Span<int> blocks,
            int sizeXZ, int sizeY,
            Vector3 center, float radius,
            float blastPower = 100f,
            BlockRegistry? registry = null)
        {
            float radiusSq = radius * radius;
            int destroyed = 0;

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
                        float dx = (x + 0.5f) - center.X;
                        float dy = (y + 0.5f) - center.Y;
                        float dz = (z + 0.5f) - center.Z;
                        float distSq = dx * dx + dy * dy + dz * dz;

                        if (distSq > radiusSq) continue;

                        int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                        int packed = blocks[idx];
                        int blockType = packed & 0xFFF;
                        if (blockType == 0) continue; // already air

                        // Check blast resistance
                        if (registry != null)
                        {
                            var props = registry.Get(blockType);
                            // Blast power decreases with distance (inverse square)
                            float effectivePower = blastPower / (1f + distSq);
                            if (effectivePower < props.BlastResistance) continue; // survives
                        }

                        blocks[idx] = 0; // destroy -> air
                        destroyed++;
                    }
                }
            }

            return destroyed;
        }

        /// <summary>
        /// Apply damage to blocks within sphere radius without destroying them.
        /// Blocks closer to center take more damage. Used for partial destruction.
        /// </summary>
        public static int DamageInSphere(
            Span<int> blocks,
            int sizeXZ, int sizeY,
            Vector3 center, float radius,
            int maxDamage = 15)
        {
            float radiusSq = radius * radius;
            int affected = 0;

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
                        float dx = (x + 0.5f) - center.X;
                        float dy = (y + 0.5f) - center.Y;
                        float dz = (z + 0.5f) - center.Z;
                        float distSq = dx * dx + dy * dy + dz * dz;

                        if (distSq > radiusSq) continue;

                        int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                        int packed = blocks[idx];
                        int blockType = packed & 0xFFF;
                        if (blockType == 0) continue;

                        // Damage proportional to proximity (closer = more damage)
                        float t = 1f - MathF.Sqrt(distSq) / radius; // 1 at center, 0 at edge
                        int damage = Math.Min(maxDamage, (int)(t * maxDamage) + 1);

                        int currentDamage = PackedBlock.GetDamage(packed);
                        int newDamage = Math.Min(15, currentDamage + damage);

                        if (newDamage >= 15)
                        {
                            blocks[idx] = 0; // fully damaged -> destroyed
                        }
                        else
                        {
                            blocks[idx] = PackedBlock.Pack(blockType, newDamage);
                        }
                        affected++;
                    }
                }
            }

            return affected;
        }

        /// <summary>
        /// GPU kernel: destroy blocks within sphere. One thread per block in bounding box.
        /// </summary>
        public static void DestroyKernel(
            Index1D index,
            ArrayView<int> blocks,
            ArrayView<int> destroyedCount,
            float centerX, float centerY, float centerZ,
            float radiusSq,
            int sizeXZ, int sizeY,
            int minX, int minY, int minZ,
            int rangeX, int rangeY, int rangeZ)
        {
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

            int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
            int packed = blocks[idx];
            if ((packed & 0xFFF) == 0) return; // already air

            blocks[idx] = 0;
            Atomic.Add(ref destroyedCount[0], 1);
        }

        /// <summary>
        /// Find which sections are affected by an explosion.
        /// Returns section coordinates that need re-meshing.
        /// </summary>
        public static HashSet<SectionCoord> GetAffectedSections(
            Vector3 worldCenter, float worldRadius,
            VoxelEngineConfig config)
        {
            var affected = new HashSet<SectionCoord>();
            float vs = config.VoxelSize;
            int ss = config.SectionSize;

            // Convert to voxel space
            float voxelRadius = worldRadius / vs;
            float vx = worldCenter.X / vs;
            float vy = (worldCenter.Y - config.BaseY) / vs;
            float vz = worldCenter.Z / vs;

            // Section range
            int minSx = (int)MathF.Floor((vx - voxelRadius) / ss);
            int maxSx = (int)MathF.Floor((vx + voxelRadius) / ss);
            int minSy = (int)MathF.Floor((vy - voxelRadius) / ss);
            int maxSy = (int)MathF.Floor((vy + voxelRadius) / ss);
            int minSz = (int)MathF.Floor((vz - voxelRadius) / ss);
            int maxSz = (int)MathF.Floor((vz + voxelRadius) / ss);

            for (int sy = minSy; sy <= maxSy; sy++)
                for (int sz = minSz; sz <= maxSz; sz++)
                    for (int sx = minSx; sx <= maxSx; sx++)
                        affected.Add(new SectionCoord(sx, sy, sz));

            return affected;
        }
    }
}
