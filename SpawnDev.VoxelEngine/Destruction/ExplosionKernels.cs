using System.Numerics;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Destruction
{
    /// <summary>
    /// Scalar params for GPU sphere destroy/fill. Packed so the kernel fits
    /// LoadAutoGroupedStreamKernel's Index+14 arity cap.
    /// </summary>
    public struct SphereOpParams
    {
        public float CenterX, CenterY, CenterZ, RadiusSq;
        public int PackedBlock; // destroy ignores; fill writes this PackedBlock value
        public int SizeXZ, SizeY;
        public int MinX, MinY, MinZ;
        public int RangeX, RangeY, RangeZ;
    }

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
            SphereOpParams p)
        {
            int rx = index % p.RangeX;
            int rz = (index / p.RangeX) % p.RangeZ;
            int ry = index / (p.RangeX * p.RangeZ);

            if (ry >= p.RangeY) return;

            int x = p.MinX + rx;
            int y = p.MinY + ry;
            int z = p.MinZ + rz;

            if (x >= p.SizeXZ || y >= p.SizeY || z >= p.SizeXZ) return;

            float dx = (x + 0.5f) - p.CenterX;
            float dy = (y + 0.5f) - p.CenterY;
            float dz = (z + 0.5f) - p.CenterZ;
            if (dx * dx + dy * dy + dz * dz > p.RadiusSq) return;

            int idx = x + z * p.SizeXZ + y * p.SizeXZ * p.SizeXZ;
            int packed = blocks[idx];
            if ((packed & 0xFFF) == 0) return; // already air

            blocks[idx] = 0;
            Atomic.Add(ref destroyedCount[0], 1);
        }

        /// <summary>
        /// GPU kernel: fill air cells within sphere with PackedBlock. Skips non-air.
        /// </summary>
        public static void FillKernel(
            Index1D index,
            ArrayView<int> blocks,
            ArrayView<int> filledCount,
            SphereOpParams p)
        {
            int rx = index % p.RangeX;
            int rz = (index / p.RangeX) % p.RangeZ;
            int ry = index / (p.RangeX * p.RangeZ);

            if (ry >= p.RangeY) return;

            int x = p.MinX + rx;
            int y = p.MinY + ry;
            int z = p.MinZ + rz;

            if (x >= p.SizeXZ || y >= p.SizeY || z >= p.SizeXZ) return;

            float dx = (x + 0.5f) - p.CenterX;
            float dy = (y + 0.5f) - p.CenterY;
            float dz = (z + 0.5f) - p.CenterZ;
            if (dx * dx + dy * dy + dz * dz > p.RadiusSq) return;

            int idx = x + z * p.SizeXZ + y * p.SizeXZ * p.SizeXZ;
            int packed = blocks[idx];
            if ((packed & 0xFFF) != 0) return; // already solid

            blocks[idx] = p.PackedBlock;
            Atomic.Add(ref filledCount[0], 1);
        }

        /// <summary>
        /// CPU: destroy blocks in a byte[] column (LostSpawns / blocky path).
        /// Layout matches MeshChunkColumnAsync: x + z*sizeXZ + y*sizeXZ*sizeXZ.
        /// Coordinates are local to the column (x,z in [0,sizeXZ), y in [0,sizeY)).
        /// </summary>
        public static int DestroyInSphereBytes(
            Span<byte> blocks,
            int sizeXZ, int sizeY,
            float centerX, float centerY, float centerZ,
            float radius)
        {
            float radiusSq = radius * radius;
            int destroyed = 0;
            int minX = Math.Max(0, (int)MathF.Floor(centerX - radius));
            int minY = Math.Max(0, (int)MathF.Floor(centerY - radius));
            int minZ = Math.Max(0, (int)MathF.Floor(centerZ - radius));
            int maxX = Math.Min(sizeXZ - 1, (int)MathF.Floor(centerX + radius));
            int maxY = Math.Min(sizeY - 1, (int)MathF.Floor(centerY + radius));
            int maxZ = Math.Min(sizeXZ - 1, (int)MathF.Floor(centerZ + radius));

            for (int y = minY; y <= maxY; y++)
            for (int z = minZ; z <= maxZ; z++)
            for (int x = minX; x <= maxX; x++)
            {
                float dx = (x + 0.5f) - centerX;
                float dy = (y + 0.5f) - centerY;
                float dz = (z + 0.5f) - centerZ;
                if (dx * dx + dy * dy + dz * dz > radiusSq) continue;
                int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                if (blocks[idx] == 0) continue;
                blocks[idx] = 0;
                destroyed++;
            }
            return destroyed;
        }

        /// <summary>
        /// CPU: fill air cells in a byte[] column within sphere. No overwrite of solid.
        /// </summary>
        public static int FillInSphereBytes(
            Span<byte> blocks,
            int sizeXZ, int sizeY,
            float centerX, float centerY, float centerZ,
            float radius,
            byte blockType)
        {
            if (blockType == 0) return 0;
            float radiusSq = radius * radius;
            int filled = 0;
            int minX = Math.Max(0, (int)MathF.Floor(centerX - radius));
            int minY = Math.Max(0, (int)MathF.Floor(centerY - radius));
            int minZ = Math.Max(0, (int)MathF.Floor(centerZ - radius));
            int maxX = Math.Min(sizeXZ - 1, (int)MathF.Floor(centerX + radius));
            int maxY = Math.Min(sizeY - 1, (int)MathF.Floor(centerY + radius));
            int maxZ = Math.Min(sizeXZ - 1, (int)MathF.Floor(centerZ + radius));

            for (int y = minY; y <= maxY; y++)
            for (int z = minZ; z <= maxZ; z++)
            for (int x = minX; x <= maxX; x++)
            {
                float dx = (x + 0.5f) - centerX;
                float dy = (y + 0.5f) - centerY;
                float dz = (z + 0.5f) - centerZ;
                if (dx * dx + dy * dy + dz * dz > radiusSq) continue;
                int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                if (blocks[idx] != 0) continue;
                blocks[idx] = blockType;
                filled++;
            }
            return filled;
        }

        /// <summary>
        /// Collect section coords touched by a sphere in voxel space (VoxelSize=1, BaseY=0).
        /// Used by LostSpawns dirty remesh queue without constructing a full VoxelEngineConfig.
        /// </summary>
        public static void CollectAffectedSections(
            float voxelCenterX, float voxelCenterY, float voxelCenterZ,
            float voxelRadius,
            int sectionSize,
            HashSet<SectionCoord> into)
        {
            int minSx = (int)MathF.Floor((voxelCenterX - voxelRadius) / sectionSize);
            int maxSx = (int)MathF.Floor((voxelCenterX + voxelRadius) / sectionSize);
            int minSy = (int)MathF.Floor((voxelCenterY - voxelRadius) / sectionSize);
            int maxSy = (int)MathF.Floor((voxelCenterY + voxelRadius) / sectionSize);
            int minSz = (int)MathF.Floor((voxelCenterZ - voxelRadius) / sectionSize);
            int maxSz = (int)MathF.Floor((voxelCenterZ + voxelRadius) / sectionSize);

            for (int sy = minSy; sy <= maxSy; sy++)
                for (int sz = minSz; sz <= maxSz; sz++)
                    for (int sx = minSx; sx <= maxSx; sx++)
                        into.Add(new SectionCoord(sx, sy, sz));
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
            CollectAffectedSections(
                worldCenter.X / vs,
                (worldCenter.Y - config.BaseY) / vs,
                worldCenter.Z / vs,
                worldRadius / vs,
                config.SectionSize,
                affected);
            return affected;
        }
    }
}
