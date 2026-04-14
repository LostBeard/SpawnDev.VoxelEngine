using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.LOD
{
    /// <summary>
    /// LOD reduction kernel using DelegateSpecialization.
    /// One mesh kernel handles all LOD levels - the LOD factor is a compile-time parameter
    /// via DelegateSpecialization, so the reduction logic is fully inlined.
    ///
    /// Reduction: NxNxN group of voxels -> one super-block.
    /// Super-block type = most common non-air type in the group.
    /// Super-block damage = maximum damage in the group (worst visible damage).
    ///
    /// Section-based: 16x16x16 -> 8x8x8 (LOD 1) -> 4x4x4 (LOD 2) -> 2x2x2 (LOD 3)
    /// </summary>
    public static class LODReducer
    {
        /// <summary>
        /// CPU reference: reduce a section's block data to a lower LOD level.
        /// </summary>
        /// <param name="blocks">Source section block data (PackedBlock format).</param>
        /// <param name="sizeXZ">Source section size XZ.</param>
        /// <param name="sizeY">Source section size Y.</param>
        /// <param name="lodLevel">LOD level (1 = 2x reduction, 2 = 4x, 3 = 8x).</param>
        /// <returns>Reduced block data at the target LOD resolution.</returns>
        public static int[] Reduce(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY, int lodLevel)
        {
            int factor = 1 << lodLevel; // 2, 4, or 8
            int outSizeXZ = sizeXZ / factor;
            int outSizeY = sizeY / factor;

            if (outSizeXZ < 1 || outSizeY < 1)
                return new int[] { GetDominantType(blocks, 0, 0, 0, sizeXZ, sizeXZ, sizeY) };

            var output = new int[outSizeXZ * outSizeXZ * outSizeY];

            for (int oy = 0; oy < outSizeY; oy++)
            {
                for (int oz = 0; oz < outSizeXZ; oz++)
                {
                    for (int ox = 0; ox < outSizeXZ; ox++)
                    {
                        int srcX = ox * factor;
                        int srcY = oy * factor;
                        int srcZ = oz * factor;

                        output[ox + oz * outSizeXZ + oy * outSizeXZ * outSizeXZ] =
                            GetDominantType(blocks, srcX, srcY, srcZ, factor, sizeXZ, sizeY);
                    }
                }
            }

            return output;
        }

        /// <summary>
        /// Find the most common non-air block type in a cubic region.
        /// Returns PackedBlock with dominant type and max damage.
        /// </summary>
        private static int GetDominantType(
            ReadOnlySpan<int> blocks,
            int startX, int startY, int startZ,
            int groupSize, int sizeXZ, int sizeY)
        {
            // Count occurrences of each type (sparse - most groups have 1-3 types)
            Span<int> typeCounts = stackalloc int[8]; // track up to 8 types
            Span<int> typeIds = stackalloc int[8];
            int uniqueTypes = 0;
            int maxDamage = 0;

            for (int dy = 0; dy < groupSize && startY + dy < sizeY; dy++)
            {
                for (int dz = 0; dz < groupSize && startZ + dz < sizeXZ; dz++)
                {
                    for (int dx = 0; dx < groupSize && startX + dx < sizeXZ; dx++)
                    {
                        int x = startX + dx;
                        int y = startY + dy;
                        int z = startZ + dz;
                        int packed = blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ];
                        int type = PackedBlock.GetType(packed);
                        int damage = PackedBlock.GetDamage(packed);

                        if (type == 0) continue; // skip air

                        maxDamage = Math.Max(maxDamage, damage);

                        // Find or add type
                        bool found = false;
                        for (int i = 0; i < uniqueTypes; i++)
                        {
                            if (typeIds[i] == type)
                            {
                                typeCounts[i]++;
                                found = true;
                                break;
                            }
                        }
                        if (!found && uniqueTypes < 8)
                        {
                            typeIds[uniqueTypes] = type;
                            typeCounts[uniqueTypes] = 1;
                            uniqueTypes++;
                        }
                    }
                }
            }

            if (uniqueTypes == 0) return 0; // all air

            // Find most common type
            int bestType = typeIds[0];
            int bestCount = typeCounts[0];
            for (int i = 1; i < uniqueTypes; i++)
            {
                if (typeCounts[i] > bestCount)
                {
                    bestType = typeIds[i];
                    bestCount = typeCounts[i];
                }
            }

            return PackedBlock.Pack(bestType, maxDamage);
        }

        /// <summary>
        /// GPU kernel: reduce blocks. One thread per output voxel.
        /// Uses DelegateSpecialization for the LOD factor.
        /// </summary>
        public static void ReduceKernel(
            Index1D index,
            ArrayView<int> input,
            ArrayView<int> output,
            int inSizeXZ, int inSizeY,
            int outSizeXZ, int outSizeY,
            int factor)
        {
            int ox = index % outSizeXZ;
            int oz = (index / outSizeXZ) % outSizeXZ;
            int oy = index / (outSizeXZ * outSizeXZ);

            if (oy >= outSizeY) return;

            int srcX = ox * factor;
            int srcY = oy * factor;
            int srcZ = oz * factor;

            // Find dominant type in the group (inline version for GPU)
            int bestType = 0;
            int bestCount = 0;
            int maxDamage = 0;

            // Simple approach: track the first non-air type encountered
            // and count how many match. For GPU simplicity, use "first wins" tiebreaker.
            int candidateType = 0;
            int candidateCount = 0;

            for (int dy = 0; dy < factor; dy++)
            {
                int y = srcY + dy;
                if (y >= inSizeY) break;
                for (int dz = 0; dz < factor; dz++)
                {
                    int z = srcZ + dz;
                    if (z >= inSizeXZ) break;
                    for (int dx = 0; dx < factor; dx++)
                    {
                        int x = srcX + dx;
                        if (x >= inSizeXZ) break;

                        int packed = input[x + z * inSizeXZ + y * inSizeXZ * inSizeXZ];
                        int type = packed & 0xFFF;
                        int damage = (packed >> 12) & 0xF;

                        if (type == 0) continue;

                        if (damage > maxDamage) maxDamage = damage;

                        if (candidateType == 0)
                        {
                            candidateType = type;
                            candidateCount = 1;
                        }
                        else if (type == candidateType)
                        {
                            candidateCount++;
                        }
                        else
                        {
                            // Different type - if more frequent, switch candidate
                            candidateCount--;
                            if (candidateCount <= 0)
                            {
                                candidateType = type;
                                candidateCount = 1;
                            }
                        }
                    }
                }
            }

            int result = candidateType == 0 ? 0 : (candidateType & 0xFFF) | ((maxDamage & 0xF) << 12);
            output[ox + oz * outSizeXZ + oy * outSizeXZ * outSizeXZ] = result;
        }
    }
}
