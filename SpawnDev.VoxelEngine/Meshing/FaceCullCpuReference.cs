namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// CPU reference implementation of face culling for test verification.
    /// Results must match GPU kernel output exactly - if they don't, the kernel is broken.
    /// </summary>
    public static class FaceCullCpuReference
    {
        /// <summary>
        /// Build occupancy mask columns from padded block data.
        /// CPU reference - must produce identical output to BuildOccupancyKernel.
        /// </summary>
        public static long[] BuildOccupancy(int[] paddedBlocks, int paddedXZ, int height)
        {
            var occupancy = new long[paddedXZ * paddedXZ];
            int stride = paddedXZ * paddedXZ;

            for (int z = 0; z < paddedXZ; z++)
            {
                for (int x = 0; x < paddedXZ; x++)
                {
                    long column = 0L;
                    int baseIdx = x + z * paddedXZ;

                    for (int y = 0; y < height && y < 64; y++)
                    {
                        int blockType = paddedBlocks[baseIdx + y * stride];
                        if ((blockType & 0xFFF) != 0) // non-air (mask to 12-bit type)
                        {
                            column |= 1L << y;
                        }
                    }

                    occupancy[x + z * paddedXZ] = column;
                }
            }

            return occupancy;
        }

        /// <summary>
        /// Bitwise face culling from occupancy columns.
        /// CPU reference - must produce identical output to FaceCullKernel.
        /// </summary>
        public static long[] FaceCull(long[] occupancy, int paddedXZ, int height)
        {
            int innerXZ = paddedXZ - 2;
            int innerCount = innerXZ * innerXZ;
            var faceMasks = new long[innerCount * 6];

            long pMask = height >= 64 ? ~0L : (1L << height) - 1L;

            for (int b = 1; b < paddedXZ - 1; b++)
            {
                for (int a = 1; a < paddedXZ - 1; a++)
                {
                    long col = occupancy[a + b * paddedXZ] & pMask;
                    long colPosX = occupancy[(a + 1) + b * paddedXZ];
                    long colNegX = occupancy[(a - 1) + b * paddedXZ];
                    long colPosZ = occupancy[a + (b + 1) * paddedXZ];
                    long colNegZ = occupancy[a + (b - 1) * paddedXZ];

                    int innerIdx = (a - 1) + (b - 1) * innerXZ;

                    faceMasks[innerIdx + 0 * innerCount] = col & ~colPosX;
                    faceMasks[innerIdx + 1 * innerCount] = col & ~colNegX;
                    faceMasks[innerIdx + 2 * innerCount] = col & ~colPosZ;
                    faceMasks[innerIdx + 3 * innerCount] = col & ~colNegZ;
                    faceMasks[innerIdx + 4 * innerCount] = col & ~(col >> 1) & pMask;
                    faceMasks[innerIdx + 5 * innerCount] = col & ~(col << 1) & pMask;
                }
            }

            return faceMasks;
        }

        /// <summary>
        /// Count total visible faces from face masks.
        /// Each set bit = one visible face.
        /// </summary>
        public static int CountVisibleFaces(long[] faceMasks)
        {
            int count = 0;
            for (int i = 0; i < faceMasks.Length; i++)
            {
                count += PopCount(faceMasks[i]);
            }
            return count;
        }

        /// <summary>
        /// Naive face count for verification - count all solid-to-air transitions
        /// by iterating every voxel and checking 6 neighbors.
        /// This is the brute-force reference that the bitwise approach must match.
        /// </summary>
        public static int CountVisibleFacesNaive(int[] paddedBlocks, int paddedXZ, int height)
        {
            int count = 0;
            int stride = paddedXZ * paddedXZ;

            for (int y = 0; y < height; y++)
            {
                for (int z = 1; z < paddedXZ - 1; z++)
                {
                    for (int x = 1; x < paddedXZ - 1; x++)
                    {
                        int idx = x + z * paddedXZ + y * stride;
                        if ((paddedBlocks[idx] & 0xFFF) == 0) continue; // air, skip

                        // Check 6 neighbors (mask to 12-bit type for air check)
                        if (x + 1 < paddedXZ && (paddedBlocks[(x + 1) + z * paddedXZ + y * stride] & 0xFFF) == 0) count++;
                        if (x - 1 >= 0 && (paddedBlocks[(x - 1) + z * paddedXZ + y * stride] & 0xFFF) == 0) count++;
                        if (z + 1 < paddedXZ && (paddedBlocks[x + (z + 1) * paddedXZ + y * stride] & 0xFFF) == 0) count++;
                        if (z - 1 >= 0 && (paddedBlocks[x + (z - 1) * paddedXZ + y * stride] & 0xFFF) == 0) count++;
                        if (y + 1 < height && (paddedBlocks[x + z * paddedXZ + (y + 1) * stride] & 0xFFF) == 0) count++;
                        else if (y + 1 >= height) count++; // top of section = face visible
                        if (y - 1 >= 0 && (paddedBlocks[x + z * paddedXZ + (y - 1) * stride] & 0xFFF) == 0) count++;
                        else if (y - 1 < 0) count++; // bottom of section = face visible
                    }
                }
            }

            return count;
        }

        private static int PopCount(long value)
        {
            // Kernighan's bit counting
            int count = 0;
            long v = value;
            while (v != 0)
            {
                v &= v - 1;
                count++;
            }
            return count;
        }
    }
}
