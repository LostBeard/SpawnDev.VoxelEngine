namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Generates test chunk data for unit tests.
    /// Produces padded block arrays in the format expected by the meshing kernels.
    /// </summary>
    public static class TestChunkGenerator
    {
        /// <summary>
        /// Generate a flat terrain chunk - solid up to given height, air above.
        /// Returns padded array: (chunkXZ+2) x (chunkXZ+2) x height.
        /// Border padding is air (simulates chunk at world edge).
        /// </summary>
        public static int[] FlatTerrain(int chunkXZ, int height, int solidHeight, int blockType = 1)
        {
            int paddedXZ = chunkXZ + 2;
            int stride = paddedXZ * paddedXZ;
            var blocks = new int[stride * height];

            for (int y = 0; y < solidHeight && y < height; y++)
            {
                for (int z = 1; z <= chunkXZ; z++)
                {
                    for (int x = 1; x <= chunkXZ; x++)
                    {
                        blocks[x + z * paddedXZ + y * stride] = blockType;
                    }
                }
            }

            return blocks;
        }

        /// <summary>
        /// Generate a solid cube chunk - every interior voxel is solid.
        /// Padding border is air.
        /// </summary>
        public static int[] SolidCube(int chunkXZ, int height, int blockType = 1)
        {
            return FlatTerrain(chunkXZ, height, height, blockType);
        }

        /// <summary>
        /// Generate a checkerboard chunk - alternating solid and air in 3D.
        /// Worst case for greedy meshing (no merging possible).
        /// </summary>
        public static int[] Checkerboard(int chunkXZ, int height, int blockType = 1)
        {
            int paddedXZ = chunkXZ + 2;
            int stride = paddedXZ * paddedXZ;
            var blocks = new int[stride * height];

            for (int y = 0; y < height; y++)
            {
                for (int z = 1; z <= chunkXZ; z++)
                {
                    for (int x = 1; x <= chunkXZ; x++)
                    {
                        if ((x + y + z) % 2 == 0)
                        {
                            blocks[x + z * paddedXZ + y * stride] = blockType;
                        }
                    }
                }
            }

            return blocks;
        }

        /// <summary>
        /// Generate a single block at the center of the chunk.
        /// Simplest non-trivial test case - should produce exactly 6 faces.
        /// </summary>
        public static int[] SingleBlock(int chunkXZ, int height, int blockType = 1)
        {
            int paddedXZ = chunkXZ + 2;
            int stride = paddedXZ * paddedXZ;
            var blocks = new int[stride * height];

            int cx = chunkXZ / 2 + 1; // center in padded coords
            int cz = chunkXZ / 2 + 1;
            int cy = height / 2;

            blocks[cx + cz * paddedXZ + cy * stride] = blockType;

            return blocks;
        }

        /// <summary>
        /// Generate a hollow box - solid shell with air interior.
        /// Tests that interior faces are correctly hidden.
        /// </summary>
        public static int[] HollowBox(int chunkXZ, int height, int wallThickness = 1, int blockType = 1)
        {
            int paddedXZ = chunkXZ + 2;
            int stride = paddedXZ * paddedXZ;
            var blocks = new int[stride * height];

            for (int y = 0; y < height; y++)
            {
                for (int z = 1; z <= chunkXZ; z++)
                {
                    for (int x = 1; x <= chunkXZ; x++)
                    {
                        // Interior coords (0-based within chunk)
                        int ix = x - 1;
                        int iz = z - 1;

                        bool isWall = ix < wallThickness || ix >= chunkXZ - wallThickness
                                   || iz < wallThickness || iz >= chunkXZ - wallThickness
                                   || y < wallThickness || y >= height - wallThickness;

                        if (isWall)
                        {
                            blocks[x + z * paddedXZ + y * stride] = blockType;
                        }
                    }
                }
            }

            return blocks;
        }

        /// <summary>
        /// Generate terrain with two different block types in horizontal stripes.
        /// Tests that greedy meshing respects block type boundaries.
        /// </summary>
        public static int[] StripedTerrain(int chunkXZ, int height, int solidHeight, int type1 = 1, int type2 = 2)
        {
            int paddedXZ = chunkXZ + 2;
            int stride = paddedXZ * paddedXZ;
            var blocks = new int[stride * height];

            for (int y = 0; y < solidHeight && y < height; y++)
            {
                for (int z = 1; z <= chunkXZ; z++)
                {
                    for (int x = 1; x <= chunkXZ; x++)
                    {
                        blocks[x + z * paddedXZ + y * stride] = (x % 2 == 0) ? type1 : type2;
                    }
                }
            }

            return blocks;
        }
    }
}
