namespace SpawnDev.VoxelEngine.Physics
{
    /// <summary>
    /// Post-destruction structural integrity check.
    ///
    /// After blocks are destroyed (explosion, mining), flood fill upward from ground level
    /// to determine which blocks are still connected to the ground. Blocks not connected
    /// are marked for collapse (become falling entities at the game level).
    ///
    /// Algorithm:
    /// 1. Start from all solid blocks at y=0 (ground level)
    /// 2. BFS upward through connected solid blocks (6-neighbor adjacency)
    /// 3. Any solid block NOT reached by the BFS is unsupported
    /// 4. Return list of unsupported block positions
    ///
    /// Only checks structural blocks (FlagStructural in BlockRegistry).
    /// Non-structural blocks (dirt, sand, leaves) don't require support.
    /// Gravity blocks (sand, gravel) always fall if air below regardless of support.
    /// </summary>
    public static class StructuralIntegrity
    {
        /// <summary>
        /// Find all unsupported blocks in a section.
        /// An unsupported block is a structural block not connected to the ground (y=0) via other structural blocks.
        /// </summary>
        public static List<(int x, int y, int z, int blockType)> FindUnsupported(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            BlockRegistry registry)
        {
            int total = sizeXZ * sizeXZ * sizeY;
            var supported = new bool[total];
            var queue = new Queue<int>();

            // Seed: all structural blocks at ground level (y=0)
            for (int z = 0; z < sizeXZ; z++)
            {
                for (int x = 0; x < sizeXZ; x++)
                {
                    int idx = x + z * sizeXZ; // y=0
                    int packed = blocks[idx];
                    int type = packed & 0xFFF;
                    if (type != 0 && registry.Get(type).IsStructural)
                    {
                        supported[idx] = true;
                        queue.Enqueue(idx);
                    }
                }
            }

            // BFS through connected structural blocks
            while (queue.Count > 0)
            {
                int idx = queue.Dequeue();
                int x = idx % sizeXZ;
                int z = (idx / sizeXZ) % sizeXZ;
                int y = idx / (sizeXZ * sizeXZ);

                TryEnqueue(blocks, sizeXZ, sizeY, x + 1, y, z, supported, queue, registry);
                TryEnqueue(blocks, sizeXZ, sizeY, x - 1, y, z, supported, queue, registry);
                TryEnqueue(blocks, sizeXZ, sizeY, x, y + 1, z, supported, queue, registry);
                TryEnqueue(blocks, sizeXZ, sizeY, x, y - 1, z, supported, queue, registry);
                TryEnqueue(blocks, sizeXZ, sizeY, x, y, z + 1, supported, queue, registry);
                TryEnqueue(blocks, sizeXZ, sizeY, x, y, z - 1, supported, queue, registry);
            }

            // Find unsupported structural blocks
            var unsupported = new List<(int, int, int, int)>();
            for (int y = 0; y < sizeY; y++)
            {
                for (int z = 0; z < sizeXZ; z++)
                {
                    for (int x = 0; x < sizeXZ; x++)
                    {
                        int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                        int packed = blocks[idx];
                        int type = packed & 0xFFF;

                        if (type == 0) continue;

                        var props = registry.Get(type);
                        if (!props.IsStructural) continue;

                        if (!supported[idx])
                        {
                            unsupported.Add((x, y, z, type));
                        }
                    }
                }
            }

            return unsupported;
        }

        /// <summary>
        /// Find blocks that should fall due to gravity (sand, gravel).
        /// A gravity block falls if the block directly below it is air.
        /// Returns positions sorted top-to-bottom (process from top so cascading works).
        /// </summary>
        public static List<(int x, int y, int z, int blockType)> FindGravityFallers(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            BlockRegistry registry)
        {
            var fallers = new List<(int, int, int, int)>();

            // Scan top-to-bottom so cascading gravity works in one pass
            for (int y = sizeY - 1; y >= 1; y--)
            {
                for (int z = 0; z < sizeXZ; z++)
                {
                    for (int x = 0; x < sizeXZ; x++)
                    {
                        int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                        int packed = blocks[idx];
                        int type = packed & 0xFFF;

                        if (type == 0) continue;
                        if (!registry.Get(type).HasGravity) continue;

                        // Check block below
                        int belowIdx = x + z * sizeXZ + (y - 1) * sizeXZ * sizeXZ;
                        int belowType = blocks[belowIdx] & 0xFFF;

                        if (belowType == 0) // air below
                        {
                            fallers.Add((x, y, z, type));
                        }
                    }
                }
            }

            return fallers;
        }

        private static void TryEnqueue(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            int x, int y, int z,
            bool[] supported, Queue<int> queue,
            BlockRegistry registry)
        {
            if ((uint)x >= (uint)sizeXZ || (uint)y >= (uint)sizeY || (uint)z >= (uint)sizeXZ)
                return;

            int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
            if (supported[idx]) return;

            int type = blocks[idx] & 0xFFF;
            if (type == 0) return;
            if (!registry.Get(type).IsStructural) return;

            supported[idx] = true;
            queue.Enqueue(idx);
        }
    }
}
