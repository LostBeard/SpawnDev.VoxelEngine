using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// GPU kernels for voxel face culling using bitwise operations.
    /// Processes 64 voxels simultaneously per uint64 column.
    /// </summary>
    public static class FaceCullKernels
    {
        /// <summary>
        /// Builds occupancy mask columns from block data.
        /// Each uint64 represents a column of up to 64 voxels along the Y axis.
        /// Bit k = 1 if the voxel at that Y position is solid (non-air).
        ///
        /// Input blocks are padded: (sizeXZ+2) x (sizeXZ+2) x height, stored as [x + z*paddedXZ + y*paddedXZ*paddedXZ].
        /// Output occupancy is paddedXZ * paddedXZ uint64 columns.
        /// </summary>
        public static void BuildOccupancyKernel(
            Index2D index,
            ArrayView<int> paddedBlocks,
            ArrayView<long> occupancy,
            int paddedXZ,
            int height)
        {
            int x = index.X;
            int z = index.Y;
            if (x >= paddedXZ || z >= paddedXZ) return;

            long column = 0L;
            int stride = paddedXZ * paddedXZ;
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

        /// <summary>
        /// Bitwise face culling kernel. For each interior column pair (a,b),
        /// produces 6 face masks by comparing adjacent columns.
        ///
        /// A face is visible when a solid voxel is adjacent to air.
        /// One AND + one NOT = 64 faces culled simultaneously.
        ///
        /// Output: faceMasks[innerIdx + face * innerCount] for 6 faces.
        /// innerIdx = (a-1) + (b-1) * innerXZ, innerXZ = paddedXZ - 2.
        /// </summary>
        public static void FaceCullKernel(
            Index2D index,
            ArrayView<long> occupancy,
            ArrayView<long> faceMasks,
            int paddedXZ,
            int height)
        {
            int a = index.X;
            int b = index.Y;
            int innerXZ = paddedXZ - 2;

            // Only process interior columns (skip padding border)
            if (a < 1 || a >= paddedXZ - 1 || b < 1 || b >= paddedXZ - 1) return;

            // Mask off padding bits at Y boundaries
            long pMask = height >= 64 ? ~0L : (1L << height) - 1L;

            long col = occupancy[a + b * paddedXZ] & pMask;
            long colPosX = occupancy[(a + 1) + b * paddedXZ];
            long colNegX = occupancy[(a - 1) + b * paddedXZ];
            long colPosZ = occupancy[a + (b + 1) * paddedXZ];
            long colNegZ = occupancy[a + (b - 1) * paddedXZ];

            int innerIdx = (a - 1) + (b - 1) * innerXZ;
            int innerCount = innerXZ * innerXZ;

            // +X face: solid here AND air at +X neighbor
            faceMasks[innerIdx + 0 * innerCount] = col & ~colPosX;
            // -X face: solid here AND air at -X neighbor
            faceMasks[innerIdx + 1 * innerCount] = col & ~colNegX;
            // +Z face: solid here AND air at +Z neighbor
            faceMasks[innerIdx + 2 * innerCount] = col & ~colPosZ;
            // -Z face: solid here AND air at -Z neighbor
            faceMasks[innerIdx + 3 * innerCount] = col & ~colNegZ;
            // +Y face: solid here AND air above (neighbor at y+1, which is bit shifted left)
            faceMasks[innerIdx + 4 * innerCount] = col & ~(col >> 1) & pMask;
            // -Y face: solid here AND air below (neighbor at y-1, which is bit shifted right)
            faceMasks[innerIdx + 5 * innerCount] = col & ~(col << 1) & pMask;
        }
    }
}
