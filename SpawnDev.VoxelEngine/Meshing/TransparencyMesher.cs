using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Separate mesh pass for transparent and translucent blocks.
    ///
    /// Transparent blocks (glass) need face emission rules different from opaque:
    /// - Glass-glass: no face emitted (same as opaque-opaque)
    /// - Glass-air: face emitted (same as opaque-air)
    /// - Glass-opaque: face emitted on the glass side (different from opaque-opaque)
    ///
    /// Translucent blocks (water) additionally need:
    /// - Back-to-front draw order for correct alpha blending
    /// - Dual-sided faces for underwater viewing
    ///
    /// The transparent mesh is stored separately from the opaque mesh.
    /// During rendering: draw opaque front-to-back, then transparent back-to-front.
    /// Same RadixSort keys, reversed iteration order.
    /// </summary>
    public static class TransparencyMesher
    {
        /// <summary>
        /// Build occupancy masks for transparent blocks in a section.
        /// Same algorithm as FaceCullKernels but with different face emission rules.
        ///
        /// A transparent face is emitted when:
        /// - The block is transparent/translucent AND the neighbor is air OR a different type
        /// </summary>
        public static long[] BuildTransparentFaceMasks(
            ReadOnlySpan<int> paddedBlocks,
            int sectionXZ, int height, int paddedXZ,
            BlockRegistry registry)
        {
            int innerCount = sectionXZ * sectionXZ;
            var faceMasks = new long[6 * innerCount];
            int stride = paddedXZ * paddedXZ;

            for (int face = 0; face < 6; face++)
            {
                int dx = 0, dy = 0, dz = 0;
                switch (face)
                {
                    case VoxelMeshConstants.FacePosX: dx = 1; break;
                    case VoxelMeshConstants.FaceNegX: dx = -1; break;
                    case VoxelMeshConstants.FacePosZ: dz = 1; break;
                    case VoxelMeshConstants.FaceNegZ: dz = -1; break;
                    case VoxelMeshConstants.FacePosY: dy = 1; break;
                    case VoxelMeshConstants.FaceNegY: dy = -1; break;
                }

                for (int iz = 0; iz < sectionXZ; iz++)
                {
                    for (int ix = 0; ix < sectionXZ; ix++)
                    {
                        int innerIdx = ix + iz * sectionXZ;
                        long mask = 0;

                        for (int iy = 0; iy < height; iy++)
                        {
                            // Padded coordinates (offset by 1 for neighbor border)
                            int px = ix + 1;
                            int py = iy;
                            int pz = iz + 1;

                            int blockPacked = paddedBlocks[px + pz * paddedXZ + py * stride];
                            int blockType = blockPacked & 0xFFF;

                            // Skip air and opaque blocks
                            if (blockType == 0) continue;
                            var blockProps = registry.Get(blockType);
                            if (blockProps.IsOpaque) continue;

                            // Check neighbor in face direction
                            int nx = px + dx;
                            int ny = py + dy;
                            int nz = pz + dz;

                            int neighborPacked = paddedBlocks[nx + nz * paddedXZ + ny * stride];
                            int neighborType = neighborPacked & 0xFFF;

                            // Emit face if neighbor is air or a different block type
                            bool emit = neighborType == 0 || neighborType != blockType;

                            // Also emit if neighbor is opaque (glass next to stone shows glass face)
                            if (!emit && neighborType != 0)
                            {
                                var neighborProps = registry.Get(neighborType);
                                if (neighborProps.IsOpaque)
                                    emit = true;
                            }

                            if (emit)
                                mask |= (1L << iy);
                        }

                        faceMasks[face * innerCount + innerIdx] = mask;
                    }
                }
            }

            return faceMasks;
        }

        /// <summary>
        /// Check if a section contains any transparent/translucent blocks.
        /// Quick pre-check to skip transparent meshing for fully opaque sections.
        /// </summary>
        public static bool HasTransparentBlocks(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY, BlockRegistry registry)
        {
            int total = sizeXZ * sizeXZ * sizeY;
            for (int i = 0; i < total; i++)
            {
                int type = blocks[i] & 0xFFF;
                if (type == 0) continue;
                var props = registry.Get(type);
                if (!props.IsOpaque) return true;
            }
            return false;
        }

        /// <summary>
        /// Check if a section contains any water (translucent) blocks.
        /// Used to determine if back-to-front sorting is needed.
        /// </summary>
        public static bool HasTranslucentBlocks(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY, BlockRegistry registry)
        {
            int total = sizeXZ * sizeXZ * sizeY;
            for (int i = 0; i < total; i++)
            {
                int type = blocks[i] & 0xFFF;
                if (type == 0) continue;
                if (registry.Get(type).IsTranslucent) return true;
            }
            return false;
        }
    }
}
