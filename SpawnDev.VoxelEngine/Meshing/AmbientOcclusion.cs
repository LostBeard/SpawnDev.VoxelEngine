namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Per-vertex ambient occlusion for voxel quads.
    ///
    /// Each quad corner samples 3 neighbor blocks (the corner's "L" shape).
    /// AO level 0 = fully lit (no neighbors), 3 = fully occluded (all 3 neighbors solid).
    ///
    /// The 4 corner AO values (2 bits each = 8 bits total) are packed into
    /// PackedQuad's AO field (bits [39:46]).
    ///
    /// This is the standard voxel AO algorithm used by Minecraft, Teardown,
    /// and virtually every voxel renderer. It's cheap (3 lookups per corner),
    /// visually effective, and requires no extra geometry.
    ///
    /// Reference: 0fps.net/2013/07/03/ambient-occlusion-for-minecraft-like-worlds/
    /// </summary>
    public static class AmbientOcclusion
    {
        /// <summary>
        /// Compute AO for all 4 corners of a quad on a specific face.
        /// Returns packed AO byte: corner0 (bits 0-1) | corner1 (bits 2-3) | corner2 (bits 4-5) | corner3 (bits 6-7).
        /// </summary>
        /// <param name="blocks">Section block data (PackedBlock format, flat array).</param>
        /// <param name="sizeXZ">Section size XZ.</param>
        /// <param name="sizeY">Section size Y.</param>
        /// <param name="bx">Block X position.</param>
        /// <param name="by">Block Y position.</param>
        /// <param name="bz">Block Z position.</param>
        /// <param name="face">Face direction (0-5).</param>
        public static int ComputeQuadAO(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            int bx, int by, int bz, int face)
        {
            // Get the 4 corner AO values for this face
            // Each corner checks 3 neighbors: side1, side2, and the diagonal corner
            int c0, c1, c2, c3;

            switch (face)
            {
                case VoxelMeshConstants.FacePosY: // +Y face, looking down at XZ plane
                    c0 = CornerAO(blocks, sizeXZ, sizeY, bx, by + 1, bz, -1, 0, -1); // -X, -Z
                    c1 = CornerAO(blocks, sizeXZ, sizeY, bx, by + 1, bz, 1, 0, -1);  // +X, -Z
                    c2 = CornerAO(blocks, sizeXZ, sizeY, bx, by + 1, bz, 1, 0, 1);   // +X, +Z
                    c3 = CornerAO(blocks, sizeXZ, sizeY, bx, by + 1, bz, -1, 0, 1);  // -X, +Z
                    break;
                case VoxelMeshConstants.FaceNegY: // -Y face
                    c0 = CornerAO(blocks, sizeXZ, sizeY, bx, by - 1, bz, -1, 0, -1);
                    c1 = CornerAO(blocks, sizeXZ, sizeY, bx, by - 1, bz, 1, 0, -1);
                    c2 = CornerAO(blocks, sizeXZ, sizeY, bx, by - 1, bz, 1, 0, 1);
                    c3 = CornerAO(blocks, sizeXZ, sizeY, bx, by - 1, bz, -1, 0, 1);
                    break;
                case VoxelMeshConstants.FacePosX: // +X face, looking at YZ plane
                    c0 = CornerAO(blocks, sizeXZ, sizeY, bx + 1, by, bz, 0, -1, -1);
                    c1 = CornerAO(blocks, sizeXZ, sizeY, bx + 1, by, bz, 0, 1, -1);
                    c2 = CornerAO(blocks, sizeXZ, sizeY, bx + 1, by, bz, 0, 1, 1);
                    c3 = CornerAO(blocks, sizeXZ, sizeY, bx + 1, by, bz, 0, -1, 1);
                    break;
                case VoxelMeshConstants.FaceNegX: // -X face
                    c0 = CornerAO(blocks, sizeXZ, sizeY, bx - 1, by, bz, 0, -1, -1);
                    c1 = CornerAO(blocks, sizeXZ, sizeY, bx - 1, by, bz, 0, 1, -1);
                    c2 = CornerAO(blocks, sizeXZ, sizeY, bx - 1, by, bz, 0, 1, 1);
                    c3 = CornerAO(blocks, sizeXZ, sizeY, bx - 1, by, bz, 0, -1, 1);
                    break;
                case VoxelMeshConstants.FacePosZ: // +Z face, looking at XY plane
                    c0 = CornerAO(blocks, sizeXZ, sizeY, bx, by, bz + 1, -1, -1, 0);
                    c1 = CornerAO(blocks, sizeXZ, sizeY, bx, by, bz + 1, 1, -1, 0);
                    c2 = CornerAO(blocks, sizeXZ, sizeY, bx, by, bz + 1, 1, 1, 0);
                    c3 = CornerAO(blocks, sizeXZ, sizeY, bx, by, bz + 1, -1, 1, 0);
                    break;
                case VoxelMeshConstants.FaceNegZ: // -Z face
                default:
                    c0 = CornerAO(blocks, sizeXZ, sizeY, bx, by, bz - 1, -1, -1, 0);
                    c1 = CornerAO(blocks, sizeXZ, sizeY, bx, by, bz - 1, 1, -1, 0);
                    c2 = CornerAO(blocks, sizeXZ, sizeY, bx, by, bz - 1, 1, 1, 0);
                    c3 = CornerAO(blocks, sizeXZ, sizeY, bx, by, bz - 1, -1, 1, 0);
                    break;
            }

            return (c0 & 3) | ((c1 & 3) << 2) | ((c2 & 3) << 4) | ((c3 & 3) << 6);
        }

        /// <summary>
        /// Compute AO for one corner of a face.
        /// Checks 3 neighbors: two sides and the diagonal.
        ///
        /// The two axis offsets (d1, d2) define which "L" shape to check.
        /// One offset is always 0 (the face normal direction is already accounted for).
        ///
        /// Returns 0 (fully lit) to 3 (fully occluded).
        /// </summary>
        private static int CornerAO(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            int cx, int cy, int cz,
            int dx, int dy, int dz)
        {
            // The two side neighbors and the diagonal corner
            bool side1, side2, corner;

            // For each axis that has a non-zero offset, check that neighbor
            // Side 1: offset on first non-zero axis only
            // Side 2: offset on second non-zero axis only
            // Corner: offset on both non-zero axes
            if (dx != 0 && dy != 0)
            {
                side1 = IsSolid(blocks, sizeXZ, sizeY, cx + dx, cy, cz);
                side2 = IsSolid(blocks, sizeXZ, sizeY, cx, cy + dy, cz);
                corner = IsSolid(blocks, sizeXZ, sizeY, cx + dx, cy + dy, cz);
            }
            else if (dx != 0 && dz != 0)
            {
                side1 = IsSolid(blocks, sizeXZ, sizeY, cx + dx, cy, cz);
                side2 = IsSolid(blocks, sizeXZ, sizeY, cx, cy, cz + dz);
                corner = IsSolid(blocks, sizeXZ, sizeY, cx + dx, cy, cz + dz);
            }
            else // dy != 0 && dz != 0
            {
                side1 = IsSolid(blocks, sizeXZ, sizeY, cx, cy + dy, cz);
                side2 = IsSolid(blocks, sizeXZ, sizeY, cx, cy, cz + dz);
                corner = IsSolid(blocks, sizeXZ, sizeY, cx, cy + dy, cz + dz);
            }

            // Standard voxel AO formula
            if (side1 && side2) return 3; // both sides block, corner is irrelevant
            return (side1 ? 1 : 0) + (side2 ? 1 : 0) + (corner ? 1 : 0);
        }

        /// <summary>Check if a block position is solid (non-air and in bounds, or out-of-bounds = solid).</summary>
        private static bool IsSolid(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY, int x, int y, int z)
        {
            // Out of bounds = treat as solid (AO at section edges)
            if ((uint)x >= (uint)sizeXZ || (uint)y >= (uint)sizeY || (uint)z >= (uint)sizeXZ)
                return true;

            return (blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ] & 0xFFF) != 0;
        }

        /// <summary>
        /// Convert AO level (0-3) to a light multiplier for vertex coloring.
        /// 0 = 1.0 (fully lit), 3 = 0.4 (heavily shadowed).
        /// </summary>
        public static float AOToLightMultiplier(int aoLevel)
        {
            return aoLevel switch
            {
                0 => 1.0f,
                1 => 0.8f,
                2 => 0.6f,
                3 => 0.4f,
                _ => 1.0f,
            };
        }

        /// <summary>
        /// Determine if a quad's triangulation should be flipped to avoid
        /// AO interpolation artifacts. When diagonal corners have different AO,
        /// the quad should be split along the other diagonal.
        ///
        /// Standard fix: if c0+c2 > c1+c3, flip the triangulation.
        /// </summary>
        public static bool ShouldFlipQuad(int packedAO)
        {
            int c0 = packedAO & 3;
            int c1 = (packedAO >> 2) & 3;
            int c2 = (packedAO >> 4) & 3;
            int c3 = (packedAO >> 6) & 3;
            return (c0 + c2) > (c1 + c3);
        }
    }
}
