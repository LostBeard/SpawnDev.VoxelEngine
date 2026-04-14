namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Shared constants for voxel meshing.
    /// The engine operates on 16x16x16 SECTIONS, not full columns.
    /// A chunk column is divided into sections vertically.
    /// </summary>
    public static class VoxelMeshConstants
    {
        /// <summary>Section size along all 3 axes (cubic sections).</summary>
        public const int SectionSize = 16;

        /// <summary>Default chunk size along X and Z axes (matches section size).</summary>
        public const int DefaultChunkSizeXZ = 16;

        /// <summary>Default section height (Y axis). Always 16 for cubic sections.</summary>
        public const int DefaultSectionHeight = 16;

        /// <summary>Maximum section height for face mask columns (64 bits per uint64).</summary>
        public const int MaxSectionHeight = 64;

        /// <summary>Padding size for neighbor data (1 voxel border on each side).</summary>
        public const int Padding = 1;

        /// <summary>Air block type (empty). Uses PackedBlock encoding (type 0).</summary>
        public const int Air = 0;

        /// <summary>Maximum block types supported (12-bit PackedBlock type field).</summary>
        public const int MaxBlockTypes = PackedBlock.MaxBlockTypes; // 4096

        /// <summary>Face directions.</summary>
        public const int FacePosX = 0;
        public const int FaceNegX = 1;
        public const int FacePosZ = 2;
        public const int FaceNegZ = 3;
        public const int FacePosY = 4;
        public const int FaceNegY = 5;
        public const int FaceCount = 6;

        /// <summary>
        /// Face normal vectors (used for face masking / back-face culling).
        /// Index matches face direction constants above.
        /// </summary>
        public static readonly (float nx, float ny, float nz)[] FaceNormals = new[]
        {
            ( 1f,  0f,  0f), // +X
            (-1f,  0f,  0f), // -X
            ( 0f,  0f,  1f), // +Z
            ( 0f,  0f, -1f), // -Z
            ( 0f,  1f,  0f), // +Y
            ( 0f, -1f,  0f), // -Y
        };
    }
}
