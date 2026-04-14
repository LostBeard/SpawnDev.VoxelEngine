namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Shared constants for voxel meshing.
    /// </summary>
    public static class VoxelMeshConstants
    {
        /// <summary>Default chunk size along X and Z axes.</summary>
        public const int DefaultChunkSizeXZ = 16;

        /// <summary>Default chunk height (Y axis).</summary>
        public const int DefaultChunkHeight = 256;

        /// <summary>Padding size for neighbor data (1 voxel border on each side).</summary>
        public const int Padding = 1;

        /// <summary>Air block type (empty).</summary>
        public const int Air = 0;

        /// <summary>Face directions.</summary>
        public const int FacePosX = 0;
        public const int FaceNegX = 1;
        public const int FacePosZ = 2;
        public const int FaceNegZ = 3;
        public const int FacePosY = 4;
        public const int FaceNegY = 5;
        public const int FaceCount = 6;
    }
}
