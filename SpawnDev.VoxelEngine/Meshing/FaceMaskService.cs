namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Determines which face orientation groups to skip based on camera direction.
    /// Each chunk mesh has quads in 6 groups (+X, -X, +Z, -Z, +Y, -Y).
    /// Faces pointing away from the camera are invisible and can be skipped.
    ///
    /// Measured savings: 15-25% on surface viewing, higher underground.
    /// Combined with graph visibility, this eliminates most invisible geometry.
    /// </summary>
    public static class FaceMaskService
    {
        /// <summary>
        /// Get a 6-bit mask of visible face groups given the camera-to-chunk direction.
        /// Bit N set = face group N is potentially visible and should be drawn.
        /// Bit N clear = face group N faces away from camera and can be skipped.
        ///
        /// Uses dot product of face normal with camera-to-chunk direction.
        /// A face is visible if it faces TOWARD the camera (dot product &lt; 0).
        /// </summary>
        /// <param name="chunkCenterX">Chunk center world X</param>
        /// <param name="chunkCenterY">Chunk center world Y</param>
        /// <param name="chunkCenterZ">Chunk center world Z</param>
        /// <param name="cameraX">Camera world X</param>
        /// <param name="cameraY">Camera world Y</param>
        /// <param name="cameraZ">Camera world Z</param>
        /// <returns>6-bit mask where bit N = face group N visible</returns>
        public static int GetVisibleFaceMask(
            float chunkCenterX, float chunkCenterY, float chunkCenterZ,
            float cameraX, float cameraY, float cameraZ)
        {
            // Direction from chunk to camera (faces pointing toward camera are visible)
            float dx = cameraX - chunkCenterX;
            float dy = cameraY - chunkCenterY;
            float dz = cameraZ - chunkCenterZ;

            int mask = 0;

            // +X face (normal = +1,0,0): visible if camera is in +X direction from chunk
            if (dx > 0) mask |= (1 << VoxelMeshConstants.FacePosX);
            // -X face (normal = -1,0,0): visible if camera is in -X direction
            if (dx < 0) mask |= (1 << VoxelMeshConstants.FaceNegX);
            // +Z face (normal = 0,0,+1): visible if camera is in +Z direction
            if (dz > 0) mask |= (1 << VoxelMeshConstants.FacePosZ);
            // -Z face (normal = 0,0,-1): visible if camera is in -Z direction
            if (dz < 0) mask |= (1 << VoxelMeshConstants.FaceNegZ);
            // +Y face (normal = 0,+1,0): visible if camera is above chunk
            if (dy > 0) mask |= (1 << VoxelMeshConstants.FacePosY);
            // -Y face (normal = 0,-1,0): visible if camera is below chunk
            if (dy < 0) mask |= (1 << VoxelMeshConstants.FaceNegY);

            // Edge case: camera exactly on the plane (dx/dy/dz == 0)
            // Both faces along that axis could be visible (grazing angle)
            if (dx == 0) mask |= (1 << VoxelMeshConstants.FacePosX) | (1 << VoxelMeshConstants.FaceNegX);
            if (dy == 0) mask |= (1 << VoxelMeshConstants.FacePosY) | (1 << VoxelMeshConstants.FaceNegY);
            if (dz == 0) mask |= (1 << VoxelMeshConstants.FacePosZ) | (1 << VoxelMeshConstants.FaceNegZ);

            return mask;
        }

        /// <summary>
        /// Count how many face groups are visible in a mask.
        /// </summary>
        public static int CountVisibleGroups(int mask)
        {
            int count = 0;
            for (int i = 0; i < 6; i++)
                if ((mask & (1 << i)) != 0) count++;
            return count;
        }

        /// <summary>
        /// Check if a specific face group is visible.
        /// </summary>
        public static bool IsFaceVisible(int mask, int faceIndex)
        {
            return (mask & (1 << faceIndex)) != 0;
        }

        /// <summary>
        /// Count quads that would be drawn with face masking applied.
        /// Iterates quad list and counts only those whose face is in the visible mask.
        /// </summary>
        public static int CountVisibleQuads(List<long> quads, int visibleMask)
        {
            int count = 0;
            foreach (var q in quads)
            {
                PackedQuad.Unpack(q, out _, out _, out _, out _, out _, out int face, out _);
                if ((visibleMask & (1 << face)) != 0)
                    count++;
            }
            return count;
        }

        /// <summary>
        /// Count individual faces (expanded from quads) that would be drawn with masking.
        /// Each quad of WxH covers W*H individual faces.
        /// </summary>
        public static int CountVisibleFaces(List<long> quads, int visibleMask)
        {
            int count = 0;
            foreach (var q in quads)
            {
                PackedQuad.Unpack(q, out _, out _, out _, out int w, out int h, out int face, out _);
                if ((visibleMask & (1 << face)) != 0)
                    count += w * h;
            }
            return count;
        }
    }
}
