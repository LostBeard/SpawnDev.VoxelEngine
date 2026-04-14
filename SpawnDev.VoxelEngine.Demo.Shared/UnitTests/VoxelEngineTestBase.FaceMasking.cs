using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Face masking tests - verify back-facing groups are correctly identified and skipped
    public partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task FaceMask_CameraAboveChunk_TopVisible() => await RunTest(async accelerator =>
        {
            // Camera directly above chunk center
            int mask = FaceMaskService.GetVisibleFaceMask(0, 0, 0, 0, 100, 0);

            // +Y face should be visible (top faces point up toward camera)
            if (!FaceMaskService.IsFaceVisible(mask, VoxelMeshConstants.FacePosY))
                throw new Exception("+Y should be visible when camera is above");

            // -Y face should NOT be visible (bottom faces point away from camera)
            if (FaceMaskService.IsFaceVisible(mask, VoxelMeshConstants.FaceNegY))
                throw new Exception("-Y should be hidden when camera is above");

            // X and Z faces should be visible on both sides (camera is directly above, edge-on)
            if (!FaceMaskService.IsFaceVisible(mask, VoxelMeshConstants.FacePosX) ||
                !FaceMaskService.IsFaceVisible(mask, VoxelMeshConstants.FaceNegX))
                throw new Exception("X faces should be visible (camera on XZ plane center)");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task FaceMask_CameraInFront_CorrectFacesVisible() => await RunTest(async accelerator =>
        {
            // Camera in front of chunk (+Z direction) and slightly above
            int mask = FaceMaskService.GetVisibleFaceMask(0, 0, 0, 0, 5, 50);

            // -Z face should be visible (front faces point toward camera in -Z)
            // Wait: camera is at +Z from chunk, so chunk-to-camera direction is +Z
            // -Z face normal is (0,0,-1), camera is in +Z direction
            // -Z face points AWAY from camera -> NOT visible
            // +Z face normal is (0,0,+1), same direction as camera -> NOT visible either?
            // No: face is visible if it faces TOWARD camera. +Z face normal points +Z,
            // camera is in +Z from chunk. The face normal and chunk-to-camera are SAME direction.
            // That means we're looking at the BACK of the +Z face. The -Z face points toward us.
            // Actually: we see the -Z face because we're looking at the chunk from +Z side.
            // No wait, let me think again.
            //
            // Camera is at (0,5,50). Chunk center at (0,0,0).
            // Direction from chunk to camera: (0,5,50).
            // +Z face has normal (0,0,1). dot(normal, chunkToCamera) = 50 > 0.
            // This means the +Z face FACES the camera -> visible.
            // -Z face has normal (0,0,-1). dot(normal, chunkToCamera) = -50 < 0.
            // This means the -Z face faces AWAY from camera -> not visible.

            // Correction: FaceMaskService uses chunk-to-camera direction.
            // +Z face visible (faces toward camera at +Z)
            if (!FaceMaskService.IsFaceVisible(mask, VoxelMeshConstants.FacePosZ))
                throw new Exception("+Z should be visible when camera is at +Z from chunk");

            // -Z face NOT visible (faces away from camera)
            if (FaceMaskService.IsFaceVisible(mask, VoxelMeshConstants.FaceNegZ))
                throw new Exception("-Z should be hidden when camera is at +Z from chunk");

            // +Y visible (camera is above)
            if (!FaceMaskService.IsFaceVisible(mask, VoxelMeshConstants.FacePosY))
                throw new Exception("+Y should be visible when camera is above");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task FaceMask_ReducesQuadCount() => await RunTest(async accelerator =>
        {
            // Build a solid cube, get all quads, verify face masking reduces count
            int chunkXZ = 8, height = 8, paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.SolidCube(chunkXZ, height);
            var cpuOcc = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuMasks = FaceCullCpuReference.FaceCull(cpuOcc, paddedXZ, height);
            var quads = GreedyMergeCpuReference.GreedyMerge(cpuMasks, blocks, chunkXZ, height, paddedXZ);

            // Solid cube has 6 quads (one per face)
            if (quads.Count != 6) throw new Exception($"Expected 6 quads for solid cube, got {quads.Count}");

            // Camera above and in front-right: should see +X, +Y, +Z (not -X, -Y, -Z)
            int mask = FaceMaskService.GetVisibleFaceMask(4, 4, 4, 10, 20, 20);
            int visibleQuads = FaceMaskService.CountVisibleQuads(quads, mask);
            int visibleGroups = FaceMaskService.CountVisibleGroups(mask);

            // Should see fewer than all 6 faces
            if (visibleQuads >= 6)
                throw new Exception($"Face masking should reduce visible quads. Mask=0b{mask:B6}, visible={visibleQuads}/6");

            // Camera at (10,20,20) vs chunk center (4,4,4): dx=6, dy=16, dz=16 -> all positive
            // Should see exactly 3 groups: +X, +Y, +Z
            if (visibleGroups != 3)
                throw new Exception($"Camera in +X,+Y,+Z octant should see 3 face groups, got {visibleGroups}. Mask=0b{mask:B6}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task FaceMask_CameraAtChunkCenter_AllVisible() => await RunTest(async accelerator =>
        {
            // Camera exactly at chunk center - all faces potentially visible (edge case)
            int mask = FaceMaskService.GetVisibleFaceMask(10, 10, 10, 10, 10, 10);

            // All 6 groups should be visible (delta is zero on all axes)
            int groups = FaceMaskService.CountVisibleGroups(mask);
            if (groups != 6)
                throw new Exception($"Camera at chunk center: all 6 groups should be visible, got {groups}");

            await Task.CompletedTask;
        });
    }
}
