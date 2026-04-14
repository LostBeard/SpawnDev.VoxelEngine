using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Greedy merge tests - verify merged quads exactly cover the same faces
    public partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task GreedyMerge_SingleBlock_6Quads() => await RunTest(async accelerator =>
        {
            // Single block produces 6 faces, none can merge -> 6 quads
            int chunkXZ = 8;
            int height = 16;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.SingleBlock(chunkXZ, height);

            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);

            // Save original masks for coverage verification
            var originalMasks = (long[])cpuFaceMasks.Clone();

            var quads = GreedyMergeCpuReference.GreedyMerge(cpuFaceMasks, blocks, chunkXZ, height, paddedXZ);

            if (quads.Count != 6)
                throw new Exception($"Single block should produce 6 quads, got {quads.Count}");

            // Verify each quad is 1x1
            foreach (var q in quads)
            {
                PackedQuad.Unpack(q, out _, out _, out _, out int w, out int h, out _, out _);
                if (w != 1 || h != 1)
                    throw new Exception($"Single block quad should be 1x1, got {w}x{h}");
            }

            // Verify coverage
            var error = GreedyMergeCpuReference.VerifyQuadCoverage(quads, originalMasks, chunkXZ, height);
            if (error != null) throw new Exception($"Coverage error: {error}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task GreedyMerge_FlatTerrain_MergesTopFace() => await RunTest(async accelerator =>
        {
            // Flat terrain of one block type: top face should merge into 1 quad
            int chunkXZ = 8;
            int height = 16;
            int solidHeight = 4;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.FlatTerrain(chunkXZ, height, solidHeight);

            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            var originalMasks = (long[])cpuFaceMasks.Clone();
            int totalFaces = FaceCullCpuReference.CountVisibleFaces(originalMasks);

            var quads = GreedyMergeCpuReference.GreedyMerge(cpuFaceMasks, blocks, chunkXZ, height, paddedXZ);

            // Must produce fewer quads than faces (greedy merge reduces count)
            if (quads.Count >= totalFaces)
                throw new Exception($"Greedy merge should reduce quad count. Faces={totalFaces}, Quads={quads.Count}");

            // Verify coverage is exact
            var error = GreedyMergeCpuReference.VerifyQuadCoverage(quads, originalMasks, chunkXZ, height);
            if (error != null) throw new Exception($"Coverage error: {error}");

            // Count +Y quads at y=solidHeight-1 - should be exactly 1 (fully merged)
            int topQuadCount = 0;
            int topQuadArea = 0;
            foreach (var q in quads)
            {
                PackedQuad.Unpack(q, out _, out int qy, out _, out int w, out int h, out int face, out _);
                if (face == VoxelMeshConstants.FacePosY && qy == solidHeight - 1)
                {
                    topQuadCount++;
                    topQuadArea += w * h;
                }
            }
            // Total area of top face quads must equal chunkXZ * chunkXZ
            int expectedArea = chunkXZ * chunkXZ;
            if (topQuadArea != expectedArea)
                throw new Exception($"Top face area should be {expectedArea}, got {topQuadArea} across {topQuadCount} quads");
            // With one block type, should merge to exactly 1 quad
            if (topQuadCount != 1)
                throw new Exception($"Expected 1 merged top quad, got {topQuadCount}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task GreedyMerge_StripedTerrain_RespectsBlockTypes() => await RunTest(async accelerator =>
        {
            // Alternating block types in X: greedy merge must NOT merge across type boundaries
            int chunkXZ = 8;
            int height = 8;
            int solidHeight = 2;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.StripedTerrain(chunkXZ, height, solidHeight, 1, 2);

            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            var originalMasks = (long[])cpuFaceMasks.Clone();

            var quads = GreedyMergeCpuReference.GreedyMerge(cpuFaceMasks, blocks, chunkXZ, height, paddedXZ);

            // Verify no quad contains mixed block types
            foreach (var q in quads)
            {
                PackedQuad.Unpack(q, out _, out _, out _, out _, out _, out _, out int blockType);
                if (blockType != 1 && blockType != 2)
                    throw new Exception($"Quad has unexpected block type {blockType}, expected 1 or 2");
            }

            // Verify coverage
            var error = GreedyMergeCpuReference.VerifyQuadCoverage(quads, originalMasks, chunkXZ, height);
            if (error != null) throw new Exception($"Coverage error: {error}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task GreedyMerge_Checkerboard_NoMergingPossible() => await RunTest(async accelerator =>
        {
            // Checkerboard: every solid block is surrounded by air on all sides
            // No merging possible -> quad count == face count
            int chunkXZ = 4; // keep small
            int height = 4;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.Checkerboard(chunkXZ, height);

            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            var originalMasks = (long[])cpuFaceMasks.Clone();
            int totalFaces = FaceCullCpuReference.CountVisibleFaces(originalMasks);

            var quads = GreedyMergeCpuReference.GreedyMerge(cpuFaceMasks, blocks, chunkXZ, height, paddedXZ);

            // Each solid block has 6 exposed faces, none can merge
            // quad count should equal face count
            if (quads.Count != totalFaces)
                throw new Exception($"Checkerboard: quads ({quads.Count}) should equal faces ({totalFaces})");

            var error = GreedyMergeCpuReference.VerifyQuadCoverage(quads, originalMasks, chunkXZ, height);
            if (error != null) throw new Exception($"Coverage error: {error}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task GreedyMerge_SolidCube_MaximalMerge() => await RunTest(async accelerator =>
        {
            // Solid cube: each face of the cube should merge into one quad
            // 6 faces total for a solid cube with no internal geometry
            int chunkXZ = 8;
            int height = 8;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.SolidCube(chunkXZ, height);

            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            var originalMasks = (long[])cpuFaceMasks.Clone();
            int totalFaces = FaceCullCpuReference.CountVisibleFaces(originalMasks);

            var quads = GreedyMergeCpuReference.GreedyMerge(cpuFaceMasks, blocks, chunkXZ, height, paddedXZ);

            // Solid cube of one type: each of the 6 exterior faces should merge to exactly 1 quad
            if (quads.Count != 6)
                throw new Exception($"Solid cube should produce 6 quads (one per face), got {quads.Count}. " +
                    $"Total faces before merge: {totalFaces}");

            var error = GreedyMergeCpuReference.VerifyQuadCoverage(quads, originalMasks, chunkXZ, height);
            if (error != null) throw new Exception($"Coverage error: {error}");

            await Task.CompletedTask;
        });
    }
}
