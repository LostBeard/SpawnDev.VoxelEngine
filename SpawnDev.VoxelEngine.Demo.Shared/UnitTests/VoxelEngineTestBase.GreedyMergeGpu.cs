using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    /// <summary>
    /// GPU greedy merge tests - run the FULL pipeline on GPU
    /// (occupancy -> face cull -> greedy merge) and verify against CPU reference.
    ///
    /// These tests prove the GPU kernel produces correct merged quads on real hardware.
    /// GPU and CPU may produce quads in different order, so we compare:
    /// 1. Both cover the exact same set of faces (VerifyQuadCoverage on both)
    /// 2. Total face coverage matches (same total face count)
    /// </summary>
    public partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task GreedyMergeGpu_SingleBlock_CoverageMatchesCpu() => await RunTest(async accelerator =>
        {
            await VerifyGpuGreedyMerge(accelerator, TestChunkGenerator.SingleBlock(8, 16), 8, 16);
        });

        [TestMethod]
        public async Task GreedyMergeGpu_FlatTerrain_CoverageMatchesCpu() => await RunTest(async accelerator =>
        {
            await VerifyGpuGreedyMerge(accelerator, TestChunkGenerator.FlatTerrain(8, 16, 4), 8, 16);
        });

        [TestMethod]
        public async Task GreedyMergeGpu_SolidCube_CoverageMatchesCpu() => await RunTest(async accelerator =>
        {
            await VerifyGpuGreedyMerge(accelerator, TestChunkGenerator.SolidCube(8, 8), 8, 8);
        });

        [TestMethod]
        public async Task GreedyMergeGpu_Checkerboard_CoverageMatchesCpu() => await RunTest(async accelerator =>
        {
            await VerifyGpuGreedyMerge(accelerator, TestChunkGenerator.Checkerboard(4, 4), 4, 4);
        });

        [TestMethod]
        public async Task GreedyMergeGpu_StripedTerrain_CoverageMatchesCpu() => await RunTest(async accelerator =>
        {
            await VerifyGpuGreedyMerge(accelerator, TestChunkGenerator.StripedTerrain(8, 8, 2, 1, 2), 8, 8);
        });

        [TestMethod]
        public async Task GreedyMergeGpu_HollowBox_CoverageMatchesCpu() => await RunTest(async accelerator =>
        {
            await VerifyGpuGreedyMerge(accelerator, TestChunkGenerator.HollowBox(8, 8, 2), 8, 8);
        });

        [TestMethod]
        public async Task GreedyMergeGpu_DamagedBlocks_MergeByType() => await RunTest(async accelerator =>
        {
            // Blocks with same type but different damage should merge
            int chunkXZ = 4, height = 4, paddedXZ = chunkXZ + 2;
            int stride = paddedXZ * paddedXZ;
            var blocks = new int[stride * height];
            for (int z = 1; z <= chunkXZ; z++)
                for (int x = 1; x <= chunkXZ; x++)
                    blocks[x + z * paddedXZ + 0 * stride] = PackedBlock.Pack(1, (x + z) % 16);

            await VerifyGpuGreedyMerge(accelerator, blocks, chunkXZ, height);
        });

        /// <summary>
        /// Run the FULL meshing pipeline on GPU and verify against CPU reference.
        /// </summary>
        private async Task VerifyGpuGreedyMerge(Accelerator accelerator, int[] paddedBlocks, int chunkXZ, int height)
        {
            int paddedXZ = chunkXZ + 2;
            int innerXZ = chunkXZ;
            int innerCount = innerXZ * innerXZ;
            int maxQuads = innerCount * height * 6; // worst case: every face visible

            // === CPU Reference ===
            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(paddedBlocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            var originalMasks = (long[])cpuFaceMasks.Clone();
            int cpuTotalFaces = FaceCullCpuReference.CountVisibleFaces(originalMasks);
            var cpuQuads = GreedyMergeCpuReference.GreedyMerge(cpuFaceMasks, paddedBlocks, chunkXZ, height, paddedXZ);

            // Verify CPU reference covers all faces
            var cpuError = GreedyMergeCpuReference.VerifyQuadCoverage(cpuQuads, originalMasks, chunkXZ, height);
            if (cpuError != null) throw new Exception($"CPU reference coverage error: {cpuError}");

            // === GPU Pipeline ===
            // Step 1: Occupancy
            using var gpuBlocks = accelerator.Allocate1D(paddedBlocks);
            using var gpuOccupancy = accelerator.Allocate1D<long>(paddedXZ * paddedXZ);

            var occupancyKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<int>, ArrayView<long>, int, int>(
                FaceCullKernels.BuildOccupancyKernel);

            occupancyKernel(new Index2D(paddedXZ, paddedXZ),
                gpuBlocks.View, gpuOccupancy.View, paddedXZ, height);

            // Step 2: Face Cull
            using var gpuFaceMasks = accelerator.Allocate1D<long>(innerCount * 6);

            var faceCullKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<long>, ArrayView<long>, int, int>(
                FaceCullKernels.FaceCullKernel);

            faceCullKernel(new Index2D(paddedXZ, paddedXZ),
                gpuOccupancy.View, gpuFaceMasks.View, paddedXZ, height);

            // Save a copy of face masks for verification (greedy merge modifies them in place)
            var gpuFaceMasksForVerify = await gpuFaceMasks.CopyToHostAsync();
            int gpuTotalFaces = FaceCullCpuReference.CountVisibleFaces(gpuFaceMasksForVerify);

            // Verify face cull matches CPU
            if (gpuTotalFaces != cpuTotalFaces)
                throw new Exception($"Face count mismatch before merge: GPU={gpuTotalFaces}, CPU={cpuTotalFaces}");

            // Step 3: Greedy Merge
            int maxMergeLayer = Math.Max(height, chunkXZ);
            using var gpuOutputQuads = accelerator.Allocate1D<long>(maxQuads);
            using var gpuQuadCounter = accelerator.Allocate1D(new int[] { 0 });

            var mergeKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<long>, ArrayView<int>, ArrayView<long>, ArrayView<int>,
                int, int, int>(GreedyMergeKernels.GreedyMergeKernel);

            mergeKernel(new Index2D(maxMergeLayer, 6),
                gpuFaceMasks.View, gpuBlocks.View, gpuOutputQuads.View, gpuQuadCounter.View,
                chunkXZ, height, paddedXZ);

            // Read back results
            var counterResult = await gpuQuadCounter.CopyToHostAsync();
            int gpuQuadCount = counterResult[0];

            if (gpuQuadCount <= 0)
                throw new Exception($"GPU produced 0 quads (CPU produced {cpuQuads.Count})");

            if (gpuQuadCount > maxQuads)
                throw new Exception($"GPU quad count {gpuQuadCount} exceeds max {maxQuads}");

            var gpuQuadsRaw = await gpuOutputQuads.CopyToHostAsync();
            var gpuQuads = new List<long>(gpuQuadCount);
            for (int i = 0; i < gpuQuadCount; i++)
                gpuQuads.Add(gpuQuadsRaw[i]);

            // === Verify GPU output ===

            // 1. GPU quads must cover the exact same faces as the original masks
            var gpuCoverageError = GreedyMergeCpuReference.VerifyQuadCoverage(
                gpuQuads, gpuFaceMasksForVerify, chunkXZ, height);
            if (gpuCoverageError != null)
                throw new Exception($"GPU coverage error: {gpuCoverageError}");

            // 2. GPU and CPU should produce similar quad counts
            // (exact match not guaranteed due to different thread ordering affecting merge patterns)
            // But both must cover the same total face count
            int gpuFacesFromQuads = CountFacesInQuads(gpuQuads, chunkXZ, height);
            int cpuFacesFromQuads = CountFacesInQuads(cpuQuads, chunkXZ, height);

            if (gpuFacesFromQuads != cpuFacesFromQuads)
                throw new Exception(
                    $"Face coverage mismatch: GPU quads cover {gpuFacesFromQuads} faces, " +
                    $"CPU quads cover {cpuFacesFromQuads} faces (total visible: {cpuTotalFaces})");
        }

        /// <summary>
        /// Count total individual faces covered by a set of merged quads.
        /// Each quad of width W and height H covers W*H faces.
        /// </summary>
        private static int CountFacesInQuads(List<long> quads, int chunkXZ, int height)
        {
            int total = 0;
            foreach (var q in quads)
            {
                PackedQuad.Unpack(q, out _, out _, out _, out int w, out int h, out _, out _);
                total += w * h;
            }
            return total;
        }
    }
}
