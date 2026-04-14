using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Culling;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Frustum + fog culling tests - verify GPU kernel matches CPU reference
    public partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task FrustumCull_AllInView_NothingCulled() => await RunTest(async accelerator =>
        {
            // Place chunks directly in front of camera, all within fog distance
            // Camera at origin looking down +Z
            int chunkCount = 16;
            var centers = new float[chunkCount * 3];
            for (int i = 0; i < chunkCount; i++)
            {
                centers[i * 3 + 0] = (i % 4 - 1.5f) * 16; // X: spread across view
                centers[i * 3 + 1] = 0;                      // Y: ground level
                centers[i * 3 + 2] = (i / 4 + 1) * 16;      // Z: in front of camera
            }

            float halfX = 8, halfY = 128, halfZ = 8;
            float fogDistSq = 10000; // 100 blocks

            var viewProj = FrustumCullCpuReference.BuildTestViewProj(
                0, 0, 0,    // camera
                0, 0, 1,    // look at
                1.2f,        // fov ~69 degrees
                1.0f,        // aspect
                0.1f, 500f); // near/far

            var planes = FrustumCullCpuReference.ExtractFrustumPlanes(viewProj);

            // CPU reference
            var cpuVisible = FrustumCullCpuReference.CullChunks(
                centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            // GPU
            var gpuVisible = await RunFrustumCullOnGpu(
                accelerator, centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            // All chunks should be visible (they're all in front of camera within fog)
            if (cpuVisible.Count != chunkCount)
                throw new Exception($"CPU: expected {chunkCount} visible, got {cpuVisible.Count}");

            if (gpuVisible.Count != chunkCount)
                throw new Exception($"GPU: expected {chunkCount} visible, got {gpuVisible.Count}");
        });

        [TestMethod]
        public async Task FrustumCull_BehindCamera_AllCulled() => await RunTest(async accelerator =>
        {
            // Place all chunks behind the camera
            int chunkCount = 8;
            var centers = new float[chunkCount * 3];
            for (int i = 0; i < chunkCount; i++)
            {
                centers[i * 3 + 0] = (i - 4) * 16;
                centers[i * 3 + 1] = 0;
                centers[i * 3 + 2] = -(i + 1) * 32; // Behind camera (negative Z)
            }

            float halfX = 8, halfY = 128, halfZ = 8;
            float fogDistSq = 1000000; // Very large fog so it's not the fog culling

            var viewProj = FrustumCullCpuReference.BuildTestViewProj(
                0, 0, 0, 0, 0, 1, 1.2f, 1.0f, 0.1f, 500f);
            var planes = FrustumCullCpuReference.ExtractFrustumPlanes(viewProj);

            var cpuVisible = FrustumCullCpuReference.CullChunks(
                centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            var gpuVisible = await RunFrustumCullOnGpu(
                accelerator, centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            if (cpuVisible.Count != 0)
                throw new Exception($"CPU: expected 0 visible behind camera, got {cpuVisible.Count}");

            if (gpuVisible.Count != 0)
                throw new Exception($"GPU: expected 0 visible behind camera, got {gpuVisible.Count}");
        });

        [TestMethod]
        public async Task FrustumCull_FogDistance_CullsFarChunks() => await RunTest(async accelerator =>
        {
            // Place chunks at various distances, tight fog should cull distant ones
            int chunkCount = 10;
            var centers = new float[chunkCount * 3];
            for (int i = 0; i < chunkCount; i++)
            {
                centers[i * 3 + 0] = 0;
                centers[i * 3 + 1] = 0;
                centers[i * 3 + 2] = (i + 1) * 20; // 20, 40, 60, ..., 200 blocks away
            }

            float halfX = 8, halfY = 128, halfZ = 8;
            float fogDist = 100; // 100 blocks
            float fogDistSq = fogDist * fogDist;

            var viewProj = FrustumCullCpuReference.BuildTestViewProj(
                0, 0, 0, 0, 0, 1, 1.2f, 1.0f, 0.1f, 500f);
            var planes = FrustumCullCpuReference.ExtractFrustumPlanes(viewProj);

            var cpuVisible = FrustumCullCpuReference.CullChunks(
                centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            var gpuVisible = await RunFrustumCullOnGpu(
                accelerator, centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            // Chunks at distance > 100 should be culled (5th chunk at 100 is borderline)
            // CPU and GPU must agree on the exact set
            if (cpuVisible.Count != gpuVisible.Count)
                throw new Exception($"Fog cull mismatch: CPU visible={cpuVisible.Count}, GPU visible={gpuVisible.Count}");

            // At least the closest chunks should survive, and the farthest should be culled
            if (cpuVisible.Count == 0)
                throw new Exception("Fog culled everything - test is broken");
            if (cpuVisible.Count == chunkCount)
                throw new Exception("Fog culled nothing - fog distance too large for test");
        });

        [TestMethod]
        public async Task FrustumCull_MixedVisibility_MatchesCpuReference() => await RunTest(async accelerator =>
        {
            // Scatter chunks around camera - some in view, some behind, some to the side, some far
            var centers = new float[]
            {
                0, 0, 30,       // 0: directly in front - visible
                0, 0, -30,      // 1: directly behind - culled
                50, 0, 30,      // 2: far right - likely culled by frustum
                -50, 0, 30,     // 3: far left - likely culled by frustum
                0, 0, 10,       // 4: very close, in front - visible
                0, 100, 30,     // 5: high up but in front - depends on frustum
                0, 0, 200,      // 6: far in front, may be culled by fog
                10, 0, 20,      // 7: slightly right, in front - visible
            };
            int chunkCount = 8;

            float halfX = 8, halfY = 8, halfZ = 8; // Small chunks for precise frustum testing
            float fogDistSq = 150 * 150; // 150 block fog

            var viewProj = FrustumCullCpuReference.BuildTestViewProj(
                0, 0, 0, 0, 0, 1, 1.0f, 1.0f, 0.1f, 500f);
            var planes = FrustumCullCpuReference.ExtractFrustumPlanes(viewProj);

            var cpuVisible = FrustumCullCpuReference.CullChunks(
                centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            var gpuVisible = await RunFrustumCullOnGpu(
                accelerator, centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            // Sort both lists for comparison (GPU output order is non-deterministic)
            cpuVisible.Sort();
            gpuVisible.Sort();

            if (cpuVisible.Count != gpuVisible.Count)
                throw new Exception(
                    $"Visible count mismatch: CPU={cpuVisible.Count} [{string.Join(",", cpuVisible)}], " +
                    $"GPU={gpuVisible.Count} [{string.Join(",", gpuVisible)}]");

            for (int i = 0; i < cpuVisible.Count; i++)
            {
                if (cpuVisible[i] != gpuVisible[i])
                    throw new Exception(
                        $"Visible set mismatch at index {i}: CPU={cpuVisible[i]}, GPU={gpuVisible[i]}. " +
                        $"CPU=[{string.Join(",", cpuVisible)}], GPU=[{string.Join(",", gpuVisible)}]");
            }

            // Sanity: not everything should be visible (we have behind-camera chunks)
            if (cpuVisible.Count == chunkCount)
                throw new Exception("Nothing was culled - test setup is wrong");
            if (cpuVisible.Count == 0)
                throw new Exception("Everything was culled - test setup is wrong");
        });

        private async Task<List<int>> RunFrustumCullOnGpu(
            Accelerator accelerator,
            float[] chunkCenters, float halfX, float halfY, float halfZ,
            float[] frustumPlanes, float cameraX, float cameraY, float cameraZ, float fogDistSq)
        {
            int chunkCount = chunkCenters.Length / 3;

            using var gpuCenters = accelerator.Allocate1D(chunkCenters);
            using var gpuHalfSizes = accelerator.Allocate1D(new float[] { halfX, halfY, halfZ });
            using var gpuPlanes = accelerator.Allocate1D(frustumPlanes);
            using var gpuFogParams = accelerator.Allocate1D(new float[] { cameraX, cameraY, cameraZ, fogDistSq });
            using var gpuVisibleIndices = accelerator.Allocate1D<int>(chunkCount);
            using var gpuVisibleCount = accelerator.Allocate1D(new int[] { 0 });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                ArrayView<float>, ArrayView<int>, ArrayView<int>, int>(
                FrustumCullKernels.FrustumFogCullKernel);

            kernel((Index1D)chunkCount,
                gpuCenters.View, gpuHalfSizes.View, gpuPlanes.View,
                gpuFogParams.View, gpuVisibleIndices.View, gpuVisibleCount.View,
                chunkCount);

            await accelerator.SynchronizeAsync();
            var countResult = await gpuVisibleCount.CopyToHostAsync();
            int count = Math.Min(countResult[0], chunkCount);

            if (count == 0) return new List<int>();

            var indicesResult = await gpuVisibleIndices.CopyToHostAsync();
            var visible = new List<int>(count);
            for (int i = 0; i < count; i++)
            {
                visible.Add(indicesResult[i]);
            }
            return visible;
        }
    }
}
