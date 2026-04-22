using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.SDF;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // SDF noise evaluation and sphere modification kernel tests (GPU, all backends).
    public abstract partial class VoxelEngineTestBase
    {
        // ===================================================================
        // EvaluateSdfKernel
        // ===================================================================

        [TestMethod]
        public async Task SdfEvaluate_SameSeedSameInput_ProducesIdenticalOutput() => await RunTest(async accelerator =>
        {
            int size = 8;
            var a = await EvaluateSdf(accelerator, size, 0, 0, 0, 1f, seed: 42);
            var b = await EvaluateSdf(accelerator, size, 0, 0, 0, 1f, seed: 42);
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                    throw new Exception($"Same seed produced different output at index {i}: {a[i]} vs {b[i]}");
            }
        });

        [TestMethod]
        public async Task SdfEvaluate_DifferentSeeds_ProduceDifferentOutput() => await RunTest(async accelerator =>
        {
            int size = 8;
            var a = await EvaluateSdf(accelerator, size, 0, 0, 0, 1f, seed: 1);
            var b = await EvaluateSdf(accelerator, size, 0, 0, 0, 1f, seed: 999);
            int differences = 0;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i]) differences++;
            // Expect most voxels to differ under different seeds
            if (differences < a.Length / 4)
                throw new Exception($"Different seeds produced too few differences ({differences} / {a.Length})");
        });

        [TestMethod]
        public async Task SdfEvaluate_DeepUnderground_MostlySolid() => await RunTest(async accelerator =>
        {
            // At Y = -100 (well below base terrain height of 64), SDF should be strongly positive everywhere.
            int size = 8;
            var values = await EvaluateSdf(accelerator, size, 0f, -100f, 0f, 1f, seed: 7);
            int solidCount = 0;
            foreach (var v in values)
                if (v > 0) solidCount++;
            if (solidCount < values.Length)
                throw new Exception($"Deep underground chunk should be fully solid, got {solidCount}/{values.Length}");
        });

        [TestMethod]
        public async Task SdfEvaluate_HighAltitude_MostlyAir() => await RunTest(async accelerator =>
        {
            // At Y = 500 (far above any terrain feature), SDF should be strongly negative.
            int size = 8;
            var values = await EvaluateSdf(accelerator, size, 0f, 500f, 0f, 1f, seed: 7);
            int airCount = 0;
            foreach (var v in values)
                if (v <= 0) airCount++;
            if (airCount < values.Length)
                throw new Exception($"High altitude chunk should be fully air, got {airCount}/{values.Length}");
        });

        // ===================================================================
        // ModifySdfSphereKernel
        // ===================================================================

        [TestMethod]
        public async Task SdfModifySphere_Dig_CarvesHoleInSolid() => await RunTest(async accelerator =>
        {
            // Start with a fully-solid chunk, dig a sphere at the center.
            int size = 8;
            float voxelSize = 1f;
            var initial = FilledSdf(size, SdfChunk.ToFixedPoint(50f)); // strongly solid
            using var buffer = accelerator.Allocate1D(initial);

            float centerX = size * voxelSize * 0.5f;
            float centerY = size * voxelSize * 0.5f;
            float centerZ = size * voxelSize * 0.5f;
            float radius = 2f;

            RunModifySphere(accelerator, buffer, size, 0, 0, 0, voxelSize,
                centerX, centerY, centerZ, radius, mode: 0 /* dig */);
            var result = await buffer.CopyToHostAsync();

            // Voxel at exact center should now be air (SDF negative).
            int cx = size / 2, cy = size / 2, cz = size / 2;
            short centerAfter = result[cx + cz * size + cy * size * size];
            if (centerAfter >= 0)
                throw new Exception($"After dig, center SDF should be negative (air), got {centerAfter}");

            // Voxel at a corner (far from sphere) should be unchanged.
            short cornerAfter = result[0];
            if (cornerAfter != initial[0])
                throw new Exception($"Corner voxel should be unchanged by dig, got {cornerAfter} vs {initial[0]}");
        });

        [TestMethod]
        public async Task SdfModifySphere_Fill_AddsSolidToAir() => await RunTest(async accelerator =>
        {
            // Start with a fully-air chunk, fill a sphere at the center.
            int size = 8;
            float voxelSize = 1f;
            var initial = FilledSdf(size, SdfChunk.ToFixedPoint(-50f)); // strongly air
            using var buffer = accelerator.Allocate1D(initial);

            float centerX = size * voxelSize * 0.5f;
            float centerY = size * voxelSize * 0.5f;
            float centerZ = size * voxelSize * 0.5f;
            float radius = 2f;

            RunModifySphere(accelerator, buffer, size, 0, 0, 0, voxelSize,
                centerX, centerY, centerZ, radius, mode: 1 /* fill */);
            var result = await buffer.CopyToHostAsync();

            // Voxel at exact center should now be solid (SDF positive).
            int cx = size / 2, cy = size / 2, cz = size / 2;
            short centerAfter = result[cx + cz * size + cy * size * size];
            if (centerAfter <= 0)
                throw new Exception($"After fill, center SDF should be positive (solid), got {centerAfter}");

            // Voxel at a corner (far from sphere) should be unchanged.
            short cornerAfter = result[0];
            if (cornerAfter != initial[0])
                throw new Exception($"Corner voxel should be unchanged by fill, got {cornerAfter} vs {initial[0]}");
        });

        [TestMethod]
        public async Task SdfModifySphere_FarFromSphere_AllUnchanged() => await RunTest(async accelerator =>
        {
            // A sphere far outside the chunk must leave every voxel untouched.
            int size = 4;
            float voxelSize = 1f;
            var initial = FilledSdf(size, (short)100);
            using var buffer = accelerator.Allocate1D(initial);

            // Sphere is 1000 units away - well beyond the early-out cutoff
            RunModifySphere(accelerator, buffer, size, 0, 0, 0, voxelSize,
                1000f, 1000f, 1000f, 2f, mode: 0);
            var result = await buffer.CopyToHostAsync();

            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] != initial[i])
                    throw new Exception($"Voxel {i} changed despite being far from sphere: {initial[i]} -> {result[i]}");
            }
        });

        // ===================================================================
        // CSG edge cases (exercises SdfSmoothUnion / SdfSmoothSubtract via production sphere kernel)
        // ===================================================================

        [TestMethod]
        public async Task SdfModifySphere_RadiusZero_LeavesFieldUnchanged() => await RunTest(async accelerator =>
        {
            // Radius 0 must early-out; no voxel should be touched.
            int size = 4;
            float voxelSize = 1f;
            var initial = FilledSdf(size, (short)100);
            using var buffer = accelerator.Allocate1D(initial);

            float centerX = size * voxelSize * 0.5f;
            float centerY = size * voxelSize * 0.5f;
            float centerZ = size * voxelSize * 0.5f;

            RunModifySphere(accelerator, buffer, size, 0, 0, 0, voxelSize,
                centerX, centerY, centerZ, radius: 0f, mode: 0);
            var result = await buffer.CopyToHostAsync();

            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] != initial[i])
                    throw new Exception($"Voxel {i} changed on radius-0 dig: {initial[i]} -> {result[i]}");
            }

            // Same check for fill mode.
            using var buffer2 = accelerator.Allocate1D(initial);
            RunModifySphere(accelerator, buffer2, size, 0, 0, 0, voxelSize,
                centerX, centerY, centerZ, radius: 0f, mode: 1);
            var result2 = await buffer2.CopyToHostAsync();
            for (int i = 0; i < result2.Length; i++)
            {
                if (result2[i] != initial[i])
                    throw new Exception($"Voxel {i} changed on radius-0 fill: {initial[i]} -> {result2[i]}");
            }
        });

        [TestMethod]
        public async Task SdfModifySphere_RepeatedDig_MonotonicallyDecreasing() => await RunTest(async accelerator =>
        {
            // Repeated dig at the same center must progressively lower (or keep) the SDF at the center.
            // SmoothSubtract is idempotent or decreasing per call - it never raises the field.
            int size = 8;
            float voxelSize = 1f;
            var initial = FilledSdf(size, SdfChunk.ToFixedPoint(50f));
            using var buffer = accelerator.Allocate1D(initial);

            float centerX = size * voxelSize * 0.5f;
            float centerY = size * voxelSize * 0.5f;
            float centerZ = size * voxelSize * 0.5f;
            float radius = 2f;

            int cx = size / 2, cy = size / 2, cz = size / 2;
            int centerIdx = cx + cz * size + cy * size * size;

            short[] snapshots = new short[4];
            snapshots[0] = initial[centerIdx];

            for (int i = 1; i < 4; i++)
            {
                RunModifySphere(accelerator, buffer, size, 0, 0, 0, voxelSize,
                    centerX, centerY, centerZ, radius, mode: 0);
                var snap = await buffer.CopyToHostAsync();
                snapshots[i] = snap[centerIdx];
            }

            for (int i = 1; i < 4; i++)
            {
                if (snapshots[i] > snapshots[i - 1])
                    throw new Exception(
                        $"Dig {i} raised SDF at center: {snapshots[i - 1]} -> {snapshots[i]} (should be <= previous)");
            }
        });

        [TestMethod]
        public async Task SdfModifySphere_RepeatedFill_MonotonicallyIncreasing() => await RunTest(async accelerator =>
        {
            // Mirror test of RepeatedDig: repeated fill at the same center must progressively raise (or keep) SDF.
            int size = 8;
            float voxelSize = 1f;
            var initial = FilledSdf(size, SdfChunk.ToFixedPoint(-50f));
            using var buffer = accelerator.Allocate1D(initial);

            float centerX = size * voxelSize * 0.5f;
            float centerY = size * voxelSize * 0.5f;
            float centerZ = size * voxelSize * 0.5f;
            float radius = 2f;

            int cx = size / 2, cy = size / 2, cz = size / 2;
            int centerIdx = cx + cz * size + cy * size * size;

            short[] snapshots = new short[4];
            snapshots[0] = initial[centerIdx];

            for (int i = 1; i < 4; i++)
            {
                RunModifySphere(accelerator, buffer, size, 0, 0, 0, voxelSize,
                    centerX, centerY, centerZ, radius, mode: 1);
                var snap = await buffer.CopyToHostAsync();
                snapshots[i] = snap[centerIdx];
            }

            for (int i = 1; i < 4; i++)
            {
                if (snapshots[i] < snapshots[i - 1])
                    throw new Exception(
                        $"Fill {i} lowered SDF at center: {snapshots[i - 1]} -> {snapshots[i]} (should be >= previous)");
            }
        });

        [TestMethod]
        public async Task SdfModifySphere_DigOnAir_NoSolidCreated() => await RunTest(async accelerator =>
        {
            // Digging an already-air region must not create solid (smooth subtract of air-from-air stays air).
            int size = 8;
            float voxelSize = 1f;
            var initial = FilledSdf(size, SdfChunk.ToFixedPoint(-50f));
            using var buffer = accelerator.Allocate1D(initial);

            float centerX = size * voxelSize * 0.5f;
            float centerY = size * voxelSize * 0.5f;
            float centerZ = size * voxelSize * 0.5f;
            float radius = 2f;

            RunModifySphere(accelerator, buffer, size, 0, 0, 0, voxelSize,
                centerX, centerY, centerZ, radius, mode: 0);
            var result = await buffer.CopyToHostAsync();

            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] > 0)
                    throw new Exception($"Dig on air created solid at voxel {i}: {initial[i]} -> {result[i]}");
            }
        });

        [TestMethod]
        public async Task SdfModifySphere_FillOnSolid_NoAirCreated() => await RunTest(async accelerator =>
        {
            // Filling an already-solid region must not create air (smooth union of solid-with-solid stays solid).
            int size = 8;
            float voxelSize = 1f;
            var initial = FilledSdf(size, SdfChunk.ToFixedPoint(50f));
            using var buffer = accelerator.Allocate1D(initial);

            float centerX = size * voxelSize * 0.5f;
            float centerY = size * voxelSize * 0.5f;
            float centerZ = size * voxelSize * 0.5f;
            float radius = 2f;

            RunModifySphere(accelerator, buffer, size, 0, 0, 0, voxelSize,
                centerX, centerY, centerZ, radius, mode: 1);
            var result = await buffer.CopyToHostAsync();

            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] < 0)
                    throw new Exception($"Fill on solid created air at voxel {i}: {initial[i]} -> {result[i]}");
            }
        });

        [TestMethod]
        public async Task SdfModifySphere_TwoAdjacentFills_BlendWider_ThanSingleFill() => await RunTest(async accelerator =>
        {
            // Two fills with centers radius-apart should produce a wider "solid" footprint than a single fill,
            // confirming smooth union's blend radius (k=1) merges adjacent spheres rather than carving a gap.
            int size = 16;
            float voxelSize = 1f;

            // Baseline: single fill at x-midpoint.
            var initialA = FilledSdf(size, SdfChunk.ToFixedPoint(-50f));
            using var bufA = accelerator.Allocate1D(initialA);
            float cyCommon = size * voxelSize * 0.5f;
            float czCommon = size * voxelSize * 0.5f;
            RunModifySphere(accelerator, bufA, size, 0, 0, 0, voxelSize,
                size * voxelSize * 0.5f, cyCommon, czCommon, radius: 2f, mode: 1);
            var a = await bufA.CopyToHostAsync();

            // Experiment: two fills, centers 3u apart along X.
            var initialB = FilledSdf(size, SdfChunk.ToFixedPoint(-50f));
            using var bufB = accelerator.Allocate1D(initialB);
            float center1X = size * voxelSize * 0.5f - 1.5f;
            float center2X = size * voxelSize * 0.5f + 1.5f;
            RunModifySphere(accelerator, bufB, size, 0, 0, 0, voxelSize,
                center1X, cyCommon, czCommon, radius: 2f, mode: 1);
            RunModifySphere(accelerator, bufB, size, 0, 0, 0, voxelSize,
                center2X, cyCommon, czCommon, radius: 2f, mode: 1);
            var b = await bufB.CopyToHostAsync();

            // Count solid voxels (SDF > 0) in each result. Two-sphere coverage must exceed single-sphere coverage.
            int solidA = 0, solidB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] > 0) solidA++;
                if (b[i] > 0) solidB++;
            }

            if (solidB <= solidA)
                throw new Exception($"Two adjacent fills did not expand solid footprint: single={solidA}, double={solidB}");
        });

        // ===================================================================
        // Helpers
        // ===================================================================

        private async Task<short[]> EvaluateSdf(
            Accelerator accelerator, int size,
            float chunkWorldX, float chunkWorldY, float chunkWorldZ,
            float voxelSize, int seed)
        {
            using var buffer = accelerator.Allocate1D<short>(size * size * size);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<short>, float, float, float, float, int, int>(
                SdfNoiseKernels.EvaluateSdfKernel);
            kernel(new Index3D(size, size, size),
                buffer.View, chunkWorldX, chunkWorldY, chunkWorldZ, voxelSize, size, seed);
            await accelerator.SynchronizeAsync();
            return await buffer.CopyToHostAsync();
        }

        private void RunModifySphere(
            Accelerator accelerator,
            MemoryBuffer1D<short, Stride1D.Dense> buffer, int size,
            float chunkWorldX, float chunkWorldY, float chunkWorldZ, float voxelSize,
            float centerX, float centerY, float centerZ, float radius, int mode,
            float blendRadius = 1f)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<short>, float, float, float, float, int, float,
                float, float, float, float, int>(
                SdfNoiseKernels.ModifySdfSphereKernel);
            kernel(new Index3D(size, size, size),
                buffer.View, centerX, centerY, centerZ, radius, mode, blendRadius,
                chunkWorldX, chunkWorldY, chunkWorldZ, voxelSize, size);
            accelerator.Synchronize();
        }

        private static short[] FilledSdf(int size, short value)
        {
            var arr = new short[size * size * size];
            Array.Fill(arr, value);
            return arr;
        }
    }
}
