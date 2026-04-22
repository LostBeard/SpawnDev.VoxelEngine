using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Carving;
using SpawnDev.VoxelEngine.SDF;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Carving / CSG service tests. Exercise ITerrainCarve end-to-end so
    // CarveMode, blend-radius resolution, AABB fast-reject and GPU dispatch are
    // all validated together. Desktop-verifiable (no browser gate needed).
    public abstract partial class VoxelEngineTestBase
    {
        // ===================================================================
        // CarveMode + ResolveKernelParams
        // ===================================================================

        [TestMethod]
        public async Task Carve_ResolveKernelParams_DigFillExplode_AllDistinct() => await RunTest(async accelerator =>
        {
            await Task.CompletedTask;
            var carve = new TerrainCarveService(accelerator);

            var dig = carve.ResolveKernelParams(CarveMode.Dig);
            var fill = carve.ResolveKernelParams(CarveMode.Fill);
            var explode = carve.ResolveKernelParams(CarveMode.Explode);

            if (dig.kernelMode != 0) throw new Exception($"Dig must map to kernel mode 0, got {dig.kernelMode}");
            if (fill.kernelMode != 1) throw new Exception($"Fill must map to kernel mode 1, got {fill.kernelMode}");
            if (explode.kernelMode != 0) throw new Exception($"Explode must use subtract mode 0, got {explode.kernelMode}");

            if (explode.blendRadius <= dig.blendRadius)
                throw new Exception(
                    $"Explode blend {explode.blendRadius} must exceed Dig blend {dig.blendRadius} for wider rim");
        });

        // ===================================================================
        // SphereIntersectsChunk - AABB fast-reject
        // ===================================================================

        [TestMethod]
        public async Task Carve_SphereIntersectsChunk_InsideOrigin_Intersects() => await RunTest(async accelerator =>
        {
            await Task.CompletedTask;
            var carve = new TerrainCarveService(accelerator);
            // Chunk at origin, 8x8x8 voxels, 1u voxel. Sphere centered in chunk - must intersect.
            var hit = carve.SphereIntersectsChunk(
                chunkWorldMin: Vector3.Zero, voxelSize: 1f, chunkSize: 8,
                worldCenter: new Vector3(4f, 4f, 4f), radius: 1f, mode: CarveMode.Dig);
            if (!hit) throw new Exception("Sphere fully inside chunk must intersect");
        });

        [TestMethod]
        public async Task Carve_SphereIntersectsChunk_FarAway_DoesNotIntersect() => await RunTest(async accelerator =>
        {
            await Task.CompletedTask;
            var carve = new TerrainCarveService(accelerator);
            var hit = carve.SphereIntersectsChunk(
                chunkWorldMin: Vector3.Zero, voxelSize: 1f, chunkSize: 8,
                worldCenter: new Vector3(1000f, 0f, 0f), radius: 1f, mode: CarveMode.Dig);
            if (hit) throw new Exception("Sphere 1000 units away must NOT intersect");
        });

        [TestMethod]
        public async Task Carve_SphereIntersectsChunk_RadiusZero_DoesNotIntersect() => await RunTest(async accelerator =>
        {
            await Task.CompletedTask;
            var carve = new TerrainCarveService(accelerator);
            var hit = carve.SphereIntersectsChunk(
                chunkWorldMin: Vector3.Zero, voxelSize: 1f, chunkSize: 8,
                worldCenter: new Vector3(4f, 4f, 4f), radius: 0f, mode: CarveMode.Dig);
            if (hit) throw new Exception("Radius-0 sphere must never intersect (kernel is a no-op)");
        });

        [TestMethod]
        public async Task Carve_SphereIntersectsChunk_ExplodeBlend_CatchesEdgeCase() => await RunTest(async accelerator =>
        {
            await Task.CompletedTask;
            var carve = new TerrainCarveService(accelerator);
            // Sphere at (9, 4, 4), radius 1.5. A crisp Dig (blend 1) would fast-reject because
            // (radius+blend+1) = 3.5 and distance from chunk_max=8 to center=9 is 1 (inside).
            // But the important case: sphere well outside chunk should differ between modes
            // when the blend margin is wider.
            // chunk [0..8], voxelSize=1. Dig margin = 1+1+1 = 3. Explode margin = 1+3+1 = 5.
            // Sphere at x=12 is 4u past the chunk edge: Dig rejects (4>3), Explode intersects (4<5).
            var dig = carve.SphereIntersectsChunk(
                chunkWorldMin: Vector3.Zero, voxelSize: 1f, chunkSize: 8,
                worldCenter: new Vector3(12f, 4f, 4f), radius: 1f, mode: CarveMode.Dig);
            var explode = carve.SphereIntersectsChunk(
                chunkWorldMin: Vector3.Zero, voxelSize: 1f, chunkSize: 8,
                worldCenter: new Vector3(12f, 4f, 4f), radius: 1f, mode: CarveMode.Explode);

            if (dig) throw new Exception("Sphere 4u beyond chunk must fast-reject for Dig (margin 3)");
            if (!explode) throw new Exception("Sphere 4u beyond chunk must STILL intersect for Explode (margin 5)");
        });

        // ===================================================================
        // ApplySphereToChunk - basic CSG operations on a real chunk
        // ===================================================================

        [TestMethod]
        public async Task Carve_ApplySphere_Dig_CarvesCenter() => await RunTest(async accelerator =>
        {
            int size = 8;
            float voxel = 1f;
            var carve = new TerrainCarveService(accelerator);
            var initial = CarvingFilledSdf(size, SdfChunk.ToFixedPoint(50f));
            using var buffer = accelerator.Allocate1D(initial);

            var dispatched = await carve.ApplySphereToChunkAsync(
                buffer,
                chunkWorldMin: Vector3.Zero, voxelSize: voxel, chunkSize: size,
                worldCenter: new Vector3(size * voxel * 0.5f),
                radius: 2f, mode: CarveMode.Dig);

            if (!dispatched) throw new Exception("Dispatch must report true for intersecting sphere");

            var result = await buffer.CopyToHostAsync();
            int mid = size / 2;
            short center = result[mid + mid * size + mid * size * size];
            if (center >= 0)
                throw new Exception($"Dig at center should turn SDF negative, got {center}");
        });

        [TestMethod]
        public async Task Carve_ApplySphere_Fill_FillsCenter() => await RunTest(async accelerator =>
        {
            int size = 8;
            float voxel = 1f;
            var carve = new TerrainCarveService(accelerator);
            var initial = CarvingFilledSdf(size, SdfChunk.ToFixedPoint(-50f));
            using var buffer = accelerator.Allocate1D(initial);

            await carve.ApplySphereToChunkAsync(
                buffer,
                chunkWorldMin: Vector3.Zero, voxelSize: voxel, chunkSize: size,
                worldCenter: new Vector3(size * voxel * 0.5f),
                radius: 2f, mode: CarveMode.Fill);

            var result = await buffer.CopyToHostAsync();
            int mid = size / 2;
            short center = result[mid + mid * size + mid * size * size];
            if (center <= 0)
                throw new Exception($"Fill at center should turn SDF positive, got {center}");
        });

        [TestMethod]
        public async Task Carve_ApplySphere_FastReject_LeavesBufferUnchanged() => await RunTest(async accelerator =>
        {
            int size = 8;
            float voxel = 1f;
            var carve = new TerrainCarveService(accelerator);
            var initial = CarvingFilledSdf(size, (short)100);
            using var buffer = accelerator.Allocate1D(initial);

            // Sphere far outside chunk AABB + margin.
            var dispatched = await carve.ApplySphereToChunkAsync(
                buffer,
                chunkWorldMin: Vector3.Zero, voxelSize: voxel, chunkSize: size,
                worldCenter: new Vector3(1000f),
                radius: 1f, mode: CarveMode.Dig);

            if (dispatched) throw new Exception("Far sphere must fast-reject, not dispatch");

            var result = await buffer.CopyToHostAsync();
            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] != initial[i])
                    throw new Exception($"Fast-reject path modified voxel {i}: {initial[i]} -> {result[i]}");
            }
        });

        [TestMethod]
        public async Task Carve_ApplySphere_RadiusZero_NoDispatch() => await RunTest(async accelerator =>
        {
            int size = 4;
            float voxel = 1f;
            var carve = new TerrainCarveService(accelerator);
            var initial = CarvingFilledSdf(size, (short)100);
            using var buffer = accelerator.Allocate1D(initial);

            var dispatched = await carve.ApplySphereToChunkAsync(
                buffer,
                chunkWorldMin: Vector3.Zero, voxelSize: voxel, chunkSize: size,
                worldCenter: new Vector3(size * voxel * 0.5f),
                radius: 0f, mode: CarveMode.Dig);

            if (dispatched) throw new Exception("Radius-0 must fast-reject");

            var result = await buffer.CopyToHostAsync();
            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] != initial[i])
                    throw new Exception($"Radius-0 modified voxel {i}: {initial[i]} -> {result[i]}");
            }
        });

        // ===================================================================
        // CSG round-trip (dig then fill at same center approximately recovers)
        // ===================================================================

        [TestMethod]
        public async Task Carve_DigThenFill_SameCenter_ApproxRoundTrip() => await RunTest(async accelerator =>
        {
            // A solid chunk that is dug-then-filled at the same center should end
            // up near its original SDF values - smooth union inverts smooth subtract
            // up to the blend's quantization / smoothing error. Exact equality is
            // not guaranteed by the smooth-min math, but the values must stay on
            // the same side (>=0) and be within a generous quantization envelope.
            int size = 8;
            float voxel = 1f;
            var carve = new TerrainCarveService(accelerator);
            var initial = CarvingFilledSdf(size, SdfChunk.ToFixedPoint(50f));
            using var buffer = accelerator.Allocate1D(initial);

            var center = new Vector3(size * voxel * 0.5f);

            await carve.ApplySphereToChunkAsync(buffer,
                Vector3.Zero, voxel, size, center, radius: 2f, CarveMode.Dig);
            await carve.ApplySphereToChunkAsync(buffer,
                Vector3.Zero, voxel, size, center, radius: 2f, CarveMode.Fill);

            var result = await buffer.CopyToHostAsync();

            // Every voxel the initial field called "solid" (>0) should still be solid
            // after the round-trip. The blend may have shaved a small amount off the
            // center value, but sign preservation is the contract for Dig+Fill.
            int flipped = 0;
            for (int i = 0; i < result.Length; i++)
            {
                if (initial[i] > 0 && result[i] <= 0) flipped++;
            }
            if (flipped > 0)
                throw new Exception($"Dig+Fill round-trip flipped sign on {flipped}/{result.Length} voxels");

            // Corner voxels (far from sphere) must be bit-exact unchanged.
            if (result[0] != initial[0])
                throw new Exception($"Corner voxel changed after round-trip: {initial[0]} -> {result[0]}");
        });

        // ===================================================================
        // Disjoint spheres - carving outside a chunk must not touch it
        // ===================================================================

        [TestMethod]
        public async Task Carve_DisjointSpheres_DoNotCrossChunkBoundary() => await RunTest(async accelerator =>
        {
            // Two chunks side by side on the X axis. Carve a sphere entirely in
            // chunk A and verify chunk B is untouched (proves the per-chunk AABB
            // reject + per-voxel reject both work).
            int size = 8;
            float voxel = 1f;
            var carve = new TerrainCarveService(accelerator);

            var chunkAInitial = CarvingFilledSdf(size, SdfChunk.ToFixedPoint(50f));
            var chunkBInitial = CarvingFilledSdf(size, SdfChunk.ToFixedPoint(50f));
            using var bufA = accelerator.Allocate1D(chunkAInitial);
            using var bufB = accelerator.Allocate1D(chunkBInitial);

            // Chunk A origin (0,0,0), chunk B origin (8,0,0) - they share the plane x=8.
            var chunkAOrigin = new Vector3(0f);
            var chunkBOrigin = new Vector3(size * voxel, 0f, 0f);

            // Sphere centered at (3, 4, 4) in chunk A with radius 1.5.
            // Reach = r + blend(1) + 1 = 3.5. Distance to chunk B min (8) = 5. Safely disjoint.
            var sphereCenter = new Vector3(3f, 4f, 4f);
            var hitA = await carve.ApplySphereToChunkAsync(
                bufA, chunkAOrigin, voxel, size, sphereCenter, radius: 1.5f, CarveMode.Dig);
            var hitB = await carve.ApplySphereToChunkAsync(
                bufB, chunkBOrigin, voxel, size, sphereCenter, radius: 1.5f, CarveMode.Dig);

            if (!hitA) throw new Exception("Sphere inside chunk A must dispatch");
            if (hitB) throw new Exception("Sphere fully inside chunk A must NOT reach chunk B");

            var resultB = await bufB.CopyToHostAsync();
            for (int i = 0; i < resultB.Length; i++)
            {
                if (resultB[i] != chunkBInitial[i])
                    throw new Exception($"Chunk B voxel {i} was modified by a disjoint carve: " +
                        $"{chunkBInitial[i]} -> {resultB[i]}");
            }
        });

        // ===================================================================
        // Explode vs Dig - Explode produces wider affected footprint
        // ===================================================================

        [TestMethod]
        public async Task Carve_Explode_AffectsWiderRegion_ThanDig() => await RunTest(async accelerator =>
        {
            // Explode's wider blend radius means the kernel touches more voxels
            // than a crisp Dig. We measure "voxels that changed from the initial
            // field" rather than "voxels that went negative" because with a strong
            // positive initial SDF the smooth-subtract formula saturates to the
            // same -sphereSdf for both modes - it's the *outer rim* that widens,
            // not the inner cavity.
            int size = 16;
            float voxel = 1f;
            var carve = new TerrainCarveService(accelerator);
            // Use a weakly-solid initial field so the smooth-blend penalty
            // (k * h * (1-h)) actually matters at the sphere rim.
            var initial = CarvingFilledSdf(size, SdfChunk.ToFixedPoint(2f));

            using var digBuf = accelerator.Allocate1D(initial);
            using var expBuf = accelerator.Allocate1D(initial);

            var center = new Vector3(size * voxel * 0.5f);

            await carve.ApplySphereToChunkAsync(digBuf,
                Vector3.Zero, voxel, size, center, radius: 2f, CarveMode.Dig);
            await carve.ApplySphereToChunkAsync(expBuf,
                Vector3.Zero, voxel, size, center, radius: 2f, CarveMode.Explode);

            var digResult = await digBuf.CopyToHostAsync();
            var expResult = await expBuf.CopyToHostAsync();

            int digChanged = 0, expChanged = 0;
            for (int i = 0; i < initial.Length; i++)
            {
                if (digResult[i] != initial[i]) digChanged++;
                if (expResult[i] != initial[i]) expChanged++;
            }

            if (expChanged <= digChanged)
                throw new Exception(
                    $"Explode should affect MORE voxels than Dig (wider blend rim). " +
                    $"Dig changed={digChanged}, Explode changed={expChanged}");
        });

        // ===================================================================
        // Null / validation guards
        // ===================================================================

        [TestMethod]
        public async Task Carve_ApplySphere_NullBuffer_Throws() => await RunTest(async accelerator =>
        {
            await Task.CompletedTask;
            var carve = new TerrainCarveService(accelerator);
            try
            {
                _ = carve.ApplySphereToChunk(
                    null!, Vector3.Zero, 1f, 8, Vector3.Zero, 1f, CarveMode.Dig);
                throw new Exception("Null buffer must throw ArgumentNullException");
            }
            catch (ArgumentNullException) { /* expected */ }
        });

        [TestMethod]
        public async Task Carve_ApplySphere_UndersizedBuffer_Throws() => await RunTest(async accelerator =>
        {
            await Task.CompletedTask;
            var carve = new TerrainCarveService(accelerator);
            // Claim chunk size 8 (needs 512 elements) but give a 64-element buffer.
            using var tiny = accelerator.Allocate1D<short>(64);
            try
            {
                _ = carve.ApplySphereToChunk(
                    tiny, Vector3.Zero, 1f, 8, new Vector3(4f), 1f, CarveMode.Dig);
                throw new Exception("Undersized buffer must throw ArgumentException");
            }
            catch (ArgumentException) { /* expected */ }
        });

        // ===================================================================
        // Helpers (local to Carving tests - named to avoid collision with other files)
        // ===================================================================

        private static short[] CarvingFilledSdf(int size, short value)
        {
            var arr = new short[size * size * size];
            Array.Fill(arr, value);
            return arr;
        }
    }
}
