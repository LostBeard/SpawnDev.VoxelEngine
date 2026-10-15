using System.Numerics;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Buffers;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // QuadArena: many meshes' quads in a few big buffers. The allocator is pure bookkeeping (checked on the CPU against a brute-force
    // model); the arena is checked where it matters: meshes written into it must decode to exactly the faces of the CPU reference.
    public abstract partial class VoxelEngineTestBase
    {
        // =====================================================================
        // RangeAllocator (CPU)
        // =====================================================================

        [TestMethod]
        public void RangeAllocator_HandsOutFromTheStart_AndFreesBackToOneRange()
        {
            var a = new RangeAllocator(100);
            int x = a.Allocate(30), y = a.Allocate(30), z = a.Allocate(40);
            if ((x, y, z) != (0, 30, 60)) throw new Exception($"Expected 0, 30, 60, got {x}, {y}, {z}");
            if (a.Allocate(1) != -1 || a.FreeTotal != 0) throw new Exception("A full allocator has nothing to give");
            // Free out of order: the ranges must merge back into one
            a.Free(y, 30); a.Free(z, 40); a.Free(x, 30);
            if (a.FreeRangeCount != 1 || a.LargestFree != 100 || a.Used != 0) throw new Exception($"Not merged: {a.FreeRangeCount} ranges, largest {a.LargestFree}, used {a.Used}");
        }

        [TestMethod]
        public void RangeAllocator_PicksTheSmallestRangeThatFits()
        {
            var a = new RangeAllocator(100);
            int p0 = a.Allocate(10), p1 = a.Allocate(20), p2 = a.Allocate(5), p3 = a.Allocate(30), p4 = a.Allocate(35);
            a.Free(p1, 20);   // a hole of 20 at 10
            a.Free(p3, 30);   // a hole of 30 at 35
            // 25 fits only the 30-hole, which leaves a hole of 5 at p3 + 25
            int got = a.Allocate(25);
            if (got != p3) throw new Exception($"25 fits only the 30-hole at {p3}, got {got}");
            // 5 fits the 20-hole (at p1, found first) and the 5-hole: best fit takes the smaller, first fit would take p1
            int small = a.Allocate(5);
            if (small != p3 + 25) throw new Exception($"5 fits the 20-hole at {p1} and the 5-hole at {p3 + 25}; best fit is the 5-hole, got {small}");
            _ = (p0, p2, p4);
        }

        [TestMethod]
        public void RangeAllocator_RejectsADoubleFree_AndARangeOutsideIt()
        {
            var a = new RangeAllocator(50);
            int x = a.Allocate(10);
            a.Free(x, 10);
            bool threw = false;
            try { a.Free(x, 10); } catch (InvalidOperationException) { threw = true; }
            if (!threw) throw new Exception("Freeing a range twice must throw");
            threw = false;
            int y = a.Allocate(10), z = a.Allocate(10);
            a.Free(z, 10);
            try { a.Free(y + 5, 10); } catch (InvalidOperationException) { threw = true; } // overlaps the free range that starts at z
            if (!threw) throw new Exception("A range overlapping a free range must throw");
            threw = false;
            try { a.Free(45, 10); } catch (ArgumentOutOfRangeException) { threw = true; }
            if (!threw) throw new Exception("A range past the capacity must throw");
        }

        [TestMethod]
        public void RangeAllocator_RandomAllocateAndFree_NeverOverlaps_AndKeepsTheTotals()
        {
            const int capacity = 5000;
            var a = new RangeAllocator(capacity);
            var owner = new int[capacity];                       // 0 = free, else the id of the allocation holding the cell
            var live = new List<(int id, int start, int len)>();
            var rng = new Random(12345);
            int nextId = 1;
            for (int step = 0; step < 6000; step++)
            {
                if (live.Count == 0 || rng.Next(100) < 55)
                {
                    int len = 1 + rng.Next(rng.Next(10) == 0 ? 400 : 40);
                    int at = a.Allocate(len);
                    if (at < 0)
                    {
                        if (a.LargestFree >= len) throw new Exception($"step {step}: refused {len} though {a.LargestFree} is free in one range");
                        continue;
                    }
                    int id = nextId++;
                    for (int i = at; i < at + len; i++)
                    {
                        if (owner[i] != 0) throw new Exception($"step {step}: cell {i} given out twice");
                        owner[i] = id;
                    }
                    live.Add((id, at, len));
                }
                else
                {
                    int k = rng.Next(live.Count);
                    var (id, start, len) = live[k];
                    live.RemoveAt(k);
                    for (int i = start; i < start + len; i++) owner[i] = 0;
                    a.Free(start, len);
                }
                int used = owner.Count(o => o != 0);
                if (a.Used != used || a.FreeTotal != capacity - used)
                    throw new Exception($"step {step}: allocator says {a.Used} used, the model has {used}");
            }
            foreach (var (_, start, len) in live) a.Free(start, len);
            if (a.FreeRangeCount != 1 || a.LargestFree != capacity) throw new Exception("After freeing everything the allocator must be one range again");
        }

        // =====================================================================
        // QuadArena (GPU)
        // =====================================================================

        private static async Task<long[]> QaQuads(ColumnMesh.QuadList list) => await list.CopyToHostAsync();

        [TestMethod]
        public async Task QuadArena_PooledMesh_HasExactlyTheQuadsOfAnUnpooledMeshTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 6;
            using var arena = new QuadArena(accelerator, 1 << 16);
            using var plain = new ColumnMesher(accelerator, sections) { EmbedSectionIndex = true };
            using var pooled = new ColumnMesher(accelerator, sections) { EmbedSectionIndex = true, Arena = arena };
            plain.SetBlockClasses(CmFlags);
            pooled.SetBlockClasses(CmFlags);

            var meshes = new List<ColumnMesh>();
            try
            {
                for (int seed = 1; seed <= 4; seed++)
                {
                    using var gpu = CmUpload(accelerator, CmTerrain(sections, seed));
                    using var reference = await plain.MeshAsync(gpu);
                    var mesh = await pooled.MeshAsync(gpu);
                    meshes.Add(mesh);
                    foreach (var (a, b, name) in new[] { (reference.Opaque, mesh.Opaque, "opaque"), (reference.Plants, mesh.Plants, "plants"), (reference.Water, mesh.Water, "water") })
                    {
                        if (a.Count != b.Count || !a.SectionStart.SequenceEqual(b.SectionStart)) throw new Exception($"seed {seed} {name}: counts differ ({a.Count} vs {b.Count})");
                        if (a.Count == 0) continue;
                        if (!b.IsPooled || a.IsPooled) throw new Exception($"seed {seed} {name}: only the arena mesh is pooled");
                        // The same quads, not the same order: the merge appends with atomics, so two meshings of one column order a section differently
                        long[] plainQuads = await QaQuads(a), pooledQuads = await QaQuads(b);
                        Array.Sort(plainQuads);
                        Array.Sort(pooledQuads);
                        if (!Enumerable.SequenceEqual(plainQuads, pooledQuads)) throw new Exception($"seed {seed} {name}: the pooled quads differ from the unpooled ones");
                    }
                }
                // Several lists share a block, each at its own place
                var ranges = meshes.SelectMany(m => new[] { m.Opaque, m.Plants, m.Water }).Where(l => l.Count > 0).Select(l => (l.ArenaBlock, l.FirstQuad, l.Count)).OrderBy(r => r).ToList();
                if (arena.BlockCount != 1) throw new Exception($"A few small meshes fit one block, got {arena.BlockCount}");
                for (int i = 1; i < ranges.Count; i++)
                    if (ranges[i].FirstQuad < ranges[i - 1].FirstQuad + ranges[i - 1].Count) throw new Exception($"Lists overlap: {ranges[i - 1]} and {ranges[i]}");
                if (arena.UsedQuads != ranges.Sum(r => (long)r.Count)) throw new Exception($"The arena says {arena.UsedQuads} quads used, the lists hold {ranges.Sum(r => (long)r.Count)}");
            }
            finally { foreach (var m in meshes) m.Dispose(); }
            if (arena.UsedQuads != 0) throw new Exception($"Disposing the meshes must give the quads back, {arena.UsedQuads} are still used");
        });

        [TestMethod]
        public async Task QuadArena_PooledMeshes_MatchTheCpuReference_AcrossBlocks_AndAfterFreeingTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 6;
            // Tiny blocks: the terrain meshes need several, and a re-mesh lands in the holes left by the one it replaced
            using var arena = new QuadArena(accelerator, 3000);
            using var mesher = new ColumnMesher(accelerator, sections) { MissingNeighborsAreSolid = false, EmbedSectionIndex = true, Arena = arena };
            mesher.SetBlockClasses(CmFlags);

            var keep = new List<ColumnMesh>();
            try
            {
                for (int seed = 1; seed <= 6; seed++)
                {
                    var col = CmTerrain(sections, seed);
                    // The verify helper meshes, decodes and compares with the reference, then disposes: the next seed reuses its ranges
                    await CmMeshAndVerify(accelerator, mesher, col, new ushort[]?[4], sections, $"pooled seed {seed}");
                    // And a mesh that stays, so the next ones must work around it
                    using var gpu = CmUpload(accelerator, col);
                    keep.Add(await mesher.MeshAsync(gpu));
                }
                if (arena.BlockCount < 2) throw new Exception($"3000-quad blocks should have overflowed into several, got {arena.BlockCount}");
                // Every kept mesh still decodes to its own column after all the allocating and freeing around it
                for (int i = 0; i < keep.Count; i++)
                {
                    var col = CmTerrain(sections, i + 1);
                    CmAssertSame(CmExpectedCubeFaces(col, new ushort[]?[4], sections, BlockClass.OpaqueShift, false), await CmDecode(keep[i].Opaque, "opaque"), $"kept mesh {i} / opaque");
                    CmAssertSame(CmExpectedCubeFaces(col, new ushort[]?[4], sections, BlockClass.WaterShift, false), await CmDecode(keep[i].Water, "water"), $"kept mesh {i} / water");
                    CmAssertSame(CmExpectedPlants(col, sections), await CmDecode(keep[i].Plants, "plants"), $"kept mesh {i} / plants");
                }
            }
            finally { foreach (var m in keep) m.Dispose(); }
            if (arena.UsedQuads != 0) throw new Exception($"{arena.UsedQuads} quads still used after every mesh was disposed");
        });

        [TestMethod]
        public async Task QuadArena_AListLongerThanABlock_GetsABlockOfItsOwn_AndDisposingTwiceFreesOnceTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 6;
            using var arena = new QuadArena(accelerator, 16);        // smaller than any real list
            using var mesher = new ColumnMesher(accelerator, sections) { EmbedSectionIndex = true, Arena = arena };
            mesher.SetBlockClasses(CmFlags);
            var col = CmTerrain(sections, 7);
            await CmMeshAndVerify(accelerator, mesher, col, new ushort[]?[4], sections, "oversize lists");

            using var gpu = CmUpload(accelerator, col);
            var mesh = await mesher.MeshAsync(gpu);
            long used = arena.UsedQuads;
            if (used != mesh.Opaque.Count + mesh.Plants.Count + mesh.Water.Count) throw new Exception("used quads must be the mesh's quads");
            mesh.Dispose();
            mesh.Dispose(); // a second Dispose must not free the ranges again (the allocator throws on a double free)
            if (arena.UsedQuads != 0) throw new Exception("freed");
            // The last mesh's compact dispatches are still queued (MeshAsync flushes, it does not wait for them) and the Wasm backend frees
            // a disposed buffer at once: without this, the next test's allocations are written into by the stale dispatches
            // ("RangeError: offset is out of bounds" in whatever test ran next; shown by an A/B that changed only this line)
            await accelerator.SynchronizeAsync();
        });

        [TestMethod]
        public async Task QuadArena_TwoColumnsInOneBlock_EachDrawsItsOwnRange_AtItsOwnOriginTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            // Column A: one cyan block. Column B: one orange block. Both are meshed into the SAME arena block, so B's quads start after A's:
            // drawing B from quad 0 would show A's cyan block at B's place
            var materials = new BlockMaterial[3];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            materials[2] = BlockMaterial.Uniform(BrLayerOrange);
            using var arena = new QuadArena(accelerator, 1 << 10);
            using var mesher = new ColumnMesher(accelerator, 1) { MissingNeighborsAreSolid = false, EmbedSectionIndex = true, Arena = arena };
            mesher.SetBlockClasses(CmFlags);
            var a = CmEmpty(1); a[CmIdx(0, 0, 0)] = 1;
            var b = CmEmpty(1); b[CmIdx(0, 0, 0)] = 2;
            using var gpuA = CmUpload(accelerator, a);
            using var gpuB = CmUpload(accelerator, b);
            using var meshA = await mesher.MeshAsync(gpuA);
            using var meshB = await mesher.MeshAsync(gpuB);
            if (meshA.Opaque.ArenaBlock != meshB.Opaque.ArenaBlock) throw new Exception("both columns should share one arena block");
            if (meshB.Opaque.FirstQuad < meshA.Opaque.FirstQuad + meshA.Opaque.Count) throw new Exception("column B's quads must start after column A's");

            var eye = new Vector3(3f, 5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(3f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            using var textures = BlockTextureArray.FromAtlas(device, queue, BrAtlas(), 32, 32, BrTile);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 16);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            pipeline.SetSectionOrigins(0, new[] { new Vector3(0, 0, 0), new Vector3(4, 0, 0) });
            pipeline.BeginFrame(frame);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero);
                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, meshA.Opaque.Buffer!.GetGPUBuffer()!);
                pipeline.Draw(pass, meshA.Opaque.FirstQuad, meshA.Opaque.Count, 0);
                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, meshB.Opaque.Buffer!.GetGPUBuffer()!); // the same buffer: no second bind
                pipeline.Draw(pass, meshB.Opaque.FirstQuad, meshB.Opaque.Count, 1);
                pass.End();
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var pixels = await ReadBackPixels(device, queue, colorTex, 64, 64);
                // 8 px per block: block x 0..1 -> px 8..16 (A), x 4..5 -> px 40..48 (B)
                BrExpect(pixels, 64, 12, 32, 0, 255, 255, "column A's cyan block at origin 0");
                var (r, g, bl, _) = GetPixel(pixels, 64, 44, 32);
                if (!(r > 200 && g > 100 && bl < 40)) throw new Exception($"column B should show its orange block at x 4, got ({r},{g},{bl}): its quads were not drawn from its own range");
                BrExpect(pixels, 64, 28, 32, 0, 0, 0, "nothing between the two columns");
            }
            finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
            await accelerator.SynchronizeAsync();
        });

        [TestMethod]
        public async Task QuadArena_HeightmapMesher_PooledMatchesTheReferenceTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            using var arena = new QuadArena(accelerator, 1 << 14);
            using var mesher = new HeightmapMesher(accelerator, 24) { SeabedTypeOffset = HmSeabedOffset, EmbedSectionIndex = true, Arena = arena };
            mesher.SetBlockClasses(CmFlags);
            for (int seed = 1; seed <= 3; seed++)
                await HmMeshAndVerify(accelerator, mesher, HmTerrain(seed), new HmMap?[] { HmTerrain(seed + 10), null, HmTerrain(seed + 30), null }, $"pooled heightmap seed {seed}");
            if (arena.UsedQuads != 0) throw new Exception($"{arena.UsedQuads} quads leaked");
            if (arena.CapacityQuads == 0) throw new Exception("the meshes were not pooled");
        });
    }
}
