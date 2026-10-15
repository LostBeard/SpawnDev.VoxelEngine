using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // ColumnMesher: a whole block column meshed on the GPU (class-aware, all sections per dispatch), checked against a brute-force CPU
    // reference. The reference answers "which unit faces are visible"; the GPU's merged quads are expanded back to unit faces and the two
    // sets must be IDENTICAL: same faces, each covered exactly once, same block type. Nothing about the greedy merge itself is assumed.
    public abstract partial class VoxelEngineTestBase
    {
        // Block types used by these tests, and their classes
        private const ushort CmAir = 0, CmStone = 1, CmDirt = 2, CmLeavesA = 3, CmLeavesB = 4, CmWater = 5, CmPlant = 6, CmGlass = 7;
        private static readonly int[] CmFlags =
        {
            BlockClass.None,    // air
            BlockClass.Opaque,  // stone
            BlockClass.Opaque,  // dirt
            BlockClass.Cutout,  // leaves A
            BlockClass.Cutout,  // leaves B
            BlockClass.Water,   // water
            BlockClass.Cross,   // plant
            BlockClass.Cutout,  // glass
            BlockClass.EmitOpaque | BlockClass.OccludeOpaque | BlockClass.EmitWater | BlockClass.OccludeWater, // a custom class drawn in both passes
        };
        private const ushort CmBoth = 8;

        private const int CmXZ = 16;

        private static void CmRequireGpuMeshing(Accelerator accelerator)
        {
            // The kernels use atomics (quad counters, mask clears) and write several elements per thread: not WebGL
            if (!accelerator.Device.Satisfies(new AcceleratorRequirements { RequiresAtomics = true, RequiresScatterStores = true, RequiresInt64 = true }))
                throw new UnsupportedTestException($"ColumnMesher needs atomics + scatter stores; {accelerator.GetType().Name} has not");
        }

        private static int CmIdx(int x, int y, int z) => x + z * CmXZ + y * CmXZ * CmXZ;

        private static ushort[] CmEmpty(int sections) => new ushort[CmXZ * CmXZ * 16 * sections];

        // ---- the CPU reference ------------------------------------------------------------------------------------------------

        private readonly record struct CmFace(int Face, int X, int Y, int Z, int Type);

        private static readonly (int dx, int dy, int dz)[] CmDirs = { (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0) };

        /// <summary>Block type at (x, y, z) of the column; -1 for a cell in a missing neighbour column; 0 above or below the column.</summary>
        private static int CmBlock(ushort[] col, ushort[]?[] nbr, int sections, int x, int y, int z)
        {
            if (y < 0 || y >= sections * 16) return 0;
            if (x >= 0 && x < CmXZ && z >= 0 && z < CmXZ) return col[CmIdx(x, y, z)] & 0xFFF;
            ushort[]? n;
            int lx = x, lz = z;
            if (x < 0 && z >= 0 && z < CmXZ) { n = nbr[0]; lx = CmXZ - 1; }
            else if (x >= CmXZ && z >= 0 && z < CmXZ) { n = nbr[1]; lx = 0; }
            else if (z < 0 && x >= 0 && x < CmXZ) { n = nbr[2]; lz = CmXZ - 1; }
            else if (z >= CmXZ && x >= 0 && x < CmXZ) { n = nbr[3]; lz = 0; }
            else return 0; // a ring corner: never a face neighbour
            return n == null ? -1 : n[CmIdx(lx, y, lz)] & 0xFFF;
        }

        private static HashSet<CmFace> CmExpectedCubeFaces(ushort[] col, ushort[]?[] nbr, int sections, int passShift, bool missingSolid)
        {
            var set = new HashSet<CmFace>();
            for (int y = 0; y < sections * 16; y++)
                for (int z = 0; z < CmXZ; z++)
                    for (int x = 0; x < CmXZ; x++)
                    {
                        int type = col[CmIdx(x, y, z)] & 0xFFF;
                        if (((CmFlags[type] >> passShift) & 1) == 0) continue; // does not emit in this pass
                        for (int f = 0; f < 6; f++)
                        {
                            var (dx, dy, dz) = CmDirs[f];
                            int n = CmBlock(col, nbr, sections, x + dx, y + dy, z + dz);
                            bool occluded = n < 0 ? missingSolid : ((CmFlags[n] >> (passShift + 1)) & 1) != 0;
                            if (!occluded) set.Add(new CmFace(f, x, y, z, type));
                        }
                    }
            return set;
        }

        private static HashSet<CmFace> CmExpectedPlants(ushort[] col, int sections)
        {
            var set = new HashSet<CmFace>();
            for (int y = 0; y < sections * 16; y++)
                for (int z = 0; z < CmXZ; z++)
                    for (int x = 0; x < CmXZ; x++)
                    {
                        int type = col[CmIdx(x, y, z)] & 0xFFF;
                        if ((CmFlags[type] & BlockClass.Plant) == 0) continue;
                        set.Add(new CmFace(6, x, y, z, type));
                        set.Add(new CmFace(7, x, y, z, type));
                    }
            return set;
        }

        // ---- decoding the GPU's quads ------------------------------------------------------------------------------------------

        /// <summary>Expand a quad list to its unit faces. Throws if two quads cover the same unit face, or a quad strays out of its section.</summary>
        private static async Task<HashSet<CmFace>> CmDecode(ColumnMesh.QuadList list, string pass)
        {
            var set = new HashSet<CmFace>();
            if (list.Buffer == null) return set;
            var quads = await list.Buffer.CopyToHostAsync();
            if (quads.Length != list.Count) throw new Exception($"{pass}: buffer holds {quads.Length} quads, list says {list.Count}");
            for (int s = 0; s < list.SectionCount; s++)
                for (int q = list.SectionStart[s]; q < list.SectionStart[s + 1]; q++)
                {
                    PackedQuad.Unpack(quads[q], out int x, out int y, out int z, out int w, out int h, out int face, out int type);
                    int baseY = s * 16;
                    for (int i = 0; i < w; i++)
                        for (int j = 0; j < h; j++)
                        {
                            CmFace unit = face switch
                            {
                                0 or 1 => new CmFace(face, x, baseY + y + j, z + i, type),   // X faces: width along z, height along y
                                2 or 3 => new CmFace(face, x + i, baseY + y + j, z, type),   // Z faces: width along x, height along y
                                4 or 5 => new CmFace(face, x + i, baseY + y, z + j, type),   // Y faces: width along x, height along z
                                _ => new CmFace(face, x, baseY + y, z, type),                // plant diagonals
                            };
                            if (!set.Add(unit)) throw new Exception($"{pass}: unit face {unit} is covered by two quads");
                        }
                }
            return set;
        }

        private static void CmAssertSame(HashSet<CmFace> expected, HashSet<CmFace> actual, string what)
        {
            if (expected.SetEquals(actual)) return;
            var missing = expected.Except(actual).Take(3).ToList();
            var extra = actual.Except(expected).Take(3).ToList();
            throw new Exception($"{what}: {expected.Count} faces expected, {actual.Count} meshed. " +
                $"Missing e.g. {string.Join(", ", missing)}; unexpected e.g. {string.Join(", ", extra)}");
        }

        private static MemoryBuffer1D<ushort, Stride1D.Dense> CmUpload(Accelerator accelerator, ushort[] ids) => accelerator.Allocate1D(ids);

        /// <summary>Mesh and compare all three passes with the reference.</summary>
        private static async Task CmMeshAndVerify(Accelerator accelerator, ColumnMesher mesher, ushort[] col, ushort[]?[] nbr, int sections, string what)
        {
            using var gpuCol = CmUpload(accelerator, col);
            var gpuNbr = nbr.Select(n => n == null ? null : CmUpload(accelerator, n)).ToArray();
            try
            {
                using var mesh = await mesher.MeshAsync(gpuCol, gpuNbr[0], gpuNbr[1], gpuNbr[2], gpuNbr[3]);
                if (mesh.Overflow != 0) throw new Exception($"{what}: {mesh.Overflow} quads overflowed the scratch space");
                bool solid = mesher.MissingNeighborsAreSolid;
                CmAssertSame(CmExpectedCubeFaces(col, nbr, sections, BlockClass.OpaqueShift, solid), await CmDecode(mesh.Opaque, "opaque"), what + " / opaque pass");
                CmAssertSame(CmExpectedCubeFaces(col, nbr, sections, BlockClass.WaterShift, solid), await CmDecode(mesh.Water, "water"), what + " / water pass");
                CmAssertSame(CmExpectedPlants(col, sections), await CmDecode(mesh.Plants, "plants"), what + " / plants");
            }
            finally { foreach (var b in gpuNbr) b?.Dispose(); }
        }

        // =====================================================================
        // Small, exact scenes
        // =====================================================================

        [TestMethod]
        public async Task ColumnMesher_SingleBlock_SixQuadsAndNothingElseTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 3;
            using var mesher = new ColumnMesher(accelerator, sections);
            mesher.SetBlockClasses(CmFlags);
            var col = CmEmpty(sections);
            col[CmIdx(5, 20, 7)] = CmStone;

            using var gpu = CmUpload(accelerator, col);
            using var mesh = await mesher.MeshAsync(gpu);
            if (mesh.Opaque.Count != 6) throw new Exception($"One block must mesh to 6 quads, got {mesh.Opaque.Count}");
            if (mesh.Plants.Count != 0 || mesh.Water.Count != 0) throw new Exception("No plants or water were placed");
            // y 20 is in section 1: the quads are in that section's slice, and only there
            if (mesh.Opaque.QuadsIn(1) != 6 || mesh.Opaque.QuadsIn(0) != 0 || mesh.Opaque.QuadsIn(2) != 0)
                throw new Exception($"The block is in section 1; the quads per section are {string.Join(",", Enumerable.Range(0, sections).Select(mesh.Opaque.QuadsIn))}");
            await CmMeshAndVerify(accelerator, mesher, col, new ushort[]?[4], sections, "single block");
        });

        [TestMethod]
        public async Task ColumnMesher_FlatSlab_MergesToOneQuadPerFaceTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 2;
            using var mesher = new ColumnMesher(accelerator, sections) { MissingNeighborsAreSolid = false };
            mesher.SetBlockClasses(CmFlags);
            var col = CmEmpty(sections);
            for (int z = 0; z < CmXZ; z++)
                for (int x = 0; x < CmXZ; x++)
                    col[CmIdx(x, 5, z)] = CmStone;

            using var gpu = CmUpload(accelerator, col);
            using var mesh = await mesher.MeshAsync(gpu);
            // top and bottom: one 16x16 quad each; the four 16x1 sides: one quad each
            if (mesh.Opaque.Count != 6) throw new Exception($"A 16x16x1 slab must greedy-merge to 6 quads, got {mesh.Opaque.Count}");
            await CmMeshAndVerify(accelerator, mesher, col, new ushort[]?[4], sections, "flat slab");

            // With solid neighbours assumed, only the top and bottom remain
            mesher.MissingNeighborsAreSolid = true;
            using var closed = await mesher.MeshAsync(gpu);
            if (closed.Opaque.Count != 2) throw new Exception($"With solid missing neighbours the slab has 2 quads (top, bottom), got {closed.Opaque.Count}");
        });

        [TestMethod]
        public async Task ColumnMesher_EmptyColumn_IsEmptyTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            using var mesher = new ColumnMesher(accelerator, 2);
            mesher.SetBlockClasses(CmFlags);
            using var gpu = CmUpload(accelerator, CmEmpty(2));
            using var mesh = await mesher.MeshAsync(gpu);
            if (!mesh.IsEmpty || mesh.Opaque.Buffer != null) throw new Exception("An all-air column has no quads and no buffers");
        });

        [TestMethod]
        public async Task ColumnMesher_ClassRules_OneByOneTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 1;
            using var mesher = new ColumnMesher(accelerator, sections);
            mesher.SetBlockClasses(CmFlags);
            var col = CmEmpty(sections);
            col[CmIdx(2, 2, 3)] = CmStone; col[CmIdx(3, 2, 3)] = CmLeavesA;   // stone | leaf
            col[CmIdx(2, 8, 3)] = CmStone; col[CmIdx(3, 8, 3)] = CmWater;     // stone | water
            col[CmIdx(2, 12, 3)] = CmWater; col[CmIdx(3, 12, 3)] = CmWater;   // water | water

            using var gpu = CmUpload(accelerator, col);
            using var mesh = await mesher.MeshAsync(gpu);
            var opaque = await CmDecode(mesh.Opaque, "opaque");
            var water = await CmDecode(mesh.Water, "water");

            // 1. A leaf does not occlude: the stone's face toward it (+X at (2,2,3)) is drawn.
            if (!opaque.Contains(new CmFace(0, 2, 2, 3, CmStone))) throw new Exception("A leaf must not hide the stone's +X face");
            // 2. Stone occludes: the leaf's face toward the stone (-X at (3,2,3)) is NOT drawn.
            if (opaque.Contains(new CmFace(1, 3, 2, 3, CmLeavesA))) throw new Exception("Stone must hide the leaf's -X face");
            // 3. Water does not occlude in the opaque pass: the stone's face toward the water is drawn.
            if (!opaque.Contains(new CmFace(0, 2, 8, 3, CmStone))) throw new Exception("Water must not hide the stone's +X face in the opaque pass");
            // 4. Stone occludes water: the water's face toward the stone is NOT drawn in the water pass.
            if (water.Contains(new CmFace(1, 3, 8, 3, CmWater))) throw new Exception("Stone must hide the water's -X face");
            // 5. Water occludes water: no face between two water cells.
            if (water.Contains(new CmFace(0, 2, 12, 3, CmWater)) || water.Contains(new CmFace(1, 3, 12, 3, CmWater)))
                throw new Exception("Water must hide the face between two water cells");
            // 6. Water is never in the opaque pass.
            if (opaque.Any(f => f.Type == CmWater)) throw new Exception("Water must not appear in the opaque pass");
        });

        [TestMethod]
        public async Task ColumnMesher_NeighborColumns_CullTheSharedFacesTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 2;
            using var mesher = new ColumnMesher(accelerator, sections) { MissingNeighborsAreSolid = false };
            mesher.SetBlockClasses(CmFlags);
            var solid = CmEmpty(sections);
            for (int i = 0; i < solid.Length; i++) solid[i] = CmStone;

            using var col = CmUpload(accelerator, solid);
            using var east = CmUpload(accelerator, solid);
            using var alone = await mesher.MeshAsync(col);
            using var withEast = await mesher.MeshAsync(col, posX: east);
            // alone (missing = air): the +X wall is visible, so there are more quads than with a solid column to the east
            if (withEast.Opaque.Count >= alone.Opaque.Count)
                throw new Exception($"A solid +X neighbour must hide the +X wall: {withEast.Opaque.Count} quads with it, {alone.Opaque.Count} without");
            var expected = CmExpectedCubeFaces(solid, new ushort[]?[] { null, solid, null, null }, sections, BlockClass.OpaqueShift, false);
            CmAssertSame(expected, await CmDecode(withEast.Opaque, "opaque"), "solid column with an east neighbour");
        });

        // =====================================================================
        // Randomized: the GPU mesh equals the reference
        // =====================================================================

        private static ushort[] CmNoise(int sections, int seed, int airPercent = 55)
        {
            var rng = new Random(seed);
            var col = CmEmpty(sections);
            for (int i = 0; i < col.Length; i++)
            {
                int r = rng.Next(100);
                if (r < airPercent) continue;
                int t = rng.Next(100);
                col[i] = t switch { < 40 => CmStone, < 50 => CmDirt, < 60 => CmLeavesA, < 66 => CmLeavesB, < 80 => CmWater, < 92 => CmPlant, _ => CmGlass };
            }
            return col;
        }

        /// <summary>Terrain-like: a smooth height field of stone and dirt, water up to a sea level, plants on top, a few leaf blobs.</summary>
        private static ushort[] CmTerrain(int sections, int seed)
        {
            var rng = new Random(seed);
            int height = sections * 16;
            var col = CmEmpty(sections);
            int sea = height / 3;
            double fx = rng.NextDouble() * 0.5 + 0.1, fz = rng.NextDouble() * 0.5 + 0.1;
            for (int z = 0; z < CmXZ; z++)
                for (int x = 0; x < CmXZ; x++)
                {
                    int top = (int)(sea + 6 * Math.Sin(x * fx) + 6 * Math.Cos(z * fz) + rng.Next(3));
                    top = Math.Clamp(top, 1, height - 2);
                    for (int y = 0; y <= top; y++) col[CmIdx(x, y, z)] = y == top ? CmDirt : CmStone;
                    for (int y = top + 1; y <= sea; y++) col[CmIdx(x, y, z)] = CmWater;
                    if (top > sea && rng.Next(4) == 0) col[CmIdx(x, top + 1, z)] = CmPlant;
                    if (top > sea && rng.Next(40) == 0)
                        for (int dy = 1; dy <= 3 && top + dy < height; dy++) col[CmIdx(x, top + dy, z)] = CmLeavesA;
                }
            // an air pocket straddling a section boundary
            for (int y = 14; y <= 18; y++) col[CmIdx(8, y, 8)] = CmAir;
            return col;
        }

        [TestMethod]
        public async Task ColumnMesher_RandomNoiseColumns_MatchTheCpuReferenceTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 3;
            using var mesher = new ColumnMesher(accelerator, sections);
            mesher.SetBlockClasses(CmFlags);
            for (int seed = 1; seed <= 4; seed++)
            {
                var col = CmNoise(sections, seed, airPercent: 40 + seed * 10);
                // neighbour presence varies: all four, none, and a mix; missing neighbours both solid and open
                var variants = new (bool xm, bool xp, bool zm, bool zp, bool solid)[]
                {
                    (true, true, true, true, true), (false, false, false, false, true), (false, false, false, false, false),
                    (true, false, true, false, true), (false, true, false, true, false),
                };
                foreach (var v in variants)
                {
                    mesher.MissingNeighborsAreSolid = v.solid;
                    var nbr = new ushort[]?[]
                    {
                        v.xm ? CmNoise(sections, seed * 10 + 1) : null, v.xp ? CmNoise(sections, seed * 10 + 2) : null,
                        v.zm ? CmNoise(sections, seed * 10 + 3) : null, v.zp ? CmNoise(sections, seed * 10 + 4) : null,
                    };
                    await CmMeshAndVerify(accelerator, mesher, col, nbr, sections, $"noise seed {seed} neighbours {(v.xm ? "-X" : "")}{(v.xp ? "+X" : "")}{(v.zm ? "-Z" : "")}{(v.zp ? "+Z" : "")} solid={v.solid}");
                }
            }
        });

        [TestMethod]
        public async Task ColumnMesher_TerrainColumns_FullHeight_MatchTheCpuReferenceTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 24; // Minecraft's 384 blocks
            using var mesher = new ColumnMesher(accelerator, sections);
            mesher.SetBlockClasses(CmFlags);
            for (int seed = 1; seed <= 3; seed++)
            {
                var col = CmTerrain(sections, seed);
                var nbr = new ushort[]?[] { CmTerrain(sections, seed + 100), CmTerrain(sections, seed + 200), CmTerrain(sections, seed + 300), null };
                mesher.MissingNeighborsAreSolid = true;
                await CmMeshAndVerify(accelerator, mesher, col, nbr, sections, $"terrain seed {seed}");
            }
        });

        // =====================================================================
        // Bounds and reuse
        // =====================================================================

        [TestMethod]
        public async Task ColumnMesher_WorstCase_ThreeDCheckerboard_FillsTheScratchWithoutOverflowTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 2;
            // Missing neighbours open (not solid): the four side walls draw too, so the count is the true worst case. (With the default
            // solid-neighbour rule the walls toward unloaded columns are hidden: 24576 - 4 x 256 = 23552 quads in a section.)
            using var mesher = new ColumnMesher(accelerator, sections) { MissingNeighborsAreSolid = false };
            mesher.SetBlockClasses(CmFlags);
            // Every cell leaves, alternating type A/B in a 3D checkerboard: leaves occlude nothing, so every cell draws all 6 faces, and
            // neighbouring coplanar faces differ in type so nothing merges: 4096 cells x 6 = 24576 quads per section = the scratch size
            var col = CmEmpty(sections);
            for (int y = 0; y < sections * 16; y++)
                for (int z = 0; z < CmXZ; z++)
                    for (int x = 0; x < CmXZ; x++)
                        col[CmIdx(x, y, z)] = ((x + y + z) & 1) == 0 ? CmLeavesA : CmLeavesB;

            using var gpu = CmUpload(accelerator, col);
            using var mesh = await mesher.MeshAsync(gpu);
            if (mesh.Overflow != 0) throw new Exception($"{mesh.Overflow} quads overflowed: the per-section scratch bound is wrong");
            for (int s = 0; s < sections; s++)
                if (mesh.Opaque.QuadsIn(s) != ColumnMeshKernels.ScratchQuadsPerSection)
                    throw new Exception($"Section {s} has {mesh.Opaque.QuadsIn(s)} quads, the worst case is exactly {ColumnMeshKernels.ScratchQuadsPerSection}");
        });

        [TestMethod]
        public async Task ColumnMesher_ReusedForAnotherColumn_LeaksNothingOfTheLastOneTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 3;
            using var mesher = new ColumnMesher(accelerator, sections);
            mesher.SetBlockClasses(CmFlags);
            // A busy column first, then a sparse one, then an empty one: stale face masks, counters or scratch would show up
            await CmMeshAndVerify(accelerator, mesher, CmNoise(sections, 7, 30), new ushort[]?[4], sections, "busy column");
            var sparse = CmEmpty(sections);
            sparse[CmIdx(1, 1, 1)] = CmStone;
            await CmMeshAndVerify(accelerator, mesher, sparse, new ushort[]?[4], sections, "sparse column after a busy one");
            using var gpu = CmUpload(accelerator, CmEmpty(sections));
            using var empty = await mesher.MeshAsync(gpu);
            if (!empty.IsEmpty) throw new Exception("An all-air column after a busy one must mesh to nothing");
        });

        [TestMethod]
        public async Task ColumnMesher_RejectsAColumnOfTheWrongSizeTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            using var mesher = new ColumnMesher(accelerator, 2);
            using var wrong = accelerator.Allocate1D<ushort>(100);
            bool threw = false;
            try { await mesher.MeshAsync(wrong); }
            catch (ArgumentException) { threw = true; }
            if (!threw) throw new Exception("A column buffer of the wrong length must be rejected");
        });

        // =====================================================================
        // The zero-copy ingest: block ids from a JavaScript typed array straight into the GPU buffer
        // =====================================================================

        [TestMethod]
        public async Task ColumnMesher_ColumnFromAJsTypedArray_MeshesLikeAManagedOneTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            if (!OperatingSystem.IsBrowser())
                throw new UnsupportedTestException("A JavaScript typed array exists only in the browser");
            const int sections = 3;
            using var mesher = new ColumnMesher(accelerator, sections);
            mesher.SetBlockClasses(CmFlags);
            var ids = CmTerrain(sections, 5);

            // The ids are JS bytes (here made from a managed array, as a test must; a real caller has a Uint8Array view of a fetched
            // ArrayBuffer): CreateColumnBufferFromJS uploads them with CopyFromJS and they never become a managed array on the way in
            var bytes = new byte[ids.Length * 2];
            Buffer.BlockCopy(ids, 0, bytes, 0, bytes.Length);
            using var js = new Uint8Array(bytes);
            MemoryBuffer1D<ushort, Stride1D.Dense> buffer;
            try { buffer = mesher.CreateColumnBufferFromJS(js); }
            catch (NotSupportedException ex) { throw new UnsupportedTestException(ex.Message); }
            using var _ = buffer;

            using var mesh = await mesher.MeshAsync(buffer);
            if (mesh.Overflow != 0) throw new Exception("overflow");
            CmAssertSame(CmExpectedCubeFaces(ids, new ushort[]?[4], sections, BlockClass.OpaqueShift, true), await CmDecode(mesh.Opaque, "opaque"), "JS-uploaded column / opaque");
            CmAssertSame(CmExpectedPlants(ids, sections), await CmDecode(mesh.Plants, "plants"), "JS-uploaded column / plants");
        });

        // =====================================================================
        // The section index inside the quad (one draw per column instead of one per section)
        // =====================================================================

        [TestMethod]
        public void PackedQuad_SectionIndex_RoundTripsAndNeverDisturbsTheOtherFields()
        {
            for (int s = 0; s < 32; s++)
            {
                long q = PackedQuad.Pack(3, 5, 7, 4, 2, 4, 1234, damage: 9, ao: 0xA5, sectionIndex: s);
                if (PackedQuad.UnpackSectionIndex(q) != s) throw new Exception($"section {s} came back as {PackedQuad.UnpackSectionIndex(q)}");
                PackedQuad.UnpackFull(q, out int x, out int y, out int z, out int w, out int h, out int face, out int type, out int damage, out int ao);
                if ((x, y, z, w, h, face, type, damage, ao) != (3, 5, 7, 4, 2, 4, 1234, 9, 0xA5))
                    throw new Exception($"section {s} disturbed another field: {(x, y, z, w, h, face, type, damage, ao)}");
            }
            // Replacing an embedded index, and the default of none
            long a = PackedQuad.Pack(1, 2, 3, 1, 1, 0, 7);
            if (PackedQuad.UnpackSectionIndex(a) != 0) throw new Exception("a quad packed without a section index has index 0");
            long b = PackedQuad.WithSectionIndex(PackedQuad.WithSectionIndex(a, 20), 3);
            if (PackedQuad.UnpackSectionIndex(b) != 3) throw new Exception("WithSectionIndex must replace the previous index");
            PackedQuad.Unpack(b, out int bx, out int by, out int bz, out _, out _, out int bf, out int bt);
            if ((bx, by, bz, bf, bt) != (1, 2, 3, 0, 7)) throw new Exception("WithSectionIndex must leave the rest of the quad alone");
        }

        [TestMethod]
        public async Task ColumnMesher_EmbedSectionIndex_EveryQuadCarriesItsSection_AndTheFacesAreUnchangedTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 6;
            var col = CmTerrain(sections, 11);
            foreach (bool embed in new[] { false, true })
            {
                using var mesher = new ColumnMesher(accelerator, sections) { EmbedSectionIndex = embed };
                mesher.SetBlockClasses(CmFlags);
                using var gpu = CmUpload(accelerator, col);
                using var mesh = await mesher.MeshAsync(gpu);
                if (mesh.Overflow != 0) throw new Exception("overflow");
                foreach (var (list, name) in new[] { (mesh.Opaque, "opaque"), (mesh.Plants, "plants"), (mesh.Water, "water") })
                {
                    if (list.Buffer == null) continue;
                    var quads = await list.Buffer.CopyToHostAsync();
                    int seen = 0;
                    for (int s = 0; s < sections; s++)
                        for (int q = list.SectionStart[s]; q < list.SectionStart[s + 1]; q++, seen++)
                        {
                            int index = PackedQuad.UnpackSectionIndex(quads[q]);
                            int want = embed ? s : 0;
                            if (index != want) throw new Exception($"embed={embed}: a {name} quad of section {s} carries section index {index}, expected {want}");
                        }
                    if (seen == 0) throw new Exception($"{name}: no quads were checked");
                }
                // The embedded bits change nothing about which faces are drawn
                await CmMeshAndVerify(accelerator, mesher, col, new ushort[]?[4], sections, $"embed={embed}");
            }
        });    }
}
