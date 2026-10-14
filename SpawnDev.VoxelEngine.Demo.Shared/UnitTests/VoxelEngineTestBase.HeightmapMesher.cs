using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // HeightmapMesher: a chunk's heightmap (top block per column, plus the seabed under water) meshed on the GPU into per-section quads.
    // Like the column tests, the GPU's quads are expanded to unit faces and must equal a CPU reference EXACTLY. The reference states the
    // rules in plain terms (below), column by column, with no quads and no sections in it.
    //   - a column's SOLID TOP is its top block's Y, or under water its seabed's Y (none: below the world); plants, air and unknown = none
    //   - a solid column draws a top face and, on each side, a wall over the blocks above the neighbour's solid top
    //   - a water column draws a water-pass top face, then its seabed as a solid column of type + offset
    //   - a neighbour that is not loaded is bottomless (the wall reaches the floor)
    public abstract partial class VoxelEngineTestBase
    {
        private const int HmFloor = -65;      // "below the lowest block"
        private const int HmSeabedOffset = 100;

        private sealed class HmMap
        {
            public int[] Height = new int[256];
            public ushort[] Type = new ushort[256];
            public int[] SeabedHeight = Enumerable.Repeat(-64, 256).ToArray();
            public ushort[] SeabedType = new ushort[256];

            public int[] Pack()
            {
                var ints = new int[HeightmapLayout.Ints];
                System.Array.Copy(Height, 0, ints, HeightmapLayout.HeightsOffset, 256);
                System.Array.Copy(SeabedHeight, 0, ints, HeightmapLayout.SeabedHeightsOffset, 256);
                for (int c = 0; c < 256; c++)
                {
                    ints[HeightmapLayout.TypesOffset + c / 2] |= Type[c] << ((c & 1) * 16);
                    ints[HeightmapLayout.SeabedTypesOffset + c / 2] |= SeabedType[c] << ((c & 1) * 16);
                }
                return ints;
            }
        }

        private static bool HmIsWater(int type) => (CmFlags[type] & BlockClass.EmitWater) != 0 && (CmFlags[type] & BlockClass.EmitOpaque) == 0;
        private static bool HmDrawable(int type) => type != 0 && type < CmFlags.Length && CmFlags[type] != 0 && (CmFlags[type] & BlockClass.Plant) == 0;

        private static int HmSolidTop(HmMap m, int cell)
        {
            int t = m.Type[cell];
            if (!HmDrawable(t)) return HmFloor;
            if (HmIsWater(t)) return HmDrawable(m.SeabedType[cell]) ? m.SeabedHeight[cell] : HmFloor;
            return m.Height[cell];
        }

        // y of the result = column grid Y (world Y + 64), as CmDecode reports it
        private static void HmSolidColumn(HmMap m, HmMap?[] nbr, int x, int z, int top, int type, HashSet<CmFace> faces)
        {
            faces.Add(new CmFace(4, x, top + 64, z, type));
            var dirs = new (int face, int dx, int dz)[] { (0, 1, 0), (1, -1, 0), (2, 0, 1), (3, 0, -1) };
            foreach (var (face, dx, dz) in dirs)
            {
                int nx = x + dx, nz = z + dz;
                int nTop;
                if (nx >= 0 && nx < 16 && nz >= 0 && nz < 16) nTop = HmSolidTop(m, nx + nz * 16);
                else
                {
                    var n = nbr[dx < 0 ? 0 : dx > 0 ? 1 : dz < 0 ? 2 : 3];
                    nTop = n == null ? HmFloor : HmSolidTop(n, (nx & 15) + (nz & 15) * 16);
                }
                for (int y = nTop + 1; y <= top; y++)
                    if (y + 64 >= 0 && y + 64 < 384) faces.Add(new CmFace(face, x, y + 64, z, type));
            }
        }

        private static (HashSet<CmFace> opaque, HashSet<CmFace> water) HmExpected(HmMap m, HmMap?[] nbr, int seabedOffset)
        {
            var opaque = new HashSet<CmFace>();
            var water = new HashSet<CmFace>();
            for (int z = 0; z < 16; z++)
                for (int x = 0; x < 16; x++)
                {
                    int cell = x + z * 16;
                    int t = m.Type[cell];
                    if (!HmDrawable(t)) continue;
                    if (HmIsWater(t))
                    {
                        water.Add(new CmFace(4, x, m.Height[cell] + 64, z, t));
                        if (HmDrawable(m.SeabedType[cell]))
                            HmSolidColumn(m, nbr, x, z, m.SeabedHeight[cell], m.SeabedType[cell] + seabedOffset, opaque);
                    }
                    else HmSolidColumn(m, nbr, x, z, m.Height[cell], t, opaque);
                }
            return (opaque, water);
        }

        private static HmMap HmFlat(int height, ushort type = CmStone)
        {
            var m = new HmMap();
            System.Array.Fill(m.Height, height);
            System.Array.Fill(m.Type, type);
            return m;
        }

        /// <summary>Hills between lo and hi, with some water lakes (seabed below the surface), air holes and plants.</summary>
        private static HmMap HmTerrain(int seed, int lo = -60, int hi = 300)
        {
            var rnd = new Random(seed);
            var m = new HmMap();
            for (int c = 0; c < 256; c++)
            {
                m.Height[c] = rnd.Next(lo, hi);
                m.Type[c] = (ushort)(rnd.Next(4) switch { 0 => CmStone, 1 => CmDirt, 2 => CmLeavesA, _ => CmGlass });
                int roll = rnd.Next(20);
                if (roll == 0) m.Type[c] = CmAir;
                else if (roll == 1) m.Type[c] = CmPlant;
                else if (roll <= 4)
                {
                    m.Type[c] = CmWater;
                    m.SeabedHeight[c] = Math.Max(-64, m.Height[c] - rnd.Next(1, 40));
                    m.SeabedType[c] = rnd.Next(8) == 0 ? CmPlant : (ushort)(rnd.Next(2) == 0 ? CmStone : CmDirt);
                    if (rnd.Next(10) == 0) m.SeabedType[c] = CmAir; // water over nothing
                }
            }
            return m;
        }

        private static MemoryBuffer1D<int, Stride1D.Dense> HmUpload(Accelerator accelerator, HmMap m) => accelerator.Allocate1D(m.Pack());

        private static async Task HmMeshAndVerify(Accelerator accelerator, HeightmapMesher mesher, HmMap map, HmMap?[] nbr, string what)
        {
            using var gpu = HmUpload(accelerator, map);
            var gpuNbr = nbr.Select(n => n == null ? null : HmUpload(accelerator, n)).ToArray();
            try
            {
                using var mesh = await mesher.MeshAsync(gpu, gpuNbr[0], gpuNbr[1], gpuNbr[2], gpuNbr[3]);
                if (mesh.Overflow != 0) throw new Exception($"{what}: {mesh.Overflow} quads overflowed the scratch space");
                if (mesh.Plants.Count != 0) throw new Exception($"{what}: a heightmap mesh has no plants");
                var (opaque, water) = HmExpected(map, nbr, mesher.SeabedTypeOffset);
                CmAssertSame(opaque, await CmDecode(mesh.Opaque, "opaque"), what + " / opaque");
                CmAssertSame(water, await CmDecode(mesh.Water, "water"), what + " / water");
            }
            finally { foreach (var b in gpuNbr) b?.Dispose(); }
        }

        private static HeightmapMesher HmMesher(Accelerator accelerator, int seabedOffset = 0)
        {
            CmRequireGpuMeshing(accelerator);
            var mesher = new HeightmapMesher(accelerator, 24) { SeabedTypeOffset = seabedOffset };
            mesher.SetBlockClasses(CmFlags);
            return mesher;
        }

        // =====================================================================

        [TestMethod]
        public async Task HeightmapMesher_FlatChunk_TopsAndFourWallsToTheFloorTest() => await RunTest(async accelerator =>
        {
            using var mesher = HmMesher(accelerator);
            // Y 10, no neighbours loaded: 256 tops, and on the four borders a 16-wide wall from the floor (-64) to 10 = 75 blocks high
            await HmMeshAndVerify(accelerator, mesher, HmFlat(10), new HmMap?[4], "flat, no neighbours");

            using var gpu = HmUpload(accelerator, HmFlat(10));
            using var mesh = await mesher.MeshAsync(gpu);
            // 75 blocks high spans sections 0..4 (Y -64..15): 256 tops + 4 borders x 16 columns x ceil-split walls
            int tops = Enumerable.Range(0, 24).Sum(s => mesh.Opaque.QuadsIn(s));
            if (mesh.Opaque.Count < 256 + 4 * 16) throw new Exception($"Expected at least 256 tops + 64 wall strips, got {mesh.Opaque.Count} ({tops})");

            // With four identical neighbours there are no walls at all: only the 256 tops
            var same = HmFlat(10);
            await HmMeshAndVerify(accelerator, mesher, HmFlat(10), new HmMap?[] { same, same, same, same }, "flat, equal neighbours");
            using var gpuSame = HmUpload(accelerator, same);
            using var closed = await mesher.MeshAsync(gpu, gpuSame, gpuSame, gpuSame, gpuSame);
            if (closed.Opaque.Count != 256) throw new Exception($"A flat chunk among equals is 256 top quads, got {closed.Opaque.Count}");
        });

        [TestMethod]
        public async Task HeightmapMesher_EmptyChunk_IsEmptyTest() => await RunTest(async accelerator =>
        {
            using var mesher = HmMesher(accelerator);
            using var gpu = HmUpload(accelerator, new HmMap());
            using var mesh = await mesher.MeshAsync(gpu);
            if (!mesh.IsEmpty) throw new Exception("A chunk with no top blocks has no quads");
        });

        [TestMethod]
        public async Task HeightmapMesher_Terrain_MatchesTheReference_WithAndWithoutNeighboursTest() => await RunTest(async accelerator =>
        {
            using var mesher = HmMesher(accelerator);
            for (int seed = 1; seed <= 4; seed++)
            {
                var map = HmTerrain(seed);
                // every combination of loaded neighbours for the first seeds, a fixed mix for the rest
                for (int mask = 0; mask < 16; mask += seed <= 2 ? 1 : 5)
                {
                    var nbr = new HmMap?[4];
                    for (int i = 0; i < 4; i++) if (((mask >> i) & 1) != 0) nbr[i] = HmTerrain(seed * 10 + i);
                    await HmMeshAndVerify(accelerator, mesher, map, nbr, $"terrain seed {seed}, neighbours {mask:b4}");
                }
            }
        });

        [TestMethod]
        public async Task HeightmapMesher_Water_IsInItsOwnPass_AndTheSeabedIsTheOffsetTypeTest() => await RunTest(async accelerator =>
        {
            using var mesher = HmMesher(accelerator, HmSeabedOffset);
            var m = HmFlat(20, CmDirt);
            // a 4x4 lake: water at 20, seabed stone at 12
            for (int z = 4; z < 8; z++)
                for (int x = 4; x < 8; x++)
                {
                    m.Type[x + z * 16] = CmWater;
                    m.SeabedHeight[x + z * 16] = 12;
                    m.SeabedType[x + z * 16] = CmStone;
                }
            await HmMeshAndVerify(accelerator, mesher, m, new HmMap?[4], "lake");

            using var gpu = HmUpload(accelerator, m);
            using var mesh = await mesher.MeshAsync(gpu);
            var water = await CmDecode(mesh.Water, "water");
            var opaque = await CmDecode(mesh.Opaque, "opaque");
            if (water.Count != 16 || water.Any(f => f.Face != 4 || f.Type != CmWater)) throw new Exception($"The lake is 16 water top faces, got {water.Count}");
            if (!opaque.Contains(new CmFace(4, 5, 12 + 64, 5, CmStone + HmSeabedOffset))) throw new Exception("The seabed must be a top face of the offset type");
            // the shore wall: the dirt next to the lake stands 8 blocks above the seabed, and its wall toward the lake is complete (Y 13..20)
            for (int y = 13; y <= 20; y++)
                if (!opaque.Contains(new CmFace(1, 8, y + 64, 5, CmDirt))) throw new Exception($"The shore wall is missing its block at Y {y}");
            // ... and the lake's own seabed has walls toward the shore only where the shore is LOWER (it is not)
            if (opaque.Any(f => f.Type == CmStone + HmSeabedOffset && f.Face != 4)) throw new Exception("The lake's seabed stands below its shore: no walls");
        });

        [TestMethod]
        public async Task HeightmapMesher_CheckerboardAcrossSections_NeverOverflowsTheScratchTest() => await RunTest(async accelerator =>
        {
            using var mesher = HmMesher(accelerator);
            var m = new HmMap();
            for (int z = 0; z < 16; z++)
                for (int x = 0; x < 16; x++)
                {
                    m.Type[x + z * 16] = CmStone;
                    m.Height[x + z * 16] = ((x + z) & 1) == 0 ? 300 : -60; // tall pillars between pits: walls through every section
                }
            await HmMeshAndVerify(accelerator, mesher, m, new HmMap?[4], "checkerboard");
        });

        [TestMethod]
        public async Task HeightmapMesher_ReusedForAnotherChunk_LeaksNothingOfTheLastOneTest() => await RunTest(async accelerator =>
        {
            using var mesher = HmMesher(accelerator);
            await HmMeshAndVerify(accelerator, mesher, HmTerrain(7), new HmMap?[4], "busy chunk first");
            await HmMeshAndVerify(accelerator, mesher, HmFlat(0), new HmMap?[4], "quiet chunk after");
            await HmMeshAndVerify(accelerator, mesher, HmTerrain(8), new HmMap?[] { HmFlat(5), null, HmFlat(50), null }, "busy chunk again");
        });

        [TestMethod]
        public async Task HeightmapMesher_RejectsAHeightmapOfTheWrongSizeTest() => await RunTest(async accelerator =>
        {
            using var mesher = HmMesher(accelerator);
            using var wrong = accelerator.Allocate1D<int>(100);
            bool threw = false;
            try { await mesher.MeshAsync(wrong); }
            catch (ArgumentException) { threw = true; }
            if (!threw) throw new Exception("A heightmap buffer of the wrong length must be rejected");
        });

        [TestMethod]
        public async Task HeightmapMesher_HeightmapFromAJsTypedArray_MeshesLikeAManagedOneTest() => await RunTest(async accelerator =>
        {
            using var mesher = HmMesher(accelerator, HmSeabedOffset);
            if (!OperatingSystem.IsBrowser())
                throw new UnsupportedTestException("A JavaScript typed array exists only in the browser");
            var map = HmTerrain(3);
            var bytes = new byte[HeightmapLayout.Bytes];
            Buffer.BlockCopy(map.Pack(), 0, bytes, 0, bytes.Length);
            using var js = new Uint8Array(bytes);
            MemoryBuffer1D<int, Stride1D.Dense> buffer;
            try { buffer = mesher.CreateHeightmapBufferFromJS(js); }
            catch (NotSupportedException ex) { throw new UnsupportedTestException(ex.Message); }
            using var _ = buffer;

            using var mesh = await mesher.MeshAsync(buffer);
            if (mesh.Overflow != 0) throw new Exception("overflow");
            var (opaque, water) = HmExpected(map, new HmMap?[4], HmSeabedOffset);
            CmAssertSame(opaque, await CmDecode(mesh.Opaque, "opaque"), "JS-uploaded heightmap / opaque");
            CmAssertSame(water, await CmDecode(mesh.Water, "water"), "JS-uploaded heightmap / water");
        });
    }
}
