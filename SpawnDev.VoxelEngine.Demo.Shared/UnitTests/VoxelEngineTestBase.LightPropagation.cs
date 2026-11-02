using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Lighting;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // LIGHT PROPAGATION: the sky light and block light of a column worked out on the GPU from its blocks and its neighbours', checked against a flood fill written the other way round.
    public abstract partial class VoxelEngineTestBase
    {
        private const int LpSections = 4, LpHeight = LpSections * 16;
        private const ushort LpAir = 0, LpStone = 1, LpGlass = 2, LpTorch = 3, LpLava = 4, LpLeaves = 5, LpMurk = 6;

        private static void LpTable(out byte[] passes, out byte[] emission, out byte[] dims)
        {
            passes = new byte[] { 1, 0, 1, 1, 0, 1, 1 };     // air, glass, torch, leaves and murk pass light; stone and lava do not
            emission = new byte[] { 0, 0, 0, 14, 15, 0, 0 }; // a torch gives 14, lava 15
            dims = new byte[] { 0, 0, 0, 0, 0, 1, 3 };       // leaves take a level off what passes, murk three
        }

        // A column of terrain with a roof over part of it, caves, a torch and a pool of lava: the same seed, the same column
        private static ushort[] LpColumn(int seed)
        {
            var rng = new Random(seed);
            var col = new ushort[LpHeight * 256];
            for (int z = 0; z < 16; z++)
                for (int x = 0; x < 16; x++)
                {
                    int top = 20 + (int)(8 * Math.Sin(x * 0.4 + seed) + 6 * Math.Cos(z * 0.3)) + rng.Next(3);
                    for (int y = 0; y <= top; y++) col[x + z * 16 + y * 256] = LpStone;
                    // a roof slab over the left half, high up, with glass windows
                    if (x < 8 && z < 12) col[x + z * 16 + 44 * 256] = (x + z) % 5 == 0 ? LpGlass : LpStone;
                }
            // a cave through the stone, a torch inside it, lava somewhere else
            for (int y = 8; y < 14; y++) for (int x = 3; x < 12; x++) col[x + 7 * 16 + y * 256] = LpAir;
            col[6 + 7 * 16 + 11 * 256] = LpTorch;
            col[12 + 3 * 16 + 9 * 256] = LpLava;
            col[1 + 14 * 16 + 40 * 256] = LpTorch;
            // a canopy of leaves two thick over a patch of ground, a pane of murk over another
            for (int y = 38; y < 40; y++) for (int z = 9; z < 15; z++) for (int x = 9; x < 15; x++) col[x + z * 16 + y * 256] = LpLeaves;
            for (int z = 1; z < 5; z++) for (int x = 1; x < 5; x++) col[x + z * 16 + 41 * 256] = LpMurk;
            return col;
        }

        // The other way round: a flood fill from the sources, nearest first, over the 3 x 3 columns (the corners not loaded, a missing neighbour open sky)
        private static byte[] LpReference(ushort[] centre, ushort[]?[] side, LightEmitter[]? emitters = null, int columnX = 0, int columnZ = 0)
        {
            int R = 48, M = 16, H = LpHeight;
            LpTable(out var passes, out var emission, out var dimsTable);
            int Dims(int t) => t >= 0 && t < dimsTable.Length ? dimsTable[t] : 0;
            bool Passes(int t) => t < passes.Length && passes[t] != 0;
            int Glow(int t) => t < emission.Length ? emission[t] : 0;
            var type = new int[H, R, R];
            var loaded = new bool[R, R];
            for (int rz = 0; rz < R; rz++)
                for (int rx = 0; rx < R; rx++)
                {
                    int x = rx - M, z = rz - M;
                    ushort[]? src = null; int sx = 0, sz = 0;
                    bool inX = x >= 0 && x < 16, inZ = z >= 0 && z < 16;
                    if (inX && inZ) { src = centre; sx = x; sz = z; }
                    else if (x < 0 && inZ) { src = side[0]; sx = x + 16; sz = z; }
                    else if (x >= 16 && inZ) { src = side[1]; sx = x - 16; sz = z; }
                    else if (z < 0 && inX) { src = side[2]; sx = x; sz = z + 16; }
                    else if (z >= 16 && inX) { src = side[3]; sx = x; sz = z - 16; }
                    loaded[rz, rx] = src != null;
                    for (int y = 0; y < H; y++) type[y, rz, rx] = src != null ? src[sx + sz * 16 + y * 256] : -1;   // -1: not loaded
                }
            bool Open(int y, int rz, int rx) => type[y, rz, rx] < 0 || Passes(type[y, rz, rx]);

            var sky = new int[H, R, R];
            var block = new int[H, R, R];
            // Sources: sky 15 above the first block that stops light in each loaded column, and in every cell that is not loaded; every glowing block
            var skyQ = new List<(int y, int z, int x)>();
            var blockQ = new List<(int y, int z, int x)>();
            for (int rz = 0; rz < R; rz++)
                for (int rx = 0; rx < R; rx++)
                {
                    bool blocked = false, dimmed = false;
                    for (int y = H - 1; y >= 0; y--)
                    {
                        int t = type[y, rz, rx];
                        if (t >= 0 && !Passes(t)) blocked = true;
                        if (!blocked && !dimmed)
                        {
                            int level = 15 - Dims(t);
                            if (Dims(t) > 0) dimmed = true;
                            if (level > 0) { sky[y, rz, rx] = level; skyQ.Add((y, rz, rx)); }
                        }
                        if (t >= 0 && Glow(t) > 0) { block[y, rz, rx] = Glow(t); blockQ.Add((y, rz, rx)); }
                    }
                }
            foreach (var e in emitters ?? System.Array.Empty<LightEmitter>())
            {
                int rx = e.X - columnX * 16 + M, rz = e.Z - columnZ * 16 + M, y = e.Y;
                if (rx < 0 || rx >= R || rz < 0 || rz >= R || y < 0 || y >= H || e.Level <= block[y, rz, rx]) continue;
                block[y, rz, rx] = e.Level; blockQ.Add((y, rz, rx));
            }
            int[] dy = { 1, -1, 0, 0, 0, 0 }, dz = { 0, 0, 1, -1, 0, 0 }, dx = { 0, 0, 0, 0, 1, -1 };
            void Flood(int[,,] level, List<(int y, int z, int x)> queue, bool isSky)
            {
                // by level, highest first, so the first time a cell is reached is by its best light
                for (int l = 15; l >= 1; l--)
                {
                    for (int q = 0; q < queue.Count; q++)
                    {
                        var (y, z, x) = queue[q];
                        if (level[y, z, x] != l) continue;
                        for (int d = 0; d < 6; d++)
                        {
                            int ny = y + dy[d], nz = z + dz[d], nx = x + dx[d];
                            if (ny < 0 || ny >= H || nz < 0 || nz >= R || nx < 0 || nx >= R || !Open(ny, nz, nx)) continue;
                            int dim = Dims(type[ny, nz, nx]);
                            int give = isSky && d == 1 && l == 15 ? 15 - dim : l - Math.Max(1, dim);   // d == 1: going down
                            if (give > level[ny, nz, nx]) { level[ny, nz, nx] = give; queue.Add((ny, nz, nx)); }
                        }
                    }
                }
            }
            Flood(sky, skyQ, true);
            Flood(block, blockQ, false);

            var bytes = new byte[LpHeight * 256];
            for (int y = 0; y < H; y++)
                for (int z = 0; z < 16; z++)
                    for (int x = 0; x < 16; x++)
                        bytes[x + z * 16 + y * 256] = (byte)((sky[y, z + M, x + M] << 4) | block[y, z + M, x + M]);
            return bytes;
        }

        private static async Task<byte[]> LpReadBack(GPUDevice device, GPUQueue queue, GPUBuffer source, ulong offset, int size)
        {
            using var readBuffer = device.CreateBuffer(new GPUBufferDescriptor { Size = (ulong)size, Usage = GPUBufferUsage.MapRead | GPUBufferUsage.CopyDst });
            using var encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(source, offset, readBuffer, 0, (ulong)size);
            using var cmd = encoder.Finish();
            queue.Submit(new[] { cmd });
            await readBuffer.MapAsync(GPUMapMode.Read);
            using var mapped = readBuffer.GetMappedRange();
            using var view = new Uint8Array(mapped);
            var bytes = (byte[])view;
            readBuffer.Unmap();
            return bytes;
        }

        [TestMethod]
        public async Task LightPropagator_LightsAColumnLikeAFloodFill_AndWritesItIntoTheVolumeSlotTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var centre = LpColumn(1);
            var sides = new[] { LpColumn(2), LpColumn(3), null, LpColumn(5) };   // -Z not loaded
            LpTable(out var passes, out var emission, out var dims);

            using var propagator = new LightPropagator(accelerator, LpSections);
            propagator.SetBlockLight(passes, emission, dims);
            using var volume = new LightVolume();
            volume.Init(device, queue, 4, 3, LpHeight, 0);
            volume.SetWindow(0, 0);
            int slot = volume.Assign(0, 0);
            using var cb = accelerator.Allocate1D(centre);
            using var b0 = accelerator.Allocate1D(sides[0]!); using var b1 = accelerator.Allocate1D(sides[1]!); using var b3 = accelerator.Allocate1D(sides[3]!);

            var borders = await propagator.PropagateColumnAsync(volume, slot, cb, b0, b1, null, b3);
            if (borders != LightBorders.All) throw new Exception($"the first light of a slot reports every side: {borders}");

            var got = await LpReadBack(device, queue, volume.Data!, (ulong)slot * (ulong)volume.BytesPerColumn, volume.BytesPerColumn);
            var want = LpReference(centre, sides);
            int wrong = 0, first = -1;
            for (int i = 0; i < want.Length; i++) if (got[i] != want[i]) { wrong++; if (first < 0) first = i; }
            if (wrong != 0)
                throw new Exception($"{wrong} of {want.Length} bytes differ from the flood fill; first at x={first & 15} z={(first >> 4) & 15} y={first >> 8}: got {got[first]:X2}, expected {want[first]:X2}");
            // The test world has light in it: a torch (block 14 at its cell), open sky above the terrain, dark under the roof
            int torch = 6 + 7 * 16 + 11 * 256;
            if ((got[torch] & 15) != 14) throw new Exception($"the torch gives 14, the cell has {got[torch] & 15}");
            if ((got[torch + 1] & 15) != 13) throw new Exception($"one cell from the torch is 13, not {got[torch + 1] & 15}");
            if ((got[15 + 15 * 16 + (LpHeight - 1) * 256] >> 4) != 15) throw new Exception("the top of the world is open sky");
        });

        [TestMethod]
        public async Task LightPropagator_EmittersAndDimmingBlocks_MatchTheFloodFillTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var centre = LpColumn(1);
            var sides = new[] { LpColumn(2), LpColumn(3), LpColumn(4), LpColumn(5) };
            LpTable(out var passes, out var emission, out var dims);
            using var propagator = new LightPropagator(accelerator, LpSections);
            propagator.SetBlockLight(passes, emission, dims);
            using var volume = new LightVolume();
            volume.Init(device, queue, 4, 3, LpHeight, 0);
            volume.SetWindow(5, -2);
            int slot = volume.Assign(5, -2);   // the column is (5, -2): the emitters are in world blocks
            using var cb = accelerator.Allocate1D(centre);
            using var b0 = accelerator.Allocate1D(sides[0]!); using var b1 = accelerator.Allocate1D(sides[1]!);
            using var b2 = accelerator.Allocate1D(sides[2]!); using var b3 = accelerator.Allocate1D(sides[3]!);

            // A campfire in the air above the terrain (14), a torch next to the column in the -X neighbour (12), two in one cell (the brighter counts), one far away, one out of the world and one dark
            var emitters = new[]
            {
                new LightEmitter(5 * 16 + 4, 30, -2 * 16 + 4, 14),
                new LightEmitter(5 * 16 - 3, 25, -2 * 16 + 9, 12),
                new LightEmitter(5 * 16 + 4, 30, -2 * 16 + 4, 9),
                new LightEmitter(5 * 16 + 100, 30, -2 * 16 + 4, 15),
                new LightEmitter(5 * 16 + 4, 9999, -2 * 16 + 4, 15),
                new LightEmitter(5 * 16 + 4, 31, -2 * 16 + 4, 0),
            };
            await propagator.PropagateColumnAsync(volume, slot, cb, b0, b1, b2, b3, 5, -2, emitters);
            var got = await LpReadBack(device, queue, volume.Data!, (ulong)slot * (ulong)volume.BytesPerColumn, volume.BytesPerColumn);
            var want = LpReference(centre, sides, emitters, 5, -2);
            int wrong = 0, first = -1;
            for (int i = 0; i < want.Length; i++) if (got[i] != want[i]) { wrong++; if (first < 0) first = i; }
            if (wrong != 0) throw new Exception($"{wrong} of {want.Length} bytes differ from the flood fill; first at x={first & 15} z={(first >> 4) & 15} y={first >> 8}: got {got[first]:X2}, expected {want[first]:X2}");
            int campfire = 4 + 4 * 16 + 30 * 256;
            if (centre[campfire] == LpAir && (got[campfire] & 15) < 14) throw new Exception($"the campfire gives 14, the cell has {got[campfire] & 15}");
            // The leaves cut the sky light: the top of the canopy is one level down from the open sky, the cell under two layers of leaves two or more
            int leafTop = 11 + 11 * 16 + 39 * 256, openAbove = 11 + 11 * 16 + 40 * 256;
            if ((got[openAbove] >> 4) != 15) throw new Exception($"above the canopy is open sky, not {got[openAbove] >> 4}");
            if ((got[leafTop] >> 4) != 14) throw new Exception($"the leaf under open sky has 14, not {got[leafTop] >> 4}");
            // Without the emitters the cell is darker
            await propagator.PropagateColumnAsync(volume, slot, cb, b0, b1, b2, b3, 5, -2);
            var plain = await LpReadBack(device, queue, volume.Data!, (ulong)slot * (ulong)volume.BytesPerColumn, volume.BytesPerColumn);
            if ((plain[campfire] & 15) >= 14) throw new Exception("without the emitter the cell is not a campfire");
        });

        [TestMethod]
        public async Task LightPropagator_AnEditNearASide_SaysWhichNeighboursToLightAgainTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var centre = LpColumn(1);
            LpTable(out var passes, out var emission, out var dims);
            using var propagator = new LightPropagator(accelerator, LpSections);
            propagator.SetBlockLight(passes, emission, dims);
            using var volume = new LightVolume();
            volume.Init(device, queue, 2, 3, LpHeight, 0);
            volume.SetWindow(0, 0);
            int slot = volume.Assign(0, 0);
            using var cb = accelerator.Allocate1D(centre);

            await propagator.PropagateColumnAsync(volume, slot, cb);
            var again = await propagator.PropagateColumnAsync(volume, slot, cb);
            if (again != LightBorders.None) throw new Exception($"the same column again changes nothing: {again}");

            // A torch beside the +X edge: the +X side reports it (the strips are 15 deep from each side, so in a column 16 wide they overlap: the other side may report it too, which only costs a neighbour lit for nothing)
            var edited = (ushort[])centre.Clone();
            edited[14 + 8 * 16 + 30 * 256] = LpTorch;
            cb.CopyFromCPU(edited);
            var changed = await propagator.PropagateColumnAsync(volume, slot, cb);
            if (!changed.HasFlag(LightBorders.PosX)) throw new Exception($"a torch next to the +X side must report it: {changed}");

            // Forgetting a slot: the next light reports everything
            propagator.Forget(volume, slot);
            if (await propagator.PropagateColumnAsync(volume, slot, cb) != LightBorders.All) throw new Exception("a forgotten slot is new again");
        });
    }
}
