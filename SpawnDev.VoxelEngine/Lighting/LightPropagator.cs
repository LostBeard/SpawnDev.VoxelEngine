using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Lighting
{
    /// <summary>The sides of a column whose neighbour has to be lit again because the column changed near that side.</summary>
    [Flags]
    public enum LightBorders
    {
        None = 0,
        NegX = 1,
        PosX = 2,
        NegZ = 4,
        PosZ = 8,
        All = 15,
    }

    /// <summary>A light that is not a block (a campfire, a torch an entity carries): a place in world blocks and a level 0..15 (campfire 14, torch 12).</summary>
    public readonly record struct LightEmitter(int X, int Y, int Z, int Level);

    /// <summary>
    /// Light for a world that has none of its own to give: works out the sky light and block light of a column on the GPU from its blocks and its four neighbours' (the same column buffers the
    /// <see cref="Meshing.ColumnMesher"/> takes) and writes it into a slot of a <see cref="LightVolume"/>, ready for the block shader. Nothing comes back to the CPU but a few hundred integers.
    ///
    /// What the blocks do to light comes from a table (<see cref="SetBlockLight"/>): per block type whether light passes through it and how much light it gives off (0..15). Sky light enters at the top
    /// of every column and goes straight down without losing anything; every other step, sideways or up or down, costs one level, as in the game. A neighbour that is not loaded (null) is open sky with
    /// no block light, which is also how the <see cref="LightVolume"/> treats a column it has no slot for, so a column at the edge of the loaded world is never darker than it will be. The corner columns
    /// (diagonal neighbours) are not taken: light that turns a corner between two columns is missed.
    ///
    /// Light that comes from something that is not a block is given as <see cref="LightEmitter"/>s with the call (a list of the emitters near the column: those within reach of its 3 x 3 columns count, the rest are ignored).
/// A block can dim what passes through it (<see cref="SetBlockLight"/>'s <c>dims</c>: leaves 1): every step costs a level, a dimming block costs its dimming instead if that is more, and sky light falling straight
/// down loses exactly the dimming (so the ground under a canopy is darker than in the open, as in the game).
///
/// An edit lights one column again: call <see cref="PropagateColumnAsync"/> for the column that changed. What it returns says which neighbours the change reaches (their cells within light's reach of
    /// the shared side changed, in blocks or light), and those columns are propagated in turn; a column whose sides did not change ends the chain.
    /// </summary>
    public sealed class LightPropagator : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly int _sections, _height;
        private readonly Action<Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, int, int> _init;
        private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _relax;
        private readonly Action<Index1D, ArrayView<int>, ArrayView<uint>, int> _pack;
        private readonly Action<Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, int> _signature;
        private readonly Action<Index1D, ArrayView<int>, ArrayView<int>> _emit;
        /// <summary>Most emitters one call takes (more are dropped: the brightest first).</summary>
        public const int MaxEmitters = 1024;
        private readonly MemoryBuffer1D<int, Stride1D.Dense> _table, _cellsA, _cellsB, _signatures, _emitters;
        private readonly MemoryBuffer1D<uint, Stride1D.Dense> _output;
        private readonly MemoryBuffer1D<ushort, Stride1D.Dense>[] _dummyColumns;
        private readonly Dictionary<(LightVolume, int), int[]> _signaturesOf = new();

        /// <param name="sections">Sections of 16 blocks in a column (24 for Minecraft's 384, 16 for a world 256 high): as the <see cref="Meshing.ColumnMesher"/>'s.</param>
        public LightPropagator(Accelerator accelerator, int sections = 24)
        {
            if (sections < 1 || sections > 24) throw new ArgumentOutOfRangeException(nameof(sections), "A column is 1 to 24 sections");
            _accelerator = accelerator;
            _sections = sections;
            _height = sections * 16;
            _init = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, int, int>(LightKernels.InitKernel);
            _relax = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(LightKernels.RelaxKernel);
            _emit = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(LightKernels.EmitKernel);
            _emitters = accelerator.Allocate1D<int>(2 * MaxEmitters);
            _pack = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<uint>, int>(LightKernels.PackKernel);
            _signature = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, int>(LightKernels.SignatureKernel);
            _table = accelerator.Allocate1D<int>(4096);
            long cells = (long)LightKernels.Region * LightKernels.Region * _height;
            _cellsA = accelerator.Allocate1D<int>(cells);
            _cellsB = accelerator.Allocate1D<int>(cells);
            _output = accelerator.Allocate1D<uint>(_height * 64);
            _signatures = accelerator.Allocate1D<int>(4 * _height);
            _dummyColumns = new[] { accelerator.Allocate1D<ushort>(4), accelerator.Allocate1D<ushort>(4), accelerator.Allocate1D<ushort>(4), accelerator.Allocate1D<ushort>(4) };
            // Until a table is given: light passes through nothing but air (type 0) and nothing glows
            SetBlockLight(new byte[] { 1 }, ReadOnlySpan<byte>.Empty);
        }

        /// <summary>Bytes of one column's light: what <see cref="LightVolume.BytesPerColumn"/> must be.</summary>
        public int BytesPerColumn => _height * 256;

        /// <summary>The light of the column propagated last, as the volume gets it (a byte a block, sky light high, block light low; four to a word): for tests and for saving light.</summary>
        public MemoryBuffer1D<uint, Stride1D.Dense> LastOutput => _output;

        /// <summary>
        /// What each block type does to light (index = the block type, up to 4096): <paramref name="passesLight"/> non-zero where light passes through the block (air, glass, leaves, a torch),
        /// <paramref name="emission"/> the light the block gives off (0..15: a torch 14, lava 15; a type past the end gives none), <paramref name="dims"/> (optional) the levels the block takes off what
        /// passes through it (0..15: leaves 1, murky water 2; a type past the end dims nothing). A type past the end of <paramref name="passesLight"/> blocks light.
        /// </summary>
        public void SetBlockLight(ReadOnlySpan<byte> passesLight, ReadOnlySpan<byte> emission, ReadOnlySpan<byte> dims = default)
        {
            if (passesLight.Length > 4096 || emission.Length > 4096 || dims.Length > 4096) throw new ArgumentOutOfRangeException(nameof(passesLight), "At most 4096 block types");
            var table = new int[4096];
            for (int i = 0; i < table.Length; i++)
            {
                int passes = i < passesLight.Length && passesLight[i] != 0 ? LightKernels.TablePasses : 0;
                int glow = i < emission.Length ? Math.Min((int)emission[i], 15) : 0;
                int dim = i < dims.Length ? Math.Min((int)dims[i], 15) : 0;
                table[i] = passes | (glow << 4) | (dim << LightKernels.TableDimsShift);
            }
            _table.CopyFromCPU(table);
        }

        /// <summary>
        /// Light one column into <paramref name="slot"/> of <paramref name="volume"/> (give it the slot with <see cref="LightVolume.Assign"/>; its column height must be this propagator's). The
        /// neighbours (-X, +X, -Z, +Z) are the adjacent columns, or null when not loaded. Returns the sides whose neighbours must be lit again; the first time a slot is lit that is all four.
        /// <paramref name="emitters"/> (optional) are the lights that are not blocks, in world blocks (the volume's <see cref="LightVolume.MinY"/> is layer 0); <paramref name="columnX"/> and
        /// <paramref name="columnZ"/> are this column's place in columns (world block x = columnX * 16 + the block's x), needed only to place the emitters.
        /// </summary>
        public async Task<LightBorders> PropagateColumnAsync(
            LightVolume volume, int slot,
            MemoryBuffer1D<ushort, Stride1D.Dense> column,
            MemoryBuffer1D<ushort, Stride1D.Dense>? negX = null,
            MemoryBuffer1D<ushort, Stride1D.Dense>? posX = null,
            MemoryBuffer1D<ushort, Stride1D.Dense>? negZ = null,
            MemoryBuffer1D<ushort, Stride1D.Dense>? posZ = null,
            int columnX = 0, int columnZ = 0,
            IReadOnlyList<LightEmitter>? emitters = null)
        {
            int length = _height * 256;
            if (column.Length != length) throw new ArgumentException($"A column of {_sections} sections is {length} blocks, got {column.Length}", nameof(column));
            foreach (var n in new[] { negX, posX, negZ, posZ })
                if (n != null && n.Length != length) throw new ArgumentException($"A neighbouring column must also be {length} blocks, got {n.Length}");
            if (volume.BytesPerColumn != BytesPerColumn)
                throw new ArgumentException($"The volume's columns are {volume.BytesPerColumn} bytes, this propagator's are {BytesPerColumn}: Init the volume with columnHeight {_height}", nameof(volume));

            int present = (negX != null ? 1 : 0) | (posX != null ? 2 : 0) | (negZ != null ? 4 : 0) | (posZ != null ? 8 : 0);
            _init((Index1D)(LightKernels.Region * LightKernels.Region), column.View,
                (negX ?? _dummyColumns[0]).View, (posX ?? _dummyColumns[1]).View, (negZ ?? _dummyColumns[2]).View, (posZ ?? _dummyColumns[3]).View,
                _table.View, _cellsA.View, _height, present);

            int emitterCount = UploadEmitters(emitters, columnX, columnZ, volume.MinY);
            if (emitterCount > 0) _emit((Index1D)emitterCount, _emitters.View, _cellsA.View);

            // Fifteen steps spread the light as far as it goes; each step reads one buffer and writes the other
            var from = _cellsA;
            var to = _cellsB;
            int cellCount = LightKernels.Region * LightKernels.Region * _height;
            for (int step = 0; step < LightKernels.Reach; step++)
            {
                _relax((Index1D)cellCount, from.View, to.View, _height);
                (from, to) = (to, from);
            }
            _pack((Index1D)(_height * 64), from.View, _output.View, _height);
            _signature((Index1D)(4 * _height), column.View, from.View, _signatures.View, _height);

            // The light goes into the volume's slot on the GPU (the kernels are submitted first: the queue keeps their order)
            _accelerator.Flush();
            volume.CopyIn(slot, _output.GetGPUBuffer()!);

            var now = new int[4];
            var raw = new int[4 * _height];
            await _signatures.CopyToHostAsync(raw);
            for (int side = 0; side < 4; side++)
            {
                int h = 17;
                for (int y = 0; y < _height; y++) h = (h * 16777619) ^ raw[side * _height + y];
                now[side] = h;
            }
            var key = (volume, slot);
            LightBorders changed;
            if (!_signaturesOf.TryGetValue(key, out var before)) changed = LightBorders.All;
            else
            {
                changed = LightBorders.None;
                for (int side = 0; side < 4; side++)
                    if (before[side] != now[side]) changed |= (LightBorders)(1 << side);
            }
            _signaturesOf[key] = now;
            return changed;
        }

        // The emitters inside the region as (cell, level) pairs, one for each cell (the brightest of two in one), the brightest MaxEmitters of them
        private int UploadEmitters(IReadOnlyList<LightEmitter>? emitters, int columnX, int columnZ, int minY)
        {
            if (emitters == null || emitters.Count == 0) return 0;
            var byCell = new Dictionary<int, int>();
            int R = LightKernels.Region;
            foreach (var e in emitters)
            {
                int level = Math.Clamp(e.Level, 0, 15);
                if (level == 0) continue;
                int rx = e.X - columnX * 16 + LightKernels.Margin, rz = e.Z - columnZ * 16 + LightKernels.Margin, y = e.Y - minY;
                if (rx < 0 || rx >= R || rz < 0 || rz >= R || y < 0 || y >= _height) continue;
                int cell = (y * R + rz) * R + rx;
                if (!byCell.TryGetValue(cell, out int have) || level > have) byCell[cell] = level;
            }
            if (byCell.Count == 0) return 0;
            var list = byCell.OrderByDescending(kv => kv.Value).Take(MaxEmitters).ToList();
            var data = new int[2 * list.Count];
            for (int i = 0; i < list.Count; i++) { data[2 * i] = list[i].Key; data[2 * i + 1] = list[i].Value; }
            _emitters.View.SubView(0, data.Length).CopyFromCPU(data);
            return list.Count;
        }

        /// <summary>Forget what a slot's column looked like (its column is gone, or the slot is given to another): the next light of the slot reports every side changed.</summary>
        public void Forget(LightVolume volume, int slot) => _signaturesOf.Remove((volume, slot));

        public void Dispose()
        {
            _table.Dispose(); _cellsA.Dispose(); _cellsB.Dispose(); _output.Dispose(); _signatures.Dispose(); _emitters.Dispose();
            foreach (var d in _dummyColumns) d.Dispose();
        }
    }
}
