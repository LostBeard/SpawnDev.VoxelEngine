using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// The result of meshing one block column: three lists of <see cref="PackedQuad"/>s - the opaque pass, the plants, the water pass -
    /// each a GPU-resident buffer in section order, with the start offset of every section in it. Nothing was read back except the
    /// 75 counters that size the buffers. The caller owns the buffers and disposes them (or copies them into a pool and disposes).
    /// </summary>
    public sealed class ColumnMesh : IDisposable
    {
        /// <summary>Cube faces of the opaque and cutout blocks (faces 0-5).</summary>
        public QuadList Opaque { get; }
        /// <summary>Crossed-plane plants (quad faces 6 and 7).</summary>
        public QuadList Plants { get; }
        /// <summary>Water faces, for the alpha-blended pass.</summary>
        public QuadList Water { get; }
        /// <summary>Quads that did not fit the scratch space of a section and were dropped. Always 0 (the scratch is sized to a proven bound); a non-zero value is a bug.</summary>
        public int Overflow { get; }

        public ColumnMesh(QuadList opaque, QuadList plants, QuadList water, int overflow)
        {
            Opaque = opaque; Plants = plants; Water = water; Overflow = overflow;
        }

        public bool IsEmpty => Opaque.Count == 0 && Plants.Count == 0 && Water.Count == 0;

        public void Dispose()
        {
            Opaque.Dispose();
            Plants.Dispose();
            Water.Dispose();
        }

        /// <summary>One pass's quads: <see cref="Buffer"/> holds <see cref="Count"/> quads, section s's at [SectionStart[s], SectionStart[s + 1]).</summary>
        public sealed class QuadList : IDisposable
        {
            /// <summary>Null when the list is empty.</summary>
            public MemoryBuffer1D<long, Stride1D.Dense>? Buffer { get; }
            public int Count { get; }
            /// <summary>sectionCount + 1 entries: the quad index where each section starts, then the total.</summary>
            public int[] SectionStart { get; }

            public QuadList(MemoryBuffer1D<long, Stride1D.Dense>? buffer, int count, int[] sectionStart)
            {
                Buffer = buffer; Count = count; SectionStart = sectionStart;
            }

            public int SectionCount => SectionStart.Length - 1;
            public int QuadsIn(int section) => SectionStart[section + 1] - SectionStart[section];

            public void Dispose() => Buffer?.Dispose();
        }
    }

    /// <summary>
    /// Meshes block columns (16 x 16 cells, <see cref="SectionCount"/> sections of 16 high) into <see cref="PackedQuad"/> lists entirely
    /// on the GPU: a handful of dispatches per column instead of one pipeline run per section (see <see cref="ColumnMeshKernels"/>).
    ///
    /// The column arrives as a GPU buffer of ushort block types (x + z * 16 + y * 256), filled by the caller: from a JS typed array with
    /// <c>((IBrowserMemoryBuffer)buffer).CopyFromJS(...)</c> - the bulk never enters the .NET heap - or from an array in tests. The
    /// four neighbouring columns are optional; a missing one counts as solid by default (no faces toward an unloaded column).
    ///
    /// What each block type is - opaque, see-through, water, plant, air - comes from the table set with <see cref="SetBlockClasses"/>
    /// (<see cref="BlockClass"/> flags per 12-bit type).
    ///
    /// Not thread-safe: it owns shared scratch buffers, so mesh one column at a time (serialise callers).
    /// </summary>
    public sealed class ColumnMesher : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly int _sections;
        private readonly int _columnLength;

        private readonly Action<Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>,
            ArrayView<int>, ArrayView<long>, ArrayView<long>, int, int, int, int> _occupancy;
        private readonly Action<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>, int> _faceCull;
        private readonly Action<Index1D, ArrayView<long>, ArrayView<ushort>, ArrayView<long>, ArrayView<int>, int, int> _merge;
        private readonly Action<Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<long>, ArrayView<int>, int> _plants;
        private readonly Action<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<long>, int> _compact;

        private readonly MemoryBuffer1D<int, Stride1D.Dense> _typeFlags;
        private readonly MemoryBuffer1D<long, Stride1D.Dense> _emit, _occlude, _faceMasks;
        private readonly MemoryBuffer1D<long, Stride1D.Dense> _scratchOpaque, _scratchPlants, _scratchWater;
        private readonly MemoryBuffer1D<int, Stride1D.Dense> _counters;       // 3 passes x (sections + 1)
        private readonly MemoryBuffer1D<int, Stride1D.Dense>[] _sectionStart; // one per pass
        private readonly MemoryBuffer1D<ushort, Stride1D.Dense>[] _dummyColumns; // one per neighbour: a bound view must never alias another binding

        /// <summary>Sections per column (24 for Minecraft's 384-block world; any column height that is a multiple of 16).</summary>
        public int SectionCount => _sections;
        public int ColumnLength => _columnLength;

        /// <summary>Whether a missing neighbour column hides faces toward it (true) or leaves them visible (false).</summary>
        public bool MissingNeighborsAreSolid { get; set; } = true;

        public ColumnMesher(Accelerator accelerator, int sectionCount = 24)
        {
            if (sectionCount < 1) throw new ArgumentOutOfRangeException(nameof(sectionCount));
            _accelerator = accelerator;
            _sections = sectionCount;
            _columnLength = ColumnMeshKernels.SectionSize * ColumnMeshKernels.SectionSize * ColumnMeshKernels.SectionSize * sectionCount;

            _occupancy = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>,
                ArrayView<ushort>, ArrayView<ushort>, ArrayView<int>, ArrayView<long>, ArrayView<long>, int, int, int, int>(
                ColumnMeshKernels.OccupancyKernel);
            _faceCull = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>, int>(
                ColumnMeshKernels.FaceCullKernel);
            _merge = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<ushort>, ArrayView<long>, ArrayView<int>, int, int>(
                ColumnMeshKernels.GreedyMergeKernel);
            _plants = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<long>, ArrayView<int>, int>(
                ColumnMeshKernels.PlantKernel);
            _compact = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<long>, int>(
                ColumnMeshKernels.CompactKernel);

            _typeFlags = accelerator.Allocate1D<int>(PackedBlock.MaxBlockTypes);
            _emit = accelerator.Allocate1D<long>(sectionCount * ColumnMeshKernels.PaddedCellsPerSection);
            _occlude = accelerator.Allocate1D<long>(sectionCount * ColumnMeshKernels.PaddedCellsPerSection);
            _faceMasks = accelerator.Allocate1D<long>(sectionCount * 6 * ColumnMeshKernels.ColumnsCellsPerLayer);
            long scratch = (long)sectionCount * ColumnMeshKernels.ScratchQuadsPerSection;
            _scratchOpaque = accelerator.Allocate1D<long>(scratch);
            _scratchPlants = accelerator.Allocate1D<long>(scratch);
            _scratchWater = accelerator.Allocate1D<long>(scratch);
            _counters = accelerator.Allocate1D<int>(3 * (sectionCount + 1));
            _sectionStart = new[]
            {
                accelerator.Allocate1D<int>(sectionCount), accelerator.Allocate1D<int>(sectionCount), accelerator.Allocate1D<int>(sectionCount),
            };
            _dummyColumns = new[]
            {
                accelerator.Allocate1D<ushort>(4), accelerator.Allocate1D<ushort>(4), accelerator.Allocate1D<ushort>(4), accelerator.Allocate1D<ushort>(4),
            };
        }

        /// <summary>
        /// A column buffer filled straight from JavaScript: <paramref name="blockIds"/> holds the column's ushort block types as raw
        /// little-endian bytes (2 per block, x + z * 16 + y * 256), for example a <c>Uint8Array</c> view of a fetched ArrayBuffer. The bytes go
        /// from the JS typed array to the GPU buffer (<c>queue.writeBuffer</c>) and never enter the .NET heap. Browser backends only.
        /// </summary>
        public MemoryBuffer1D<ushort, Stride1D.Dense> CreateColumnBufferFromJS(Uint8Array blockIds)
        {
            if (blockIds.Length != _columnLength * 2)
                throw new ArgumentException($"A column of {_sections} sections is {_columnLength * 2} bytes of block ids, got {blockIds.Length}", nameof(blockIds));
            var buffer = _accelerator.Allocate1D<ushort>(_columnLength);
            if (buffer.Buffer is not IBrowserMemoryBuffer browser)
            {
                buffer.Dispose();
                throw new NotSupportedException($"{_accelerator.GetType().Name} cannot take a JavaScript typed array: CopyFromJS exists on the browser backends only");
            }
            browser.CopyFromJS(blockIds);
            return buffer;
        }

        /// <summary>
        /// The class of each block type (<see cref="BlockClass"/> flags, index = the 12-bit type; up to 4096 entries). Types past the end of
        /// <paramref name="flags"/> are <see cref="BlockClass.None"/> (never drawn).
        /// </summary>
        public void SetBlockClasses(ReadOnlySpan<int> flags)
        {
            if (flags.Length > PackedBlock.MaxBlockTypes)
                throw new ArgumentOutOfRangeException(nameof(flags), $"{flags.Length} entries; at most {PackedBlock.MaxBlockTypes} block types");
            var table = new int[PackedBlock.MaxBlockTypes];
            flags.CopyTo(table);
            _typeFlags.CopyFromCPU(table);
        }

        /// <summary>
        /// Mesh one column. <paramref name="column"/> is ushort block types, x + z * 16 + y * 256 for y in [0, 16 * SectionCount); the
        /// neighbours (-X, +X, -Z, +Z) are the adjacent columns in the same layout, or null when not loaded.
        /// </summary>
        public async Task<ColumnMesh> MeshAsync(
            MemoryBuffer1D<ushort, Stride1D.Dense> column,
            MemoryBuffer1D<ushort, Stride1D.Dense>? negX = null,
            MemoryBuffer1D<ushort, Stride1D.Dense>? posX = null,
            MemoryBuffer1D<ushort, Stride1D.Dense>? negZ = null,
            MemoryBuffer1D<ushort, Stride1D.Dense>? posZ = null)
        {
            if (column.Length != _columnLength)
                throw new ArgumentException($"A column of {_sections} sections is {_columnLength} blocks, got {column.Length}", nameof(column));
            foreach (var n in new[] { negX, posX, negZ, posZ })
                if (n != null && n.Length != _columnLength)
                    throw new ArgumentException($"A neighbouring column must also be {_columnLength} blocks, got {n.Length}");

            int missing = (negX == null ? 1 : 0) | (posX == null ? 2 : 0) | (negZ == null ? 4 : 0) | (posZ == null ? 8 : 0);
            int solid = MissingNeighborsAreSolid ? 1 : 0;
            var vNegX = (negX ?? _dummyColumns[0]).View;
            var vPosX = (posX ?? _dummyColumns[1]).View;
            var vNegZ = (negZ ?? _dummyColumns[2]).View;
            var vPosZ = (posZ ?? _dummyColumns[3]).View;

            int stride = _sections + 1;
            _counters.MemSetToZero();

            // Opaque pass, then water pass: same kernels, different bits of the class table
            RunCubePass(column.View, vNegX, vPosX, vNegZ, vPosZ, BlockClass.OpaqueShift, missing, solid, _scratchOpaque, _counters.View.SubView(0, stride));
            RunCubePass(column.View, vNegX, vPosX, vNegZ, vPosZ, BlockClass.WaterShift, missing, solid, _scratchWater, _counters.View.SubView(2 * stride, stride));

            // Plants: two quads per plant block
            _plants((Index1D)(_sections * ColumnMeshKernels.CellsPerSection),
                column.View, _typeFlags.View, _scratchPlants.View, _counters.View.SubView(stride, stride), _sections);

            // The only readback: the 3 x (sections + 1) counters that size the output buffers
            var counters = await _counters.CopyToHostAsync();

            var opaque = Compact(counters, 0, _scratchOpaque, _sectionStart[0]);
            var plants = Compact(counters, stride, _scratchPlants, _sectionStart[1]);
            var water = Compact(counters, 2 * stride, _scratchWater, _sectionStart[2]);
            int overflow = counters[_sections] + counters[stride + _sections] + counters[2 * stride + _sections];

            // Submit now: whoever draws these buffers submits to the same queue after us
            _accelerator.Flush();
            return new ColumnMesh(opaque, plants, water, overflow);
        }

        private void RunCubePass(
            ArrayView<ushort> column, ArrayView<ushort> negX, ArrayView<ushort> posX, ArrayView<ushort> negZ, ArrayView<ushort> posZ,
            int passShift, int missing, int solid, MemoryBuffer1D<long, Stride1D.Dense> scratch, ArrayView<int> counters)
        {
            _occupancy((Index1D)(_sections * ColumnMeshKernels.PaddedCellsPerSection),
                column, negX, posX, negZ, posZ, _typeFlags.View, _emit.View, _occlude.View, passShift, missing, solid, _sections);
            _faceCull((Index1D)(_sections * ColumnMeshKernels.ColumnsCellsPerLayer), _emit.View, _occlude.View, _faceMasks.View, _sections);
            // +X -X +Z -Z, then +Y -Y: separate launches because the Y faces of different layers share mask words
            _merge((Index1D)(_sections * 4 * ColumnMeshKernels.SectionSize), _faceMasks.View, column, scratch.View, counters, 0, _sections);
            _merge((Index1D)(_sections * 2 * ColumnMeshKernels.SectionSize), _faceMasks.View, column, scratch.View, counters, 4, _sections);
        }

        private ColumnMesh.QuadList Compact(int[] counters, int offset, MemoryBuffer1D<long, Stride1D.Dense> scratch, MemoryBuffer1D<int, Stride1D.Dense> sectionStartBuffer)
        {
            var start = new int[_sections + 1];
            for (int s = 0; s < _sections; s++)
                start[s + 1] = start[s] + counters[offset + s];
            int total = start[_sections];
            if (total == 0)
                return new ColumnMesh.QuadList(null, 0, start);

            var output = _accelerator.Allocate1D<long>(total);
            sectionStartBuffer.CopyFromCPU(start.AsSpan(0, _sections).ToArray());
            _compact((Index1D)(_sections * ColumnMeshKernels.ScratchQuadsPerSection),
                scratch.View, _counters.View.SubView(offset, _sections), sectionStartBuffer.View, output.View, _sections);
            return new ColumnMesh.QuadList(output, total, start);
        }

        public void Dispose()
        {
            _typeFlags.Dispose();
            _emit.Dispose(); _occlude.Dispose(); _faceMasks.Dispose();
            _scratchOpaque.Dispose(); _scratchPlants.Dispose(); _scratchWater.Dispose();
            _counters.Dispose();
            foreach (var b in _sectionStart) b.Dispose();
            foreach (var d in _dummyColumns) d.Dispose();
        }
    }
}
