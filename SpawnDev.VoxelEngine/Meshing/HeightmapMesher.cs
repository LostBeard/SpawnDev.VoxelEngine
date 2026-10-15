using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Meshes a chunk's heightmap (<see cref="HeightmapLayout"/>) into <see cref="PackedQuad"/> lists on the GPU: the far tier of a world
    /// viewer, where a column of blocks is replaced by its surface. The result is a <see cref="ColumnMesh"/> (opaque and water lists,
    /// no plants) so it is drawn exactly like a meshed column, by section origin.
    ///
    /// The heightmap arrives as a GPU buffer of <see cref="HeightmapLayout.Ints"/> ints, filled by the caller (from a JS typed array with
    /// <see cref="CreateHeightmapBufferFromJS"/>: the bytes never enter the .NET heap). The four neighbouring chunks' heightmaps are
    /// optional: they decide where walls are needed along the shared border.
    ///
    /// Types come from the same per-type class table as <see cref="ColumnMesher"/> (<see cref="SetBlockClasses"/>).
    /// <c>seabedTypeOffset</c> is added to the type of anything seen through water, so a viewer can give those types darker materials.
    ///
    /// Not thread-safe: shared scratch buffers; mesh one chunk at a time.
    /// </summary>
    public sealed class HeightmapMesher : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly int _sections;

        private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<long>, ArrayView<int>, int, int, int> _mesh;
        private readonly Action<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<long>, int, int> _compact;

        private readonly MemoryBuffer1D<int, Stride1D.Dense> _typeFlags;
        private readonly MemoryBuffer1D<int, Stride1D.Dense> _hood;
        private readonly MemoryBuffer1D<long, Stride1D.Dense> _scratchOpaque, _scratchWater;
        private readonly MemoryBuffer1D<int, Stride1D.Dense> _counters;       // 2 x (sections + 1)
        private readonly int[] _zeroCounters;
        private readonly MemoryBuffer1D<int, Stride1D.Dense>[] _sectionStart; // opaque, water

        /// <summary>Added to the type of a seabed (what lies under water); 0 keeps the type.</summary>
        public int SeabedTypeOffset { get; set; }

        public int SectionCount => _sections;

        /// <summary>See <see cref="ColumnMesher.EmbedSectionIndex"/>: one draw per chunk instead of one per section.</summary>
        public bool EmbedSectionIndex { get; set; }

        public HeightmapMesher(Accelerator accelerator, int sectionCount = 24)
        {
            if (sectionCount < 1) throw new ArgumentOutOfRangeException(nameof(sectionCount));
            _accelerator = accelerator;
            _sections = sectionCount;

            _mesh = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<long>,
                ArrayView<int>, int, int, int>(HeightmapMeshKernels.HeightmapKernel);
            _compact = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<long>, int, int>(
                HeightmapMeshKernels.CompactKernel);

            _typeFlags = accelerator.Allocate1D<int>(PackedBlock.MaxBlockTypes);
            _hood = accelerator.Allocate1D<int>(5 * HeightmapLayout.Ints);
            long scratch = (long)sectionCount * HeightmapMeshKernels.ScratchQuadsPerSection;
            _scratchOpaque = accelerator.Allocate1D<long>(scratch);
            _scratchWater = accelerator.Allocate1D<long>(scratch);
            _counters = accelerator.Allocate1D<int>(2 * (sectionCount + 1));
            _zeroCounters = new int[2 * (sectionCount + 1)];
            _sectionStart = new[] { accelerator.Allocate1D<int>(sectionCount), accelerator.Allocate1D<int>(sectionCount) };
        }

        /// <summary>
        /// A heightmap buffer filled straight from JavaScript: <paramref name="bytes"/> holds the <see cref="HeightmapLayout"/> as raw
        /// little-endian bytes (<see cref="HeightmapLayout.Bytes"/>), for example a <c>Uint8Array</c> view into a received frame.
        /// Browser backends only.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> CreateHeightmapBufferFromJS(Uint8Array bytes)
        {
            if (bytes.Length != HeightmapLayout.Bytes)
                throw new ArgumentException($"A heightmap is {HeightmapLayout.Bytes} bytes, got {bytes.Length}", nameof(bytes));
            var buffer = _accelerator.Allocate1D<int>(HeightmapLayout.Ints);
            if (buffer.Buffer is not IBrowserMemoryBuffer browser)
            {
                buffer.Dispose();
                throw new NotSupportedException($"{_accelerator.GetType().Name} cannot take a JavaScript typed array: CopyFromJS exists on the browser backends only");
            }
            browser.CopyFromJS(bytes);
            return buffer;
        }

        /// <summary>The class of each block type, as for <see cref="ColumnMesher.SetBlockClasses"/>.</summary>
        public void SetBlockClasses(ReadOnlySpan<int> flags)
        {
            if (flags.Length > PackedBlock.MaxBlockTypes)
                throw new ArgumentOutOfRangeException(nameof(flags), $"{flags.Length} entries; at most {PackedBlock.MaxBlockTypes} block types");
            var table = new int[PackedBlock.MaxBlockTypes];
            flags.CopyTo(table);
            _typeFlags.CopyFromCPU(table);
        }

        /// <summary>Mesh one chunk's heightmap; the neighbours (-X, +X, -Z, +Z) are the adjacent chunks' heightmaps, or null when not loaded.</summary>
        public async Task<ColumnMesh> MeshAsync(
            MemoryBuffer1D<int, Stride1D.Dense> heightmap,
            MemoryBuffer1D<int, Stride1D.Dense>? negX = null,
            MemoryBuffer1D<int, Stride1D.Dense>? posX = null,
            MemoryBuffer1D<int, Stride1D.Dense>? negZ = null,
            MemoryBuffer1D<int, Stride1D.Dense>? posZ = null)
        {
            var maps = new[] { heightmap, negX, posX, negZ, posZ };
            foreach (var m in maps)
                if (m != null && m.Length != HeightmapLayout.Ints)
                    throw new ArgumentException($"A heightmap is {HeightmapLayout.Ints} ints, got {m.Length}");

            // One view of all five, so the kernel stays within WebGPU's storage-buffer limit
            int missing = 0;
            for (int i = 0; i < 5; i++)
            {
                if (maps[i] == null) { if (i > 0) missing |= 1 << (i - 1); continue; }
                _hood.View.SubView(i * HeightmapLayout.Ints, HeightmapLayout.Ints).CopyFrom(maps[i]!.View);
            }

            int stride = _sections + 1;
            // CopyFromCPU, not MemSetToZero: an upload is ordered after the previous run's still-queued kernels on every backend (the Wasm
            // backend queues it behind pending dispatches), a memset is not documented to be - and on Wasm the previous run's compaction
            // could still be reading these counters when they were cleared
            _counters.CopyFromCPU(_zeroCounters);
            _mesh((Index1D)HeightmapLayout.Cells, _hood.View, _typeFlags.View, _scratchOpaque.View, _scratchWater.View, _counters.View,
                missing, SeabedTypeOffset, _sections);

            var counters = await _counters.CopyToHostAsync();
            var opaque = Compact(counters, 0, _scratchOpaque, _sectionStart[0]);
            var water = Compact(counters, stride, _scratchWater, _sectionStart[1]);
            int overflow = counters[_sections] + counters[stride + _sections];

            _accelerator.Flush();
            return new ColumnMesh(opaque, new ColumnMesh.QuadList(null, 0, new int[_sections + 1]), water, overflow);
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
            _compact((Index1D)(_sections * HeightmapMeshKernels.ScratchQuadsPerSection),
                scratch.View, _counters.View.SubView(offset, _sections), sectionStartBuffer.View, output.View, _sections, EmbedSectionIndex ? 1 : 0);
            return new ColumnMesh.QuadList(output, total, start);
        }

        public void Dispose()
        {
            _typeFlags.Dispose();
            _hood.Dispose();
            _scratchOpaque.Dispose(); _scratchWater.Dispose();
            _counters.Dispose();
            foreach (var b in _sectionStart) b.Dispose();
        }
    }
}
