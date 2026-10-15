using ILGPU;
using ILGPU.Runtime;
using SpawnDev.VoxelEngine.Buffers;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>Where a quad list lives in a <see cref="QuadArena"/>: <paramref name="Count"/> quads from quad <paramref name="First"/> of block <paramref name="Block"/>.</summary>
    public readonly record struct QuadRange(int Block, int First, int Count);

    /// <summary>
    /// A few large GPU buffers that hold the quads of many meshes, instead of one small buffer per mesh. A renderer then binds a buffer
    /// once for hundreds of columns (a bind and a draw are calls into the browser, and on a weak CPU they are the whole frame) and the
    /// GPU is not asked to create and destroy three buffers for every column that streams in or is re-meshed.
    ///
    /// Give one to <see cref="ColumnMesher.Arena"/> / <see cref="HeightmapMesher.Arena"/>: the compact pass then writes each list
    /// straight to its place in a block. A list is drawn with <c>Draw(pass, list.FirstQuad, list.Count, ...)</c> and goes back to the arena
    /// when the mesh is disposed. A block is added when nothing fits; blocks are kept until the arena is disposed, so a bind group
    /// cached for a block's buffer stays valid.
    ///
    /// Not thread-safe (like the meshers): allocate and free from one thread at a time.
    /// </summary>
    public sealed class QuadArena : IDisposable
    {
        /// <summary>Quads per block unless one list needs more: 2 Mi quads = 16 MiB, well inside WebGPU's default 128 MiB binding limit.</summary>
        public const int DefaultQuadsPerBlock = 1 << 21;

        private sealed class Block
        {
            public required MemoryBuffer1D<long, Stride1D.Dense> Buffer;
            public required RangeAllocator Ranges;
        }

        private readonly Accelerator _accelerator;
        private readonly int _quadsPerBlock;
        private readonly List<Block> _blocks = new();
        private bool _disposed;

        public QuadArena(Accelerator accelerator, int quadsPerBlock = DefaultQuadsPerBlock)
        {
            if (quadsPerBlock < 1) throw new ArgumentOutOfRangeException(nameof(quadsPerBlock));
            _accelerator = accelerator;
            _quadsPerBlock = quadsPerBlock;
        }

        /// <summary>Blocks allocated so far.</summary>
        public int BlockCount => _blocks.Count;

        /// <summary>Quads handed out and not freed.</summary>
        public long UsedQuads
        {
            get { long n = 0; foreach (var b in _blocks) n += b.Ranges.Used; return n; }
        }

        /// <summary>Quads the allocated blocks can hold.</summary>
        public long CapacityQuads
        {
            get { long n = 0; foreach (var b in _blocks) n += b.Ranges.Capacity; return n; }
        }

        /// <summary>The buffer of block <paramref name="block"/>: what a renderer binds.</summary>
        public MemoryBuffer1D<long, Stride1D.Dense> BufferOf(int block) => _blocks[block].Buffer;

        /// <summary>Room for <paramref name="quadCount"/> quads: the first block with a long enough free range, else a new block.</summary>
        public QuadRange Allocate(int quadCount)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(QuadArena));
            if (quadCount < 1) throw new ArgumentOutOfRangeException(nameof(quadCount));
            for (int i = 0; i < _blocks.Count; i++)
            {
                int at = _blocks[i].Ranges.Allocate(quadCount);
                if (at >= 0) return new QuadRange(i, at, quadCount);
            }

            int capacity = Math.Max(_quadsPerBlock, quadCount); // a list longer than a block gets a block of its own size
            var block = new Block { Buffer = _accelerator.Allocate1D<long>(capacity), Ranges = new RangeAllocator(capacity) };
            _blocks.Add(block);
            return new QuadRange(_blocks.Count - 1, block.Ranges.Allocate(quadCount), quadCount);
        }

        /// <summary>Give a range back (a mesh was disposed). Its quads are not cleared: the next list written there overwrites them.</summary>
        public void Free(QuadRange range)
        {
            if (_disposed) return; // the blocks went with the arena
            _blocks[range.Block].Ranges.Free(range.First, range.Count);
        }

        /// <summary>
        /// Free every block. Meshing queues its last dispatches without waiting for them (<c>MeshAsync</c> flushes), and a backend may free a
        /// disposed buffer at once (Wasm does): after meshing, <c>await accelerator.SynchronizeAsync()</c> before disposing the arena.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var b in _blocks) b.Buffer.Dispose();
            _blocks.Clear();
        }
    }
}
