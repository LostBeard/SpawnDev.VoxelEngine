using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Complete GPU meshing pipeline: blocks -> face cull -> greedy merge -> PackedQuad buffer.
    ///
    /// Wraps FaceCullKernels + GreedyMergeKernels into a single async call.
    /// Output is a GPU-resident PackedQuad buffer ready for VertexPullPipeline.DrawSection().
    /// No CPU readback on the hot path. Data stays on GPU from input to render.
    ///
    /// Usage:
    ///   var pipeline = new VoxelMeshPipeline(accelerator);
    ///   var result = await pipeline.MeshSectionAsync(paddedBlocks, sectionSize, sectionHeight);
    ///   // result.QuadBuffer is ready for VertexPullPipeline.DrawSection()
    ///   // result.QuadCount is the number of quads to draw
    ///
    /// Device-adaptive: queries accelerator limits at init to set buffer sizes and budgets.
    /// Works on Quest 3S (low-end mobile GPU) through RTX 4090 (desktop).
    /// </summary>
    public class VoxelMeshPipeline : IDisposable
    {
        private readonly Accelerator _accelerator;

        // Compiled kernels
        private readonly Action<Index2D, ArrayView<int>, ArrayView<long>, int, int> _occupancyKernel;
        private readonly Action<Index2D, ArrayView<long>, ArrayView<long>, int, int> _faceCullKernel;
        private readonly Action<Index2D, ArrayView<long>, ArrayView<int>, ArrayView<long>, ArrayView<int>, int, int, int> _mergeKernel;

        // Shared GPU buffers (reused across MeshSectionAsync calls)
        private MemoryBuffer1D<long, Stride1D.Dense>? _occupancyBuffer;
        private MemoryBuffer1D<long, Stride1D.Dense>? _faceMaskBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _counterBuffer;
        private int _lastPaddedXZ;
        private int _lastInnerCount;

        // Device-adaptive budget
        private readonly int _maxQuadsPerSection;
        private readonly long _maxGpuMemoryBytes;

        /// <summary>Whether the pipeline is initialized and ready.</summary>
        public bool IsReady { get; private set; }

        /// <summary>Maximum quads per section (device-adaptive).</summary>
        public int MaxQuadsPerSection => _maxQuadsPerSection;

        /// <summary>
        /// Create a new mesh pipeline bound to an accelerator.
        /// Compiles all kernels and queries device limits.
        /// </summary>
        public VoxelMeshPipeline(Accelerator accelerator)
        {
            _accelerator = accelerator;

            // Compile kernels once
            _occupancyKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<int>, ArrayView<long>, int, int>(
                FaceCullKernels.BuildOccupancyKernel);

            _faceCullKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<long>, ArrayView<long>, int, int>(
                FaceCullKernels.FaceCullKernel);

            _mergeKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<long>, ArrayView<int>, ArrayView<long>, ArrayView<int>,
                int, int, int>(GreedyMergeKernels.GreedyMergeKernel);

            // Device-adaptive budget
            // Query max memory and set conservative quad budget.
            // PackedQuad = 8 bytes. Intermediate buffers also consume memory.
            // Target: no single section exceeds 5% of available GPU memory.
            _maxGpuMemoryBytes = (long)accelerator.MemorySize;
            long fivePercent = _maxGpuMemoryBytes / 20;
            _maxQuadsPerSection = (int)Math.Min(fivePercent / 8, 2_000_000); // cap at 2M quads
            if (_maxQuadsPerSection < 10_000) _maxQuadsPerSection = 10_000; // floor for tiny GPUs

            // Pre-allocate counter buffer (1 int, reused for every call)
            _counterBuffer = accelerator.Allocate1D(new int[] { 0 });

            IsReady = true;
        }

        /// <summary>
        /// Result of a mesh operation. Contains GPU-resident data ready for rendering.
        /// The caller owns the QuadBuffer and must dispose it when the section is unloaded.
        /// </summary>
        public readonly struct MeshResult
        {
            /// <summary>GPU buffer containing PackedQuad data (i64 per quad). Null if section is empty (all air).</summary>
            public readonly MemoryBuffer1D<long, Stride1D.Dense>? QuadBuffer;

            /// <summary>Number of valid quads in the buffer.</summary>
            public readonly int QuadCount;

            /// <summary>Whether the section produced any visible quads.</summary>
            public bool HasMesh => QuadCount > 0 && QuadBuffer != null;

            public MeshResult(MemoryBuffer1D<long, Stride1D.Dense>? buffer, int count)
            {
                QuadBuffer = buffer;
                QuadCount = count;
            }

            /// <summary>Empty result for all-air sections.</summary>
            public static readonly MeshResult Empty = new(null, 0);
        }

        /// <summary>
        /// Run the full meshing pipeline for one section.
        ///
        /// Input: padded block data as int[] using PackedBlock format.
        /// Padded means (sectionSize+2) in X and Z, with neighbor data in the 1-block border.
        /// This border is required for correct face culling at section boundaries.
        ///
        /// Output: GPU-resident PackedQuad buffer + count. Ready for VertexPullPipeline.DrawSection().
        /// The returned MeshResult.QuadBuffer is owned by the caller and must be disposed.
        /// </summary>
        /// <param name="paddedBlocks">Block data with 1-block padding. Size: (sectionSize+2)*(sectionSize+2)*height. Uses PackedBlock.Pack() encoding.</param>
        /// <param name="sectionSize">Interior section size (e.g., 16 for a 16x16 section). Padding adds +2.</param>
        /// <param name="height">Section height in blocks (e.g., 16 for section-based, or 256 for chunk-column).</param>
        public async Task<MeshResult> MeshSectionAsync(int[] paddedBlocks, int sectionSize, int height)
        {
            if (height > 64)
                throw new ArgumentOutOfRangeException(nameof(height),
                    $"Section height {height} exceeds maximum of 64. " +
                    "The occupancy kernel uses 64-bit columns (1 bit per Y level). " +
                    "Split tall chunks into 16x16x16 sections before meshing.");

            int paddedXZ = sectionSize + 2;
            int innerXZ = sectionSize;
            int innerCount = innerXZ * innerXZ;

            // Ensure shared intermediate buffers are large enough
            EnsureBuffers(paddedXZ, innerCount, height);

            // Upload blocks
            using var gpuBlocks = _accelerator.Allocate1D(paddedBlocks);

            // Step 1: Build occupancy columns
            _occupancyKernel(new Index2D(paddedXZ, paddedXZ),
                gpuBlocks.View, _occupancyBuffer!.View, paddedXZ, height);
            await _accelerator.SynchronizeAsync();

            // Step 2: Face cull (bitwise)
            _faceCullKernel(new Index2D(paddedXZ, paddedXZ),
                _occupancyBuffer!.View, _faceMaskBuffer!.View, paddedXZ, height);
            await _accelerator.SynchronizeAsync();

            // Step 3: Greedy merge
            int maxMergeLayer = Math.Max(height, sectionSize);
            int maxQuads = Math.Min(innerCount * height * 6, _maxQuadsPerSection);
            using var gpuOutputQuads = _accelerator.Allocate1D<long>(maxQuads);

            // Reset counter to 0
            _counterBuffer!.CopyFromCPU(new int[] { 0 });

            _mergeKernel(new Index2D(maxMergeLayer, 6),
                _faceMaskBuffer!.View, gpuBlocks.View, gpuOutputQuads.View, _counterBuffer.View,
                sectionSize, height, paddedXZ);
            await _accelerator.SynchronizeAsync();

            // Read counter (only 4 bytes - negligible transfer)
            var counterResult = await _counterBuffer.CopyToHostAsync();
            int quadCount = counterResult[0];

            if (quadCount <= 0)
                return MeshResult.Empty;

            // Clamp to allocated size
            if (quadCount > maxQuads)
                quadCount = maxQuads;

            // Allocate a right-sized output buffer owned by the caller.
            // Copy only the used portion from the oversized working buffer.
            var resultBuffer = _accelerator.Allocate1D<long>(quadCount);
            resultBuffer.View.SubView(0, quadCount).CopyFrom(
                gpuOutputQuads.View.SubView(0, quadCount));
            await _accelerator.SynchronizeAsync();

            return new MeshResult(resultBuffer, quadCount);
        }

        /// <summary>
        /// Convenience overload: mesh from unpadded block data.
        /// Automatically pads with air (0) on all sides.
        /// Use this when you don't have neighbor section data.
        /// Faces at section boundaries will be visible (air border).
        /// </summary>
        /// <param name="blocks">Unpadded block data. Size: sectionSize * sectionSize * height.</param>
        /// <param name="sectionSize">Section size (e.g., 16).</param>
        /// <param name="height">Section height (e.g., 16).</param>
        public Task<MeshResult> MeshSectionUnpaddedAsync(int[] blocks, int sectionSize, int height)
        {
            int paddedXZ = sectionSize + 2;
            var padded = new int[paddedXZ * paddedXZ * height];

            // Copy blocks into center of padded array (offset by 1 in X and Z)
            for (int y = 0; y < height; y++)
                for (int z = 0; z < sectionSize; z++)
                    for (int x = 0; x < sectionSize; x++)
                        padded[(x + 1) + (z + 1) * paddedXZ + y * paddedXZ * paddedXZ] =
                            blocks[x + z * sectionSize + y * sectionSize * sectionSize];

            return MeshSectionAsync(padded, sectionSize, height);
        }

        /// <summary>
        /// Ensure shared intermediate buffers are allocated and large enough.
        /// Buffers grow but never shrink (avoids allocation churn).
        /// </summary>
        private void EnsureBuffers(int paddedXZ, int innerCount, int height)
        {
            if (_occupancyBuffer == null || _lastPaddedXZ < paddedXZ)
            {
                _occupancyBuffer?.Dispose();
                _occupancyBuffer = _accelerator.Allocate1D<long>(paddedXZ * paddedXZ);
            }

            int neededMasks = innerCount * 6;
            if (_faceMaskBuffer == null || _lastInnerCount < innerCount)
            {
                _faceMaskBuffer?.Dispose();
                _faceMaskBuffer = _accelerator.Allocate1D<long>(neededMasks);
            }

            _lastPaddedXZ = paddedXZ;
            _lastInnerCount = innerCount;
        }

        public void Dispose()
        {
            _occupancyBuffer?.Dispose();
            _faceMaskBuffer?.Dispose();
            _counterBuffer?.Dispose();
            IsReady = false;
        }
    }
}
