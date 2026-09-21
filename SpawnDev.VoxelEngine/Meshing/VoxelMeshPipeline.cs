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
        private readonly Action<Index2D, ArrayView<long>, ArrayView<long>, ArrayView<long>, ArrayView<long>, int, int> _faceCullKernel;
        private readonly Action<Index2D, ArrayView<long>, ArrayView<int>, ArrayView<long>, ArrayView<int>, int, int, int, int> _mergeKernel;

        // Shared GPU buffers (reused across MeshSectionAsync calls)
        private MemoryBuffer1D<long, Stride1D.Dense>? _occupancyBuffer;
        private MemoryBuffer1D<long, Stride1D.Dense>? _faceMaskBuffer;
        private MemoryBuffer1D<long, Stride1D.Dense>? _yPadMinusBuffer;
        private MemoryBuffer1D<long, Stride1D.Dense>? _yPadPlusBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _counterBuffer;
        // Reused padded-block upload buffer - do NOT Allocate1D(padded) every section.
        private MemoryBuffer1D<int, Stride1D.Dense>? _gpuBlocksBuffer;
        private int _gpuBlocksCapacity;
        private long[]? _yPadMinusStaging;
        private long[]? _yPadPlusStaging;
        // CPU staging for MeshChunkColumnAsync - reused across remesh calls (dig hot path).
        private int[]? _paddedStaging;
        private int[]? _yPadMinusIntStaging;
        private int[]? _yPadPlusIntStaging;
        private int[]? _uploadPrefix; // exact-length GPU upload when staging is oversized
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
                Index2D, ArrayView<long>, ArrayView<long>, ArrayView<long>, ArrayView<long>, int, int>(
                FaceCullKernels.FaceCullKernelWithYPad);

            _mergeKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<long>, ArrayView<int>, ArrayView<long>, ArrayView<int>,
                int, int, int, int>(GreedyMergeKernels.GreedyMergeKernel);

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
        /// <param name="yPadMinusSlab">Optional XZ slice of the -Y-neighbor section at its top layer (y=sectionHeight-1 in that neighbor).
        /// Size (sectionSize+2)*(sectionSize+2) as int PackedBlock values. When a block is solid (lower 12 bits non-zero)
        /// at an interior (x,z) position, the corresponding y=0 voxel in THIS section will not emit a -Y face.
        /// Null = air neighbor (bottom boundary faces remain visible).</param>
        /// <param name="yPadPlusSlab">Optional XZ slice of the +Y-neighbor section at its bottom layer (y=0 in that neighbor).
        /// Same format as yPadMinusSlab. Controls whether y=height-1 voxels emit a +Y face.
        /// Null = air neighbor (top boundary faces remain visible).</param>
        public async Task<MeshResult> MeshSectionAsync(
            int[] paddedBlocks,
            int sectionSize,
            int height,
            int[]? yPadMinusSlab = null,
            int[]? yPadPlusSlab = null)
        {
            if (height > 64)
                throw new ArgumentOutOfRangeException(nameof(height),
                    $"Section height {height} exceeds maximum of 64. " +
                    "The occupancy kernel uses 64-bit columns (1 bit per Y level). " +
                    "Split tall chunks into 16x16x16 sections before meshing.");

            int paddedXZ = sectionSize + 2;
            int innerXZ = sectionSize;
            int innerCount = innerXZ * innerXZ;
            int expectedLen = paddedXZ * paddedXZ * height;
            if (paddedBlocks.Length < expectedLen)
                throw new ArgumentException(
                    $"paddedBlocks length {paddedBlocks.Length} is less than (sectionSize+2)^2 * height = {expectedLen}",
                    nameof(paddedBlocks));

            int slabLen = paddedXZ * paddedXZ;
            if (yPadMinusSlab != null && yPadMinusSlab.Length != slabLen)
                throw new ArgumentException(
                    $"yPadMinusSlab length {yPadMinusSlab.Length} does not match (sectionSize+2)^2 = {slabLen}",
                    nameof(yPadMinusSlab));
            if (yPadPlusSlab != null && yPadPlusSlab.Length != slabLen)
                throw new ArgumentException(
                    $"yPadPlusSlab length {yPadPlusSlab.Length} does not match (sectionSize+2)^2 = {slabLen}",
                    nameof(yPadPlusSlab));

            // Ensure shared intermediate buffers are large enough
            EnsureBuffers(paddedXZ, innerCount, height);
            EnsureGpuBlocksCapacity(expectedLen);

            // Upload into reused GPU buffer (dig/remesh hot path - no per-call Allocate1D).
            // CopyFromCPU requires a T[] of matching view length - use prefix scratch if staging is oversized.
            int[] uploadSrc = paddedBlocks;
            if (paddedBlocks.Length != expectedLen)
            {
                if (_uploadPrefix == null || _uploadPrefix.Length != expectedLen)
                    _uploadPrefix = new int[expectedLen];
                Array.Copy(paddedBlocks, 0, _uploadPrefix, 0, expectedLen);
                uploadSrc = _uploadPrefix;
            }
            _gpuBlocksBuffer!.View.SubView(0, expectedLen).CopyFromCPU(uploadSrc);

            // Step 0: Prepare Y-pad slabs. Bit 0 of each long = solid (lower 12 bits non-zero in the PackedBlock source).
            UploadYPadSlab(yPadMinusSlab, ref _yPadMinusStaging, _yPadMinusBuffer!, slabLen);
            UploadYPadSlab(yPadPlusSlab, ref _yPadPlusStaging, _yPadPlusBuffer!, slabLen);

            // Kernel chain: occupancy -> face-cull -> merge(0-3) -> merge(4-5).
            // All four dispatches run on the same ILGPU default stream, so the runtime
            // serializes them automatically - no SynchronizeAsync needed between them.
            // The only load-bearing sync is the one before we CPU-read the counter below.
            //
            // Step 1: Build occupancy columns
            var blocksView = _gpuBlocksBuffer.View.SubView(0, expectedLen);
            _occupancyKernel(new Index2D(paddedXZ, paddedXZ),
                blocksView, _occupancyBuffer!.View, paddedXZ, height);

            // Step 2: Face cull (bitwise), Y-padded
            _faceCullKernel(new Index2D(paddedXZ, paddedXZ),
                _occupancyBuffer!.View,
                _yPadMinusBuffer!.View, _yPadPlusBuffer!.View,
                _faceMaskBuffer!.View, paddedXZ, height);

            // Step 3: Greedy merge (two dispatches to eliminate +Y/-Y race condition)
            int maxQuads = Math.Min(innerCount * height * 6, _maxQuadsPerSection);
            using var gpuOutputQuads = _accelerator.Allocate1D<long>(maxQuads);

            // Reset counter to 0 (queued on same stream, runs before merge dispatches)
            _counterBuffer!.CopyFromCPU(new int[] { 0 });

            // Dispatch 1: faces 0-3 (+X,-X,+Z,-Z) - one thread per layer, no race
            _mergeKernel(new Index2D(sectionSize, 4),
                _faceMaskBuffer!.View, blocksView, gpuOutputQuads.View, _counterBuffer.View,
                sectionSize, height, paddedXZ, 0);

            // Dispatch 2: faces 4-5 (+Y,-Y) - one thread per Y layer, full XZ merge
            // Separate dispatch ensures no concurrent access to the same faceMask entries.
            // Stream serialization (not an explicit sync) is what guarantees dispatch 1
            // finishes before dispatch 2 reads the shared faceMask buffer.
            // faceStart=4 offsets face indices so dispatch face 0,1 -> kernel face 4,5
            _mergeKernel(new Index2D(height, 2),
                _faceMaskBuffer!.View, blocksView, gpuOutputQuads.View, _counterBuffer.View,
                sectionSize, height, paddedXZ, 4);

            // Sync before CPU readback of the counter. Only load-bearing sync in the pipeline.
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
        /// Mesh every 16-high (or sectionSize-high) section of a full chunk column in one call,
        /// feeding raw chunk bytes directly. Skips all-air sections before any kernel dispatch —
        /// no GPU work is done for sections whose interior blocks are all zero, which for typical
        /// terrain (geometry concentrated in a narrow Y band) eliminates ~80% of mesh dispatches.
        ///
        /// Layout of `blocks` (and each optional neighbor): flat byte[] indexed as
        ///   b[x + z * sectionSize + y * sectionSize * sectionSize]
        /// where x,z are in [0,sectionSize) and y is in [0,totalHeight). A value of 0 is air.
        ///
        /// XZ neighbor blocks (if supplied) are used only to fill the 1-block padding border so
        /// boundary faces adjacent to solid neighbors are culled. Y padding within the column is
        /// built automatically from the chunk's own adjacent sections (no see-through at y=16,32,...).
        ///
        /// Returns a list of (sectionY, MeshResult) for sections that produced at least one quad.
        /// Empty sections are omitted entirely.
        /// </summary>
        /// <param name="blocks">Full chunk-column block bytes. Size = sectionSize * sectionSize * totalHeight.</param>
        /// <param name="neighborXMinus">Blocks for chunk at (cx-1, cz), or null for air neighbor.</param>
        /// <param name="neighborXPlus">Blocks for chunk at (cx+1, cz), or null for air neighbor.</param>
        /// <param name="neighborZMinus">Blocks for chunk at (cx, cz-1), or null for air neighbor.</param>
        /// <param name="neighborZPlus">Blocks for chunk at (cx, cz+1), or null for air neighbor.</param>
        /// <param name="sectionSize">XZ size of a section. Must be &lt;= 16 for face-mask kernels. Default 16.</param>
        /// <param name="totalHeight">Total chunk column height. Must be a multiple of sectionSize. Default 256.</param>
        public Task<List<(int sectionY, MeshResult mesh)>> MeshChunkColumnAsync(
            byte[] blocks,
            byte[]? neighborXMinus = null,
            byte[]? neighborXPlus = null,
            byte[]? neighborZMinus = null,
            byte[]? neighborZPlus = null,
            int sectionSize = 16,
            int totalHeight = 256)
            => MeshChunkColumnAsync(blocks, neighborXMinus, neighborXPlus, neighborZMinus, neighborZPlus,
                sectionSize, totalHeight, minSectionY: null, maxSectionY: null, sectionYs: null);

        /// <summary>
        /// Mesh only sections whose section-Y index is in [minSectionY, maxSectionY] inclusive.
        /// Prefer this on dig/carve remesh so untouched bedrock/sky sections are not re-uploaded.
        /// </summary>
        public Task<List<(int sectionY, MeshResult mesh)>> MeshChunkColumnAsync(
            byte[] blocks,
            byte[]? neighborXMinus,
            byte[]? neighborXPlus,
            byte[]? neighborZMinus,
            byte[]? neighborZPlus,
            int sectionSize,
            int totalHeight,
            int minSectionY,
            int maxSectionY)
            => MeshChunkColumnAsync(blocks, neighborXMinus, neighborXPlus, neighborZMinus, neighborZPlus,
                sectionSize, totalHeight, minSectionY, maxSectionY, sectionYs: null);

        /// <summary>
        /// Mesh only the listed section-Y indices. Duplicate indices are ignored.
        /// Null/empty <paramref name="sectionYs"/> meshes the full column (same as the base overload).
        /// </summary>
        public Task<List<(int sectionY, MeshResult mesh)>> MeshChunkColumnSectionsAsync(
            byte[] blocks,
            IReadOnlyCollection<int> sectionYs,
            byte[]? neighborXMinus = null,
            byte[]? neighborXPlus = null,
            byte[]? neighborZMinus = null,
            byte[]? neighborZPlus = null,
            int sectionSize = 16,
            int totalHeight = 256)
            => MeshChunkColumnAsync(blocks, neighborXMinus, neighborXPlus, neighborZMinus, neighborZPlus,
                sectionSize, totalHeight, minSectionY: null, maxSectionY: null, sectionYs: sectionYs);

        private async Task<List<(int sectionY, MeshResult mesh)>> MeshChunkColumnAsync(
            byte[] blocks,
            byte[]? neighborXMinus,
            byte[]? neighborXPlus,
            byte[]? neighborZMinus,
            byte[]? neighborZPlus,
            int sectionSize,
            int totalHeight,
            int? minSectionY,
            int? maxSectionY,
            IReadOnlyCollection<int>? sectionYs)
        {
            if (blocks == null) throw new ArgumentNullException(nameof(blocks));
            if (sectionSize <= 0) throw new ArgumentOutOfRangeException(nameof(sectionSize));
            if (totalHeight <= 0 || totalHeight % sectionSize != 0)
                throw new ArgumentOutOfRangeException(nameof(totalHeight),
                    $"totalHeight {totalHeight} must be a positive multiple of sectionSize {sectionSize}");

            int sectionsPerColumn = totalHeight / sectionSize;
            int xzArea = sectionSize * sectionSize;
            int columnLen = xzArea * totalHeight;
            if (blocks.Length != columnLen)
                throw new ArgumentException(
                    $"blocks length {blocks.Length} does not match sectionSize*sectionSize*totalHeight = {columnLen}",
                    nameof(blocks));
            ValidateNeighborLength(neighborXMinus, columnLen, nameof(neighborXMinus));
            ValidateNeighborLength(neighborXPlus, columnLen, nameof(neighborXPlus));
            ValidateNeighborLength(neighborZMinus, columnLen, nameof(neighborZMinus));
            ValidateNeighborLength(neighborZPlus, columnLen, nameof(neighborZPlus));

            int paddedXZ = sectionSize + 2;
            int paddedSlabLen = paddedXZ * paddedXZ;
            int paddedLen = paddedSlabLen * sectionSize;

            // Reuse member staging - dig remesh calls this every stroke; new int[] each time is treason.
            if (_paddedStaging == null || _paddedStaging.Length < paddedLen)
                _paddedStaging = new int[paddedLen];
            if (_yPadMinusIntStaging == null || _yPadMinusIntStaging.Length < paddedSlabLen)
                _yPadMinusIntStaging = new int[paddedSlabLen];
            if (_yPadPlusIntStaging == null || _yPadPlusIntStaging.Length < paddedSlabLen)
                _yPadPlusIntStaging = new int[paddedSlabLen];

            var padded = _paddedStaging;
            var yPadMinusSlab = _yPadMinusIntStaging;
            var yPadPlusSlab = _yPadPlusIntStaging;

            HashSet<int>? syFilter = null;
            if (sectionYs != null && sectionYs.Count > 0)
            {
                syFilter = new HashSet<int>(sectionYs.Count);
                foreach (int sy in sectionYs)
                    if (sy >= 0 && sy < sectionsPerColumn)
                        syFilter.Add(sy);
            }

            int syStart = minSectionY ?? 0;
            int syEnd = maxSectionY ?? (sectionsPerColumn - 1);
            if (syStart < 0) syStart = 0;
            if (syEnd >= sectionsPerColumn) syEnd = sectionsPerColumn - 1;

            var results = new List<(int, MeshResult)>();

            for (int sy = syStart; sy <= syEnd; sy++)
            {
                if (syFilter != null && !syFilter.Contains(sy))
                    continue;

                int yOffset = sy * sectionSize;

                if (IsSectionInteriorAllAir(blocks, yOffset, sectionSize, xzArea))
                    continue;

                Array.Clear(padded, 0, paddedLen);
                for (int y = 0; y < sectionSize; y++)
                {
                    int paddedYBase = y * paddedSlabLen;
                    int srcYBase = (yOffset + y) * xzArea;

                    for (int z = 0; z < sectionSize; z++)
                        for (int x = 0; x < sectionSize; x++)
                        {
                            byte b = blocks[x + z * sectionSize + srcYBase];
                            if (b != 0)
                                padded[(x + 1) + (z + 1) * paddedXZ + paddedYBase] = PackedBlock.Pack(b);
                        }

                    if (neighborXMinus != null)
                        for (int z = 0; z < sectionSize; z++)
                        {
                            byte b = neighborXMinus[(sectionSize - 1) + z * sectionSize + srcYBase];
                            if (b != 0)
                                padded[0 + (z + 1) * paddedXZ + paddedYBase] = PackedBlock.Pack(b);
                        }

                    if (neighborXPlus != null)
                        for (int z = 0; z < sectionSize; z++)
                        {
                            byte b = neighborXPlus[0 + z * sectionSize + srcYBase];
                            if (b != 0)
                                padded[(sectionSize + 1) + (z + 1) * paddedXZ + paddedYBase] = PackedBlock.Pack(b);
                        }

                    if (neighborZMinus != null)
                        for (int x = 0; x < sectionSize; x++)
                        {
                            byte b = neighborZMinus[x + (sectionSize - 1) * sectionSize + srcYBase];
                            if (b != 0)
                                padded[(x + 1) + 0 * paddedXZ + paddedYBase] = PackedBlock.Pack(b);
                        }

                    if (neighborZPlus != null)
                        for (int x = 0; x < sectionSize; x++)
                        {
                            byte b = neighborZPlus[x + 0 * sectionSize + srcYBase];
                            if (b != 0)
                                padded[(x + 1) + (sectionSize + 1) * paddedXZ + paddedYBase] = PackedBlock.Pack(b);
                        }
                }

                int[]? yMinusArg = null;
                int[]? yPlusArg = null;

                if (sy > 0)
                {
                    Array.Clear(yPadMinusSlab, 0, paddedSlabLen);
                    int srcYBase = (yOffset - 1) * xzArea;
                    for (int z = 0; z < sectionSize; z++)
                        for (int x = 0; x < sectionSize; x++)
                        {
                            byte b = blocks[x + z * sectionSize + srcYBase];
                            if (b != 0)
                                yPadMinusSlab[(x + 1) + (z + 1) * paddedXZ] = PackedBlock.Pack(b);
                        }
                    yMinusArg = yPadMinusSlab;
                }

                if (sy < sectionsPerColumn - 1)
                {
                    Array.Clear(yPadPlusSlab, 0, paddedSlabLen);
                    int srcYBase = (yOffset + sectionSize) * xzArea;
                    for (int z = 0; z < sectionSize; z++)
                        for (int x = 0; x < sectionSize; x++)
                        {
                            byte b = blocks[x + z * sectionSize + srcYBase];
                            if (b != 0)
                                yPadPlusSlab[(x + 1) + (z + 1) * paddedXZ] = PackedBlock.Pack(b);
                        }
                    yPlusArg = yPadPlusSlab;
                }

                var result = await MeshSectionAsync(padded, sectionSize, sectionSize, yMinusArg, yPlusArg);
                if (result.HasMesh)
                    results.Add((sy, result));
            }

            return results;
        }

        /// <summary>
        /// Upload a full column of byte block IDs into a persistent GPU int buffer (PackedBlock).
        /// Caller owns the returned buffer and must dispose it. Use with MeshFromGpuColumnAsync
        /// or ExplosionKernels.DestroyKernel so dig edits stay GPU-resident.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> UploadColumnPacked(byte[] blocks)
        {
            if (blocks == null) throw new ArgumentNullException(nameof(blocks));
            var packed = new int[blocks.Length];
            for (int i = 0; i < blocks.Length; i++)
            {
                byte b = blocks[i];
                if (b != 0) packed[i] = PackedBlock.Pack(b);
            }
            return _accelerator.Allocate1D(packed);
        }

        /// <summary>
        /// Copy a contiguous Y slab of a byte column into an existing GPU packed column buffer.
        /// Indices are flat column indices: start = yStart * xzArea, length = yCount * xzArea.
        /// Avoids re-uploading the entire 65KB+ column when only a dig AABB changed.
        /// </summary>
        public void CopyColumnSlabFromCpu(
            MemoryBuffer1D<int, Stride1D.Dense> gpuColumn,
            byte[] blocks,
            int yStart,
            int yCount,
            int sectionSize)
        {
            if (gpuColumn == null) throw new ArgumentNullException(nameof(gpuColumn));
            if (blocks == null) throw new ArgumentNullException(nameof(blocks));
            if (yCount <= 0) return;

            int xzArea = sectionSize * sectionSize;
            int start = yStart * xzArea;
            int len = yCount * xzArea;
            if (start < 0 || start + len > blocks.Length)
                throw new ArgumentOutOfRangeException(nameof(yStart),
                    $"Slab [{yStart},{yStart + yCount}) out of column bounds");
            if (start + len > gpuColumn.Length)
                throw new ArgumentOutOfRangeException(nameof(gpuColumn), "GPU column shorter than slab");

            // Exact-length array required by CopyFromCPU on SubView.
            if (_uploadPrefix == null || _uploadPrefix.Length != len)
                _uploadPrefix = new int[len];
            for (int i = 0; i < len; i++)
            {
                byte b = blocks[start + i];
                _uploadPrefix[i] = b != 0 ? PackedBlock.Pack(b) : 0;
            }
            gpuColumn.View.SubView(start, len).CopyFromCPU(_uploadPrefix);
        }

        private void EnsureGpuBlocksCapacity(int needed)
        {
            if (_gpuBlocksBuffer != null && _gpuBlocksCapacity >= needed)
                return;
            _gpuBlocksBuffer?.Dispose();
            _gpuBlocksCapacity = Math.Max(needed, 18 * 18 * 16); // default one padded section
            _gpuBlocksBuffer = _accelerator.Allocate1D<int>(_gpuBlocksCapacity);
        }

        private static bool IsSectionInteriorAllAir(byte[] blocks, int yOffset, int sectionSize, int xzArea)
        {
            // Treat any non-zero block byte as "has geometry". Kernel later only emits quads
            // for cells whose lower 12 bits of PackedBlock are non-zero, which is equivalent
            // for single-byte block IDs <= 255.
            int start = yOffset * xzArea;
            int end = start + sectionSize * xzArea;
            for (int i = start; i < end; i++)
                if (blocks[i] != 0) return false;
            return true;
        }

        private static void ValidateNeighborLength(byte[]? neighbor, int expectedLen, string paramName)
        {
            if (neighbor != null && neighbor.Length != expectedLen)
                throw new ArgumentException(
                    $"Neighbor block buffer length {neighbor.Length} does not match expected column length {expectedLen}",
                    paramName);
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
            int slabSize = paddedXZ * paddedXZ;
            if (_occupancyBuffer == null || _lastPaddedXZ < paddedXZ)
            {
                _occupancyBuffer?.Dispose();
                _occupancyBuffer = _accelerator.Allocate1D<long>(slabSize);

                _yPadMinusBuffer?.Dispose();
                _yPadMinusBuffer = _accelerator.Allocate1D<long>(slabSize);
                _yPadPlusBuffer?.Dispose();
                _yPadPlusBuffer = _accelerator.Allocate1D<long>(slabSize);
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

        /// <summary>
        /// Convert an int[] PackedBlock slab into a long[] with bit 0 set iff
        /// the block at that XZ position is solid (lower 12 bits non-zero),
        /// and upload to the given GPU buffer. If the input slab is null,
        /// uploads a zero-filled buffer (matches air-neighbor behavior).
        /// </summary>
        private void UploadYPadSlab(
            int[]? slab,
            ref long[]? staging,
            MemoryBuffer1D<long, Stride1D.Dense> gpuBuffer,
            int slabLen)
        {
            if (staging == null || staging.Length != slabLen)
                staging = new long[slabLen];

            if (slab == null)
            {
                Array.Clear(staging, 0, slabLen);
            }
            else
            {
                for (int i = 0; i < slabLen; i++)
                    staging[i] = (slab[i] & 0xFFF) != 0 ? 1L : 0L;
            }

            gpuBuffer.View.SubView(0, slabLen).CopyFromCPU(staging);
        }

        public void Dispose()
        {
            _occupancyBuffer?.Dispose();
            _faceMaskBuffer?.Dispose();
            _yPadMinusBuffer?.Dispose();
            _yPadPlusBuffer?.Dispose();
            _counterBuffer?.Dispose();
            _gpuBlocksBuffer?.Dispose();
            IsReady = false;
        }
    }
}
