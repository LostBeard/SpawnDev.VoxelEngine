using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.VoxelEngine.SDF
{
    /// <summary>
    /// GPU pipeline for SDF terrain meshing via Dual Marching Cubes.
    ///
    /// Pipeline stages (all GPU-resident, no CPU readback on hot path):
    /// 1. Evaluate SDF noise -> short[] SDF values
    /// 2. Classify active cells -> int[] mask
    /// 3. Prefix sum for compaction -> int[] offsets
    /// 4. Generate dual vertices -> float[] positions + normals
    /// 5. Generate quads -> int[] quad indices
    ///
    /// Data stays on GPU from noise evaluation through to final mesh buffer.
    /// Only the quad count (4 bytes) transfers to CPU for draw call setup.
    /// </summary>
    public class SdfMeshPipeline : IDisposable
    {
        private readonly Accelerator _accelerator;

        // Compiled kernels
        private readonly Action<Index3D, ArrayView<short>, float, float, float, float, int, int> _evaluateKernel;
        private readonly Action<Index3D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int> _classifyKernel;
        private readonly Action<Index1D, ArrayView<int>, ArrayView<short>, ArrayView<int>, ArrayView<float>, ArrayView<float>, int, float, float, float, float, int> _vertexKernel;
        private readonly Action<Index3D, ArrayView<short>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _quadKernel;
        private readonly Action<Index3D, ArrayView<short>, float, float, float, float, int, float, float, float, float, float, int> _modifyKernel;

        // Shared buffers (reused across calls)
        private MemoryBuffer1D<int, Stride1D.Dense>? _cellMaskBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _cellCasesBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _cellToVertexBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _counterBuffer;
        private int _lastCellCount;

        public bool IsReady { get; private set; }

        public SdfMeshPipeline(Accelerator accelerator)
        {
            _accelerator = accelerator;

            _evaluateKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<short>, float, float, float, float, int, int>(
                SdfNoiseKernels.EvaluateSdfKernel);

            _classifyKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int>(
                DualMarchingCubesKernels.ClassifyActiveCellsKernel);

            _vertexKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<short>, ArrayView<int>,
                ArrayView<float>, ArrayView<float>,
                int, float, float, float, float, int>(
                DualMarchingCubesKernels.GenerateDualVerticesKernel);

            _quadKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<short>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(
                DualMarchingCubesKernels.GenerateQuadsKernel);

            _modifyKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<short>, float, float, float, float, int, float, float, float, float, float, int>(
                SdfNoiseKernels.ModifySdfSphereKernel);

            IsReady = true;
        }

        /// <summary>
        /// Evaluate SDF noise for a chunk. Produces the raw SDF volume.
        /// </summary>
        public async Task<MemoryBuffer1D<short, Stride1D.Dense>> EvaluateSdfAsync(
            int chunkSize, float chunkWorldX, float chunkWorldY, float chunkWorldZ,
            float voxelSize, int seed)
        {
            var sdfBuffer = _accelerator.Allocate1D<short>(chunkSize * chunkSize * chunkSize);

            _evaluateKernel(new Index3D(chunkSize, chunkSize, chunkSize),
                sdfBuffer.View, chunkWorldX, chunkWorldY, chunkWorldZ,
                voxelSize, chunkSize, seed);
            await _accelerator.SynchronizeAsync();

            return sdfBuffer;
        }

        /// <summary>
        /// Mesh an SDF volume using Dual Marching Cubes.
        /// Returns the mesh result with GPU-resident vertex and quad buffers.
        /// </summary>
        public async Task<SdfMeshResult> MeshSdfAsync(
            MemoryBuffer1D<short, Stride1D.Dense> sdfBuffer,
            int chunkSize, float chunkWorldX, float chunkWorldY, float chunkWorldZ,
            float voxelSize)
        {
            int cellCount = chunkSize - 1;
            int totalCells = cellCount * cellCount * cellCount;

            EnsureBuffers(totalCells);

            // Step 1: Classify active cells
            _classifyKernel(new Index3D(cellCount, cellCount, cellCount),
                sdfBuffer.View, _cellMaskBuffer!.View, _cellCasesBuffer!.View, chunkSize);
            await _accelerator.SynchronizeAsync();

            // Step 2: Count active cells (prefix sum would be better, but simple count works for now)
            // Read back the mask to CPU to count (temporary - will be replaced with GPU scan)
            var maskHost = await _cellMaskBuffer.CopyToHostAsync();
            int activeCellCount = 0;
            var activeCellIds = new int[totalCells];
            for (int i = 0; i < totalCells; i++)
            {
                if (maskHost[i] != 0)
                {
                    activeCellIds[activeCellCount] = i;
                    activeCellCount++;
                }
            }

            if (activeCellCount == 0)
                return SdfMeshResult.Empty;

            // Build cell-to-vertex mapping (-1 for inactive, vertex index for active)
            var cellToVertex = new int[totalCells];
            System.Array.Fill(cellToVertex, -1);
            for (int i = 0; i < activeCellCount; i++)
                cellToVertex[activeCellIds[i]] = i;

            // Upload to GPU
            using var gpuActiveCellIds = _accelerator.Allocate1D(activeCellIds.AsSpan(0, activeCellCount).ToArray());
            _cellToVertexBuffer!.CopyFromCPU(cellToVertex);

            // Step 3: Generate dual vertices (3 floats per vertex for both positions and normals).
            var vertexPositions = _accelerator.Allocate1D<float>(activeCellCount * 3);
            var vertexNormals = _accelerator.Allocate1D<float>(activeCellCount * 3);

            _vertexKernel((Index1D)activeCellCount,
                gpuActiveCellIds.View, sdfBuffer.View, _cellCasesBuffer!.View,
                vertexPositions.View, vertexNormals.View,
                chunkSize, voxelSize, chunkWorldX, chunkWorldY, chunkWorldZ,
                activeCellCount);
            await _accelerator.SynchronizeAsync();

            // Step 4: Generate quads
            int maxQuads = activeCellCount * 3; // upper bound: 3 quads per cell
            var quadOutput = _accelerator.Allocate1D<int>(maxQuads * 4);
            _counterBuffer!.CopyFromCPU(new int[] { 0 });

            _quadKernel(new Index3D(cellCount, cellCount, cellCount),
                sdfBuffer.View, _cellToVertexBuffer!.View, quadOutput.View,
                _counterBuffer.View, chunkSize, maxQuads);
            await _accelerator.SynchronizeAsync();

            // Read quad count (only 4 bytes - negligible transfer)
            var counterResult = await _counterBuffer.CopyToHostAsync();
            int quadCount = counterResult[0];

            if (quadCount <= 0)
            {
                vertexPositions.Dispose();
                vertexNormals.Dispose();
                quadOutput.Dispose();
                return SdfMeshResult.Empty;
            }

            return new SdfMeshResult(vertexPositions, vertexNormals, quadOutput, activeCellCount, quadCount);
        }

        /// <summary>
        /// Full pipeline: evaluate SDF from noise then mesh via DMC.
        /// </summary>
        public async Task<(MemoryBuffer1D<short, Stride1D.Dense> sdf, SdfMeshResult mesh)> GenerateAndMeshAsync(
            int chunkSize, float chunkWorldX, float chunkWorldY, float chunkWorldZ,
            float voxelSize, int seed)
        {
            var sdfBuffer = await EvaluateSdfAsync(chunkSize, chunkWorldX, chunkWorldY, chunkWorldZ, voxelSize, seed);
            var mesh = await MeshSdfAsync(sdfBuffer, chunkSize, chunkWorldX, chunkWorldY, chunkWorldZ, voxelSize);
            return (sdfBuffer, mesh);
        }

        /// <summary>
        /// Modify SDF values in a sphere (terrain deformation).
        /// After calling this, re-mesh the affected chunk(s).
        /// <paramref name="blendRadius"/> controls smooth-CSG blend width (1.0 = default).
        /// </summary>
        public async Task ModifySpherAsync(
            MemoryBuffer1D<short, Stride1D.Dense> sdfBuffer,
            int chunkSize, float chunkWorldX, float chunkWorldY, float chunkWorldZ,
            float voxelSize,
            float centerX, float centerY, float centerZ,
            float radius, int mode, float blendRadius = 1f)
        {
            _modifyKernel(new Index3D(chunkSize, chunkSize, chunkSize),
                sdfBuffer.View, centerX, centerY, centerZ, radius, mode, blendRadius,
                chunkWorldX, chunkWorldY, chunkWorldZ, voxelSize, chunkSize);
            await _accelerator.SynchronizeAsync();
        }

        private void EnsureBuffers(int totalCells)
        {
            if (_lastCellCount >= totalCells) return;

            _cellMaskBuffer?.Dispose();
            _cellCasesBuffer?.Dispose();
            _cellToVertexBuffer?.Dispose();
            _counterBuffer?.Dispose();

            _cellMaskBuffer = _accelerator.Allocate1D<int>(totalCells);
            _cellCasesBuffer = _accelerator.Allocate1D<int>(totalCells);
            _cellToVertexBuffer = _accelerator.Allocate1D<int>(totalCells);
            _counterBuffer = _accelerator.Allocate1D<int>(1);
            _lastCellCount = totalCells;
        }

        public void Dispose()
        {
            _cellMaskBuffer?.Dispose();
            _cellCasesBuffer?.Dispose();
            _cellToVertexBuffer?.Dispose();
            _counterBuffer?.Dispose();
        }
    }

    /// <summary>
    /// Result of SDF mesh generation. Contains GPU-resident vertex and quad buffers.
    /// Caller owns the buffers and must dispose them.
    /// </summary>
    public class SdfMeshResult : IDisposable
    {
        public static readonly SdfMeshResult Empty = new(null, null, null, 0, 0);

        public MemoryBuffer1D<float, Stride1D.Dense>? VertexPositions { get; }
        public MemoryBuffer1D<float, Stride1D.Dense>? VertexNormals { get; }
        public MemoryBuffer1D<int, Stride1D.Dense>? QuadIndices { get; }
        public int VertexCount { get; }
        public int QuadCount { get; }
        public bool HasMesh => QuadCount > 0;

        public SdfMeshResult(
            MemoryBuffer1D<float, Stride1D.Dense>? vertexPositions,
            MemoryBuffer1D<float, Stride1D.Dense>? vertexNormals,
            MemoryBuffer1D<int, Stride1D.Dense>? quadIndices,
            int vertexCount, int quadCount)
        {
            VertexPositions = vertexPositions;
            VertexNormals = vertexNormals;
            QuadIndices = quadIndices;
            VertexCount = vertexCount;
            QuadCount = quadCount;
        }

        public void Dispose()
        {
            VertexPositions?.Dispose();
            VertexNormals?.Dispose();
            QuadIndices?.Dispose();
        }
    }
}
