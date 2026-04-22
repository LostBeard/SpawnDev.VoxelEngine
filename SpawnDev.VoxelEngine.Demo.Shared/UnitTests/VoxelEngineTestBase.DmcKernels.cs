using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.SDF;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Dual Marching Cubes kernel tests: cell classification, vertex generation, quad output.
    public abstract partial class VoxelEngineTestBase
    {
        // ===================================================================
        // ClassifyActiveCellsKernel
        // ===================================================================

        [TestMethod]
        public async Task DmcClassify_AllSolid_NoActiveCells() => await RunTest(async accelerator =>
        {
            // Every voxel positive -> every cell is fully inside solid -> 0 active cells.
            int size = 4;
            var sdf = FilledSdf(size, (short)500);
            var (mask, cases) = await ClassifyCells(accelerator, sdf, size);
            int activeCount = 0;
            foreach (var m in mask) if (m != 0) activeCount++;
            if (activeCount != 0)
                throw new Exception($"All-solid chunk must have 0 active cells, got {activeCount}");
            // Every case should be 255 (all 8 corners solid).
            foreach (var c in cases)
                if (c != 255)
                    throw new Exception($"All-solid chunk case index should be 255, got {c}");
        });

        [TestMethod]
        public async Task DmcClassify_AllAir_NoActiveCells() => await RunTest(async accelerator =>
        {
            int size = 4;
            var sdf = FilledSdf(size, (short)-500);
            var (mask, cases) = await ClassifyCells(accelerator, sdf, size);
            int activeCount = 0;
            foreach (var m in mask) if (m != 0) activeCount++;
            if (activeCount != 0)
                throw new Exception($"All-air chunk must have 0 active cells, got {activeCount}");
            foreach (var c in cases)
                if (c != 0)
                    throw new Exception($"All-air chunk case index should be 0, got {c}");
        });

        [TestMethod]
        public async Task DmcClassify_HorizontalPlane_OneActiveLayer() => await RunTest(async accelerator =>
        {
            // Terrain fills bottom half (y < midY) with air, top half solid.
            // Only the cell layer straddling the split has active cells.
            int size = 4;
            int cellCount = size - 1; // 3
            int midY = 2; // split between y=1 (air) and y=2 (solid)

            var sdf = new short[size * size * size];
            for (int y = 0; y < size; y++)
            for (int z = 0; z < size; z++)
            for (int x = 0; x < size; x++)
                sdf[x + z * size + y * size * size] = (y >= midY) ? (short)500 : (short)-500;

            var (mask, _) = await ClassifyCells(accelerator, sdf, size);

            // Expected: cells at cy == midY-1 straddle the boundary (corners at y=midY-1 and y=midY).
            // That gives cellCount * cellCount = 9 active cells.
            int expectedActive = cellCount * cellCount;
            int actualActive = 0;
            foreach (var m in mask) if (m != 0) actualActive++;
            if (actualActive != expectedActive)
                throw new Exception($"Horizontal plane: expected {expectedActive} active cells, got {actualActive}");

            // Verify only the correct Y layer is active.
            for (int cy = 0; cy < cellCount; cy++)
            for (int cz = 0; cz < cellCount; cz++)
            for (int cx = 0; cx < cellCount; cx++)
            {
                int idx = cx + cz * cellCount + cy * cellCount * cellCount;
                bool expectActive = (cy == midY - 1);
                bool actuallyActive = mask[idx] != 0;
                if (expectActive != actuallyActive)
                    throw new Exception($"Cell ({cx},{cy},{cz}) expected active={expectActive}, got {actuallyActive}");
            }
        });

        [TestMethod]
        public async Task DmcClassify_SingleCornerSolid_CaseIndexEquals1() => await RunTest(async accelerator =>
        {
            // Only voxel (0,0,0) positive -> corner 0 of cell (0,0,0) is solid, rest are air.
            // Case index bit 0 set -> caseIndex == 1.
            int size = 2; // smallest: one cell
            var sdf = FilledSdf(size, (short)-100);
            sdf[0] = 100; // corner 0 solid

            var (mask, cases) = await ClassifyCells(accelerator, sdf, size);
            if (mask[0] != 1)
                throw new Exception($"Single-corner-solid cell must be active, got mask={mask[0]}");
            if (cases[0] != 1)
                throw new Exception($"Case index should be 1 (bit 0 only), got {cases[0]}");
        });

        // ===================================================================
        // GenerateQuadsKernel
        // ===================================================================

        [TestMethod]
        public async Task DmcQuads_AllSolid_NoQuads() => await RunTest(async accelerator =>
        {
            int size = 4;
            var sdf = FilledSdf(size, (short)500);
            int quadCount = await FullDmcPipelineQuadCount(accelerator, sdf, size);
            if (quadCount != 0)
                throw new Exception($"All-solid chunk must produce 0 quads, got {quadCount}");
        });

        [TestMethod]
        public async Task DmcQuads_AllAir_NoQuads() => await RunTest(async accelerator =>
        {
            int size = 4;
            var sdf = FilledSdf(size, (short)-500);
            int quadCount = await FullDmcPipelineQuadCount(accelerator, sdf, size);
            if (quadCount != 0)
                throw new Exception($"All-air chunk must produce 0 quads, got {quadCount}");
        });

        [TestMethod]
        public async Task DmcQuads_HorizontalPlane_ExactlyFourInteriorQuads() => await RunTest(async accelerator =>
        {
            // Horizontal boundary at midY, size=4 -> 9 active cells (cy=1 layer).
            // GenerateQuadsKernel checks Y-edge only where cx>0 AND cz>0 (to have 4 neighbors),
            // giving 2*2 = 4 interior Y-edge quads. No +X or +Z axis crossings exist on this
            // pure horizontal boundary, so expected total = 4.
            int size = 4;
            int midY = 2;
            var sdf = PlanarSdf(size, midY);

            int quadCount = await FullDmcPipelineQuadCount(accelerator, sdf, size);
            if (quadCount != 4)
                throw new Exception($"Horizontal boundary expected 4 quads, got {quadCount}");
        });

        // ===================================================================
        // GenerateDualVerticesKernel (via full pipeline)
        // ===================================================================

        [TestMethod]
        public async Task DmcVertex_HorizontalPlane_AllVerticesOnPlane() => await RunTest(async accelerator =>
        {
            // On a pure horizontal boundary at midY=2 (split between y=1 air and y=2 solid),
            // all dual vertices must sit on the Y=1.5 plane (edge crossings are exactly t=0.5
            // because abs(SDF) is equal on both sides).
            int size = 4;
            int midY = 2;
            var sdf = PlanarSdf(size, midY);
            float voxelSize = 1f;

            using var pipeline = new SdfMeshPipeline(accelerator);
            using var sdfBuffer = accelerator.Allocate1D(sdf);
            using var mesh = await pipeline.MeshSdfAsync(sdfBuffer, size, 0, 0, 0, voxelSize);

            if (mesh.VertexCount == 0 || mesh.VertexPositions == null)
                throw new Exception("Horizontal plane must produce vertices");

            var positions = await mesh.VertexPositions.CopyToHostAsync();
            // 3 floats per vertex: x,y,z
            for (int v = 0; v < mesh.VertexCount; v++)
            {
                float vy = positions[v * 3 + 1];
                float diff = Math.Abs(vy - 1.5f);
                if (diff > 0.01f)
                    throw new Exception($"Vertex {v} Y={vy} not on plane Y=1.5 (diff {diff})");
            }
        });

        [TestMethod]
        public async Task DmcVertex_HorizontalPlane_CountMatchesActiveCells() => await RunTest(async accelerator =>
        {
            // size=4 -> cellCount=3 -> 9 active cells on horizontal boundary -> 9 dual vertices.
            int size = 4;
            var sdf = PlanarSdf(size, 2);
            using var pipeline = new SdfMeshPipeline(accelerator);
            using var sdfBuffer = accelerator.Allocate1D(sdf);
            using var mesh = await pipeline.MeshSdfAsync(sdfBuffer, size, 0, 0, 0, 1f);

            if (mesh.VertexCount != 9)
                throw new Exception($"Horizontal plane expected 9 vertices, got {mesh.VertexCount}");
        });

        // ===================================================================
        // Helpers
        // ===================================================================

        private async Task<(int[] mask, int[] cases)> ClassifyCells(Accelerator accelerator, short[] sdf, int size)
        {
            int cellCount = size - 1;
            int totalCells = cellCount * cellCount * cellCount;
            using var gpuSdf = accelerator.Allocate1D(sdf);
            using var gpuMask = accelerator.Allocate1D<int>(totalCells);
            using var gpuCases = accelerator.Allocate1D<int>(totalCells);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int>(
                DualMarchingCubesKernels.ClassifyActiveCellsKernel);
            kernel(new Index3D(cellCount, cellCount, cellCount),
                gpuSdf.View, gpuMask.View, gpuCases.View, size);
            await accelerator.SynchronizeAsync();

            var mask = await gpuMask.CopyToHostAsync();
            var cases = await gpuCases.CopyToHostAsync();
            return (mask, cases);
        }

        /// <summary>
        /// Run the full SdfMeshPipeline.MeshSdfAsync path and return the quad count.
        /// Exercises classify + vertex gen + quad gen together.
        /// </summary>
        private async Task<int> FullDmcPipelineQuadCount(Accelerator accelerator, short[] sdf, int size)
        {
            using var pipeline = new SdfMeshPipeline(accelerator);
            using var sdfBuffer = accelerator.Allocate1D(sdf);
            using var mesh = await pipeline.MeshSdfAsync(sdfBuffer, size, 0, 0, 0, 1f);
            return mesh.QuadCount;
        }

        /// <summary>
        /// Horizontal planar SDF: y &lt; splitY -> air (-500), y &gt;= splitY -> solid (+500).
        /// </summary>
        private static short[] PlanarSdf(int size, int splitY)
        {
            var sdf = new short[size * size * size];
            for (int y = 0; y < size; y++)
            for (int z = 0; z < size; z++)
            for (int x = 0; x < size; x++)
                sdf[x + z * size + y * size * size] = (y >= splitY) ? (short)500 : (short)-500;
            return sdf;
        }
    }
}
