using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // VoxelMeshPipeline.MeshChunkColumnAsync tests. Covers:
    //  - All-air fast path: zero kernel dispatches, zero results
    //  - Selective meshing: only sections with blocks produce results
    //  - Output equivalence vs per-section MeshSectionAsync (byte-for-byte match)
    //  - XZ neighbor padding still honored
    //  - Y padding derived from the same column (intra-chunk seam hidden)
    //  - Argument validation (neighbor length, blocks length)
    //
    // The biggest win this API provides is the all-air fast path: for terrain where
    // geometry lives in a narrow Y band (e.g. y=10-38 on a 256-high column), 13 of
    // 16 sections have no blocks and must not spin up kernels, sync fences, or
    // counter readbacks. These tests exist to prevent regression of that win and to
    // prove the produced quads match the legacy path for the sections that do mesh.
    public partial class VoxelEngineTestBase
    {
        // Helper: build a full chunk-column block array (sectionSize * sectionSize * totalHeight).
        // Caller supplies a fill predicate: (x, y, z) -> blockId (0 = air).
        private static byte[] BuildColumn(int sectionSize, int totalHeight, Func<int, int, int, byte> fill)
        {
            int xz = sectionSize * sectionSize;
            var blocks = new byte[xz * totalHeight];
            for (int y = 0; y < totalHeight; y++)
                for (int z = 0; z < sectionSize; z++)
                    for (int x = 0; x < sectionSize; x++)
                        blocks[x + z * sectionSize + y * xz] = fill(x, y, z);
            return blocks;
        }

        // Minimum-size column where totalHeight is a small multiple of sectionSize.
        // Kernels require section height <= 64, and sectionSize here is <= 16 for face-mask
        // kernels. Using ss=4 keeps upload small while still exercising the real code path.
        private const int TestSectionSize = 4;
        private const int TestSectionsPerColumn = 4; // total height = 16
        private const int TestTotalHeight = TestSectionSize * TestSectionsPerColumn;

        /// <summary>
        /// All-air column: the fast path must skip every section with zero kernel dispatches
        /// and return an empty result list. Regression guard - if this test starts taking
        /// significantly longer, the fast path has been lost.
        /// </summary>
        [TestMethod]
        public async Task MeshChunkColumn_AllAirColumn_ReturnsEmptyList() => await RunTest(async accelerator =>
        {
            var blocks = new byte[TestSectionSize * TestSectionSize * TestTotalHeight]; // all zero

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var results = await pipeline.MeshChunkColumnAsync(
                blocks, null, null, null, null,
                TestSectionSize, TestTotalHeight);

            if (results.Count != 0)
                throw new Exception($"All-air column should produce zero mesh results, got {results.Count}");
        });

        /// <summary>
        /// Column with exactly one section containing blocks (sy=1, the Y-range [4..7]).
        /// The other three sections are all air and must be skipped entirely.
        /// </summary>
        [TestMethod]
        public async Task MeshChunkColumn_OnlyMiddleSectionHasBlocks_SkipsAirSections() => await RunTest(async accelerator =>
        {
            var blocks = BuildColumn(TestSectionSize, TestTotalHeight,
                (x, y, z) => (y >= 4 && y < 8) ? (byte)1 : (byte)0);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var results = await pipeline.MeshChunkColumnAsync(
                blocks, null, null, null, null,
                TestSectionSize, TestTotalHeight);

            try
            {
                if (results.Count != 1)
                    throw new Exception($"Expected exactly 1 non-air section, got {results.Count}");
                if (results[0].sectionY != 1)
                    throw new Exception($"Expected sectionY=1, got {results[0].sectionY}");
                if (!results[0].mesh.HasMesh)
                    throw new Exception("Expected the one non-air section to produce quads");
            }
            finally
            {
                foreach (var (_, m) in results) m.QuadBuffer?.Dispose();
            }
        });

        /// <summary>
        /// Output of MeshChunkColumnAsync for a non-air section must match the output of
        /// calling MeshSectionAsync directly on the equivalent padded int[]. Proves the
        /// new API doesn't change the mesh; it only skips the empty ones.
        /// </summary>
        [TestMethod]
        public async Task MeshChunkColumn_NonAirSection_MatchesLegacyMeshSectionAsync() => await RunTest(async accelerator =>
        {
            // Section sy=2 is solid stone (block id 1), everything else air.
            var blocks = BuildColumn(TestSectionSize, TestTotalHeight,
                (x, y, z) => (y >= 8 && y < 12) ? (byte)1 : (byte)0);

            using var pipeline = new VoxelMeshPipeline(accelerator);

            // New API
            var results = await pipeline.MeshChunkColumnAsync(
                blocks, null, null, null, null,
                TestSectionSize, TestTotalHeight);

            // Legacy API: build equivalent padded int[] for sy=2.
            int paddedXZ = TestSectionSize + 2;
            var padded = new int[paddedXZ * paddedXZ * TestSectionSize];
            for (int y = 0; y < TestSectionSize; y++)
                for (int z = 1; z <= TestSectionSize; z++)
                    for (int x = 1; x <= TestSectionSize; x++)
                        padded[x + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            // Legacy path also needs intra-chunk Y pads. For sy=2:
            //   yPadMinus = blocks at y=7 (section sy=1's top layer) -> all air -> null/air slab
            //   yPadPlus  = blocks at y=12 (section sy=3's bottom layer) -> all air -> null/air slab
            var legacyResult = await pipeline.MeshSectionAsync(padded, TestSectionSize, TestSectionSize, null, null);

            try
            {
                if (results.Count != 1)
                    throw new Exception($"New API: expected 1 non-air section, got {results.Count}");

                var newResult = results[0].mesh;
                if (newResult.QuadCount != legacyResult.QuadCount)
                    throw new Exception($"QuadCount mismatch: new API {newResult.QuadCount} vs legacy {legacyResult.QuadCount}");

                // Byte-for-byte equality of the packed quad output
                var newQuads = await newResult.QuadBuffer!.CopyToHostAsync();
                var legacyQuads = await legacyResult.QuadBuffer!.CopyToHostAsync();

                // Sort both because kernel ordering is not required to be stable
                Array.Sort(newQuads, 0, newResult.QuadCount);
                Array.Sort(legacyQuads, 0, legacyResult.QuadCount);

                for (int i = 0; i < newResult.QuadCount; i++)
                {
                    if (newQuads[i] != legacyQuads[i])
                        throw new Exception($"Quad[{i}] differs: new=0x{newQuads[i]:X16} legacy=0x{legacyQuads[i]:X16}");
                }
            }
            finally
            {
                foreach (var (_, m) in results) m.QuadBuffer?.Dispose();
                legacyResult.QuadBuffer?.Dispose();
            }
        });

        /// <summary>
        /// Solid X+ neighbor column hides +X boundary faces on every section that has blocks.
        /// Only non-air sections should pay kernel cost.
        /// </summary>
        [TestMethod]
        public async Task MeshChunkColumn_SolidXPlusNeighbor_HidesPositiveXFaces() => await RunTest(async accelerator =>
        {
            // Single solid section at sy=1
            var blocks = BuildColumn(TestSectionSize, TestTotalHeight,
                (x, y, z) => (y >= 4 && y < 8) ? (byte)1 : (byte)0);

            // +X neighbor is solid stone in the SAME Y range so its -X-most column (x=0) is solid
            var nxPlus = BuildColumn(TestSectionSize, TestTotalHeight,
                (x, y, z) => (y >= 4 && y < 8) ? (byte)1 : (byte)0);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var results = await pipeline.MeshChunkColumnAsync(
                blocks, null, nxPlus, null, null,
                TestSectionSize, TestTotalHeight);

            try
            {
                if (results.Count != 1)
                    throw new Exception($"Expected 1 non-air section, got {results.Count}");

                var quads = await results[0].mesh.QuadBuffer!.CopyToHostAsync();
                int plusXFaces = 0;
                int minusXFaces = 0;
                for (int i = 0; i < results[0].mesh.QuadCount; i++)
                {
                    int face = (int)((quads[i] >> 20) & 0x7);
                    if (face == 0) plusXFaces++;
                    else if (face == 1) minusXFaces++;
                }

                if (plusXFaces != 0)
                    throw new Exception($"Expected 0 +X faces with solid +X neighbor, got {plusXFaces}");
                if (minusXFaces < 1)
                    throw new Exception($"Expected at least 1 -X face with air -X neighbor, got {minusXFaces}");
            }
            finally
            {
                foreach (var (_, m) in results) m.QuadBuffer?.Dispose();
            }
        });

        /// <summary>
        /// Two adjacent solid sections (sy=1 and sy=2) in the same column must have their
        /// shared Y boundary hidden. The library derives y-pad slabs from the column itself,
        /// so the top of sy=1 and the bottom of sy=2 must not emit quads at that seam.
        /// </summary>
        [TestMethod]
        public async Task MeshChunkColumn_AdjacentSolidSections_HidesSharedYSeam() => await RunTest(async accelerator =>
        {
            // sy=1 solid (y=4..7), sy=2 solid (y=8..11). Shared seam at y=7<->8.
            var blocks = BuildColumn(TestSectionSize, TestTotalHeight,
                (x, y, z) => (y >= 4 && y < 12) ? (byte)1 : (byte)0);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var results = await pipeline.MeshChunkColumnAsync(
                blocks, null, null, null, null,
                TestSectionSize, TestTotalHeight);

            try
            {
                if (results.Count != 2)
                    throw new Exception($"Expected 2 non-air sections, got {results.Count}");

                // sy=1: top face (+Y) must be hidden by sy=2. bottom face (-Y) visible (sy=0 is air).
                var sy1Quads = await results[0].mesh.QuadBuffer!.CopyToHostAsync();
                int sy1PlusY = 0, sy1MinusY = 0;
                for (int i = 0; i < results[0].mesh.QuadCount; i++)
                {
                    int face = (int)((sy1Quads[i] >> 20) & 0x7);
                    if (face == 4) sy1PlusY++;
                    else if (face == 5) sy1MinusY++;
                }
                if (sy1PlusY != 0)
                    throw new Exception($"sy=1 should have 0 +Y faces (sy=2 solid above), got {sy1PlusY}");
                if (sy1MinusY < 1)
                    throw new Exception($"sy=1 should have at least 1 -Y face (sy=0 is air), got {sy1MinusY}");

                // sy=2: bottom face (-Y) must be hidden by sy=1. top face (+Y) visible (sy=3 is air).
                var sy2Quads = await results[1].mesh.QuadBuffer!.CopyToHostAsync();
                int sy2PlusY = 0, sy2MinusY = 0;
                for (int i = 0; i < results[1].mesh.QuadCount; i++)
                {
                    int face = (int)((sy2Quads[i] >> 20) & 0x7);
                    if (face == 4) sy2PlusY++;
                    else if (face == 5) sy2MinusY++;
                }
                if (sy2MinusY != 0)
                    throw new Exception($"sy=2 should have 0 -Y faces (sy=1 solid below), got {sy2MinusY}");
                if (sy2PlusY < 1)
                    throw new Exception($"sy=2 should have at least 1 +Y face (sy=3 is air), got {sy2PlusY}");
            }
            finally
            {
                foreach (var (_, m) in results) m.QuadBuffer?.Dispose();
            }
        });

        /// <summary>
        /// Argument validation: wrong-length neighbor blocks array throws ArgumentException.
        /// </summary>
        [TestMethod]
        public async Task MeshChunkColumn_WrongNeighborLength_Throws() => await RunTest(async accelerator =>
        {
            var blocks = new byte[TestSectionSize * TestSectionSize * TestTotalHeight];
            var badNeighbor = new byte[4]; // wrong size

            using var pipeline = new VoxelMeshPipeline(accelerator);
            try
            {
                await pipeline.MeshChunkColumnAsync(
                    blocks, badNeighbor, null, null, null,
                    TestSectionSize, TestTotalHeight);
                throw new Exception("Expected ArgumentException for wrong-size neighbor");
            }
            catch (ArgumentException) { /* expected */ }
        });

        /// <summary>
        /// Argument validation: wrong-length blocks array throws ArgumentException.
        /// </summary>
        [TestMethod]
        public async Task MeshChunkColumn_WrongBlocksLength_Throws() => await RunTest(async accelerator =>
        {
            var badBlocks = new byte[4]; // wrong size

            using var pipeline = new VoxelMeshPipeline(accelerator);
            try
            {
                await pipeline.MeshChunkColumnAsync(
                    badBlocks, null, null, null, null,
                    TestSectionSize, TestTotalHeight);
                throw new Exception("Expected ArgumentException for wrong-size blocks");
            }
            catch (ArgumentException) { /* expected */ }
        });

        /// <summary>
        /// totalHeight must be a positive multiple of sectionSize.
        /// </summary>
        [TestMethod]
        public async Task MeshChunkColumn_BadTotalHeight_Throws() => await RunTest(async accelerator =>
        {
            // totalHeight 17 is not a multiple of sectionSize 4
            var blocks = new byte[TestSectionSize * TestSectionSize * 17];

            using var pipeline = new VoxelMeshPipeline(accelerator);
            try
            {
                await pipeline.MeshChunkColumnAsync(
                    blocks, null, null, null, null,
                    TestSectionSize, 17);
                throw new Exception("Expected ArgumentOutOfRangeException for non-multiple totalHeight");
            }
            catch (ArgumentOutOfRangeException) { /* expected */ }
        });
    }
}
