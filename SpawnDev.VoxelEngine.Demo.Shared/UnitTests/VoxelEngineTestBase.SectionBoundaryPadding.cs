using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Section boundary padding tests: verify MeshSectionAsync correctly hides
    // boundary faces when neighbor block data is provided in the 1-block XZ border.
    //
    // The meshing pipeline supports XZ neighbor padding (the +2 size on X and Z).
    // These tests prove that a boundary face adjacent to a solid neighbor produces
    // zero quads, which is the fix for Lost Spawns' see-through chunk boundaries.
    public partial class VoxelEngineTestBase
    {
        /// <summary>
        /// Air-filled XZ neighbor padding: every boundary face of a solid section is visible.
        /// This matches MeshSectionUnpaddedAsync behavior and reproduces the see-through bug
        /// that Lost Spawns exhibits at chunk edges when neighbor chunk data is not provided.
        /// </summary>
        [TestMethod]
        public async Task SectionBoundaryPadding_AirNeighbors_ShowsAllFaces() => await RunTest(async accelerator =>
        {
            int ss = 4;
            int height = 4;
            int paddedXZ = ss + 2;
            var padded = new int[paddedXZ * paddedXZ * height];

            // Solid interior, air XZ border (the padding cells at x=0, x=ss+1, z=0, z=ss+1 stay 0)
            for (int y = 0; y < height; y++)
                for (int z = 1; z <= ss; z++)
                    for (int x = 1; x <= ss; x++)
                        padded[x + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var result = await pipeline.MeshSectionAsync(padded, ss, height);
            try
            {
                if (!result.HasMesh)
                    throw new Exception("Expected visible quads for solid section with air neighbors, got empty mesh");

                // Expected: +X, -X, +Z, -Z each contribute one ss*height rect (4 face-groups, greedy merged)
                // plus +Y and -Y each contribute one ss*ss rect (2 more face-groups)
                // = 6 greedy-merged face-groups. Total visible unit faces = 6 * ss * (ss or height).
                // 4 side faces: ss*height = 16 units each = 64, 2 top/bottom: ss*ss = 16 each = 32. Total 96 unit faces.
                // We don't check exact quad count (greedy merging can split), only that > 0 quads exist.
                if (result.QuadCount < 6)
                    throw new Exception($"Expected at least 6 greedy-merged quads (one per face direction), got {result.QuadCount}");
            }
            finally { result.QuadBuffer?.Dispose(); }
        });

        /// <summary>
        /// Solid XZ neighbor padding: all four ±X/±Z boundary faces are hidden by neighbor blocks.
        /// Only the ±Y faces (top and bottom) remain visible.
        /// This is the target behavior for the Lost Spawns fix - when neighbor chunk data is
        /// available, boundary faces should not be rendered.
        /// </summary>
        [TestMethod]
        public async Task SectionBoundaryPadding_SolidXZNeighbors_HidesXZFaces() => await RunTest(async accelerator =>
        {
            int ss = 4;
            int height = 4;
            int paddedXZ = ss + 2;
            var padded = new int[paddedXZ * paddedXZ * height];

            // Solid everywhere including the XZ padding border.
            // This simulates all four XZ neighbor chunks being fully solid stone.
            for (int y = 0; y < height; y++)
                for (int z = 0; z < paddedXZ; z++)
                    for (int x = 0; x < paddedXZ; x++)
                        padded[x + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var result = await pipeline.MeshSectionAsync(padded, ss, height);
            try
            {
                // Expect only ±Y faces: 2 greedy-merged quads total.
                // The ±X and ±Z faces should produce zero quads because solid neighbors hide them.
                if (!result.HasMesh)
                    throw new Exception("Expected ±Y quads for solid section with solid XZ neighbors, got empty mesh");

                // Read back to verify: should be exactly 2 quads (one top, one bottom, both greedy-merged to ss*ss).
                var quadData = await result.QuadBuffer!.CopyToHostAsync();
                int quadCount = result.QuadCount;
                int yFaces = 0;
                int xzFaces = 0;
                for (int i = 0; i < quadCount; i++)
                {
                    // PackedQuad face direction at bits 20-22 (0=+X, 1=-X, 2=+Z, 3=-Z, 4=+Y, 5=-Y)
                    int face = (int)((quadData[i] >> 20) & 0x7);
                    if (face == 4 || face == 5) yFaces++;
                    else xzFaces++;
                }

                if (xzFaces != 0)
                    throw new Exception($"Expected 0 XZ boundary faces with solid neighbors, got {xzFaces} XZ quads (total={quadCount})");
                if (yFaces < 2)
                    throw new Exception($"Expected at least 2 Y faces (top+bottom), got {yFaces}");
            }
            finally { result.QuadBuffer?.Dispose(); }
        });

        /// <summary>
        /// Asymmetric XZ padding: solid +X neighbor only, the other three sides air.
        /// Only the +X boundary face should be hidden; -X, +Z, -Z should remain visible.
        /// Proves per-direction padding works independently.
        /// </summary>
        [TestMethod]
        public async Task SectionBoundaryPadding_OnlyPositiveXSolid_HidesOnlyPositiveXFace() => await RunTest(async accelerator =>
        {
            int ss = 4;
            int height = 4;
            int paddedXZ = ss + 2;
            var padded = new int[paddedXZ * paddedXZ * height];

            // Solid interior
            for (int y = 0; y < height; y++)
                for (int z = 1; z <= ss; z++)
                    for (int x = 1; x <= ss; x++)
                        padded[x + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            // Solid +X padding only (x = paddedXZ - 1 = 5 = ss+1)
            for (int y = 0; y < height; y++)
                for (int z = 1; z <= ss; z++)
                    padded[(paddedXZ - 1) + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var result = await pipeline.MeshSectionAsync(padded, ss, height);
            try
            {
                if (!result.HasMesh)
                    throw new Exception("Expected visible quads for partially-padded section");

                var quadData = await result.QuadBuffer!.CopyToHostAsync();
                int quadCount = result.QuadCount;
                int plusXFaces = 0;
                int minusXFaces = 0;
                for (int i = 0; i < quadCount; i++)
                {
                    int face = (int)((quadData[i] >> 20) & 0x7);
                    if (face == 0) plusXFaces++;
                    else if (face == 1) minusXFaces++;
                }

                if (plusXFaces != 0)
                    throw new Exception($"Expected 0 +X faces with solid +X neighbor, got {plusXFaces}");
                if (minusXFaces < 1)
                    throw new Exception($"Expected at least 1 -X face (no -X neighbor), got {minusXFaces}");
            }
            finally { result.QuadBuffer?.Dispose(); }
        });

        // ========================================================================
        // Y-padding tests (Task #11 Option A): verify MeshSectionAsync honors
        // the optional yPadMinusSlab and yPadPlusSlab parameters. Each slab is a
        // (sectionSize+2)^2 int array in PackedBlock format representing the
        // adjacent Y-neighbor section's boundary layer. Solid slab bits hide ±Y
        // boundary faces; null/air leaves them visible.
        //
        // Fixes the intra-chunk Y see-through artifact at Lost Spawns section
        // boundaries (y=16, 32, ... 240). Format-preserving: same PackedQuad
        // layout, same face mask bit positions — only the kernel inputs grow.
        // ========================================================================

        /// <summary>
        /// Null Y pads = air neighbors. Should match the legacy MeshSectionAsync behavior:
        /// both ±Y boundary faces visible (a solid cube shows all 6 face directions).
        /// Regression guard - confirms the new kernel path defaults correctly.
        /// </summary>
        [TestMethod]
        public async Task SectionBoundaryPadding_NullYPads_KeepsBothYFacesVisible() => await RunTest(async accelerator =>
        {
            int ss = 4;
            int height = 4;
            int paddedXZ = ss + 2;
            var padded = new int[paddedXZ * paddedXZ * height];

            for (int y = 0; y < height; y++)
                for (int z = 1; z <= ss; z++)
                    for (int x = 1; x <= ss; x++)
                        padded[x + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var result = await pipeline.MeshSectionAsync(padded, ss, height,
                yPadMinusSlab: null, yPadPlusSlab: null);
            try
            {
                if (!result.HasMesh)
                    throw new Exception("Expected visible quads with air Y pads");

                var quadData = await result.QuadBuffer!.CopyToHostAsync();
                int plusY = 0, minusY = 0;
                for (int i = 0; i < result.QuadCount; i++)
                {
                    int face = (int)((quadData[i] >> 20) & 0x7);
                    if (face == 4) plusY++;
                    else if (face == 5) minusY++;
                }

                if (plusY < 1)
                    throw new Exception($"Expected at least 1 +Y face with null yPadPlus, got {plusY}");
                if (minusY < 1)
                    throw new Exception($"Expected at least 1 -Y face with null yPadMinus, got {minusY}");
            }
            finally { result.QuadBuffer?.Dispose(); }
        });

        /// <summary>
        /// Solid ±Y pads = stone above and below. Both ±Y boundary faces must be
        /// hidden, leaving only XZ faces (at least 4 greedy quads for ±X/±Z).
        /// This is the target behavior that eliminates the intra-chunk Y
        /// see-through artifact in Lost Spawns.
        /// </summary>
        [TestMethod]
        public async Task SectionBoundaryPadding_SolidYPads_HidesBothYFaces() => await RunTest(async accelerator =>
        {
            int ss = 4;
            int height = 4;
            int paddedXZ = ss + 2;
            int slabLen = paddedXZ * paddedXZ;
            var padded = new int[slabLen * height];

            for (int y = 0; y < height; y++)
                for (int z = 1; z <= ss; z++)
                    for (int x = 1; x <= ss; x++)
                        padded[x + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            // Solid slabs for both ±Y neighbors. Interior (x,z) in [1..ss] only
            // matters; the XZ padding border of the slab is never read.
            var yPadMinus = new int[slabLen];
            var yPadPlus = new int[slabLen];
            for (int z = 1; z <= ss; z++)
                for (int x = 1; x <= ss; x++)
                {
                    yPadMinus[x + z * paddedXZ] = PackedBlock.Pack(1);
                    yPadPlus[x + z * paddedXZ] = PackedBlock.Pack(1);
                }

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var result = await pipeline.MeshSectionAsync(padded, ss, height, yPadMinus, yPadPlus);
            try
            {
                if (!result.HasMesh)
                    throw new Exception("Expected XZ quads with solid Y pads, got empty mesh");

                var quadData = await result.QuadBuffer!.CopyToHostAsync();
                int plusY = 0, minusY = 0, xzFaces = 0;
                for (int i = 0; i < result.QuadCount; i++)
                {
                    int face = (int)((quadData[i] >> 20) & 0x7);
                    if (face == 4) plusY++;
                    else if (face == 5) minusY++;
                    else xzFaces++;
                }

                if (plusY != 0)
                    throw new Exception($"Expected 0 +Y faces with solid yPadPlus, got {plusY}");
                if (minusY != 0)
                    throw new Exception($"Expected 0 -Y faces with solid yPadMinus, got {minusY}");
                if (xzFaces < 4)
                    throw new Exception($"Expected at least 4 XZ faces (±X/±Z greedy-merged), got {xzFaces}");
            }
            finally { result.QuadBuffer?.Dispose(); }
        });

        /// <summary>
        /// Asymmetric Y pads: solid +Y neighbor only, air -Y. Only the top face
        /// should be hidden. The bottom face (y=0) must still render because
        /// its -Y neighbor is air. Proves per-direction Y padding is independent.
        /// </summary>
        [TestMethod]
        public async Task SectionBoundaryPadding_OnlyPositiveYSolid_HidesOnlyTopFace() => await RunTest(async accelerator =>
        {
            int ss = 4;
            int height = 4;
            int paddedXZ = ss + 2;
            int slabLen = paddedXZ * paddedXZ;
            var padded = new int[slabLen * height];

            for (int y = 0; y < height; y++)
                for (int z = 1; z <= ss; z++)
                    for (int x = 1; x <= ss; x++)
                        padded[x + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            var yPadPlus = new int[slabLen];
            for (int z = 1; z <= ss; z++)
                for (int x = 1; x <= ss; x++)
                    yPadPlus[x + z * paddedXZ] = PackedBlock.Pack(1);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var result = await pipeline.MeshSectionAsync(padded, ss, height,
                yPadMinusSlab: null, yPadPlusSlab: yPadPlus);
            try
            {
                if (!result.HasMesh)
                    throw new Exception("Expected visible quads for asymmetric Y padding");

                var quadData = await result.QuadBuffer!.CopyToHostAsync();
                int plusY = 0, minusY = 0;
                for (int i = 0; i < result.QuadCount; i++)
                {
                    int face = (int)((quadData[i] >> 20) & 0x7);
                    if (face == 4) plusY++;
                    else if (face == 5) minusY++;
                }

                if (plusY != 0)
                    throw new Exception($"Expected 0 +Y faces with solid yPadPlus, got {plusY}");
                if (minusY < 1)
                    throw new Exception($"Expected at least 1 -Y face with air yPadMinus, got {minusY}");
            }
            finally { result.QuadBuffer?.Dispose(); }
        });

        /// <summary>
        /// Partial +Y slab: only half the interior columns have a solid +Y neighbor.
        /// Exactly those (x,z) positions should lose their +Y boundary face while
        /// the other half keeps theirs. Proves Y padding honors per-XZ granularity.
        /// </summary>
        [TestMethod]
        public async Task SectionBoundaryPadding_PartialPositiveYSlab_HidesOnlyCoveredColumns() => await RunTest(async accelerator =>
        {
            int ss = 4;
            int height = 4;
            int paddedXZ = ss + 2;
            int slabLen = paddedXZ * paddedXZ;
            var padded = new int[slabLen * height];

            for (int y = 0; y < height; y++)
                for (int z = 1; z <= ss; z++)
                    for (int x = 1; x <= ss; x++)
                        padded[x + z * paddedXZ + y * paddedXZ * paddedXZ] = PackedBlock.Pack(1);

            // +Y slab covers only the lower-z half of the interior (z in [1..ss/2]).
            var yPadPlus = new int[slabLen];
            int coveredZMax = ss / 2; // = 2
            for (int z = 1; z <= coveredZMax; z++)
                for (int x = 1; x <= ss; x++)
                    yPadPlus[x + z * paddedXZ] = PackedBlock.Pack(1);

            using var pipeline = new VoxelMeshPipeline(accelerator);
            var result = await pipeline.MeshSectionAsync(padded, ss, height,
                yPadMinusSlab: null, yPadPlusSlab: yPadPlus);
            try
            {
                if (!result.HasMesh)
                    throw new Exception("Expected visible quads for partial +Y padding");

                var quadData = await result.QuadBuffer!.CopyToHostAsync();

                // Count +Y unit faces in covered vs uncovered halves.
                int plusYUnitsCovered = 0;
                int plusYUnitsUncovered = 0;
                for (int i = 0; i < result.QuadCount; i++)
                {
                    long q = quadData[i];
                    int face = (int)((q >> 20) & 0x7);
                    if (face != 4) continue;
                    int qx = (int)(q & 0xF);
                    int qz = (int)((q >> 8) & 0xF);
                    int qw = (int)((q >> 12) & 0xF) + 1; // X extent
                    int qh = (int)((q >> 16) & 0xF) + 1; // Z extent
                    for (int dz = 0; dz < qh; dz++)
                        for (int dx = 0; dx < qw; dx++)
                        {
                            int localZ = qz + dz;     // 0..ss-1 section-local Z
                            // covered range in section-local Z: [0 .. coveredZMax-1] (since padded z in [1..coveredZMax] -> local z in [0..coveredZMax-1])
                            if (localZ < coveredZMax) plusYUnitsCovered++;
                            else plusYUnitsUncovered++;
                        }
                }

                if (plusYUnitsCovered != 0)
                    throw new Exception($"Expected 0 +Y unit faces in covered region, got {plusYUnitsCovered}");
                // Uncovered half: (ss - coveredZMax) rows * ss cols = 2*4 = 8 unit faces expected
                int expectedUncovered = (ss - coveredZMax) * ss;
                if (plusYUnitsUncovered != expectedUncovered)
                    throw new Exception($"Expected {expectedUncovered} +Y unit faces in uncovered region, got {plusYUnitsUncovered}");
            }
            finally { result.QuadBuffer?.Dispose(); }
        });

        /// <summary>
        /// Argument validation: wrong-length yPad slab throws ArgumentException.
        /// </summary>
        [TestMethod]
        public async Task SectionBoundaryPadding_YPadWrongLength_Throws() => await RunTest(async accelerator =>
        {
            int ss = 4;
            int height = 4;
            int paddedXZ = ss + 2;
            var padded = new int[paddedXZ * paddedXZ * height];
            var badSlab = new int[4]; // clearly wrong size

            using var pipeline = new VoxelMeshPipeline(accelerator);
            try
            {
                await pipeline.MeshSectionAsync(padded, ss, height, yPadMinusSlab: badSlab);
                throw new Exception("Expected ArgumentException for wrong-length slab");
            }
            catch (ArgumentException) { /* expected */ }
        });
    }
}
