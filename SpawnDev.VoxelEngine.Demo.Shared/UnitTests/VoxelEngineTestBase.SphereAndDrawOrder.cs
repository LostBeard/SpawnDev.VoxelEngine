using System.Numerics;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Physics;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Sphere query + draw order tests.
    // VoxelSphereQuery: area-of-effect, explosion damage, proximity detection.
    // DrawOrderKernels: front-to-back (opaque) and back-to-front (transparent) section sorting.
    public abstract partial class VoxelEngineTestBase
    {
        // ===== VoxelSphereQuery tests =====

        /// <summary>
        /// Test 1: Sphere centered exactly on a single block - returns that one block.
        /// Radius is small enough to exclude all neighbors.
        /// </summary>
        [TestMethod]
        public void SphereQuery_SingleBlock_HitsOneTest()
        {
            // Single stone block at (8, 8, 8). Block center is (8.5, 8.5, 8.5).
            var section = CreateTestSection((8, 8, 8, 42));

            // Sphere at block center with radius 0.4 - just inside the block, excludes neighbors
            var hits = VoxelSphereQuery.Query(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 0.4f);

            if (hits.Count != 1)
                throw new Exception($"SphereQuery_SingleBlock: expected 1 hit, got {hits.Count}");
            if (hits[0].X != 8 || hits[0].Y != 8 || hits[0].Z != 8)
                throw new Exception($"SphereQuery_SingleBlock: hit at ({hits[0].X},{hits[0].Y},{hits[0].Z}), expected (8,8,8)");
            if (hits[0].BlockType != 42)
                throw new Exception($"SphereQuery_SingleBlock: block type {hits[0].BlockType}, expected 42");
            if (hits[0].Distance > 0.001f)
                throw new Exception($"SphereQuery_SingleBlock: distance should be ~0 (centered), got {hits[0].Distance}");
        }

        /// <summary>
        /// Test 2: Sphere with radius 2 centered on a solid 3x3x3 cube.
        /// Verifies correct count - the sphere must contain all 27 blocks since the
        /// farthest block center is at distance sqrt(1^2+1^2+1^2) = 1.73, within radius 2.
        /// </summary>
        [TestMethod]
        public void SphereQuery_3x3Cube_CorrectCountTest()
        {
            // 3x3x3 cube of stone at (7..9, 7..9, 7..9)
            var blocks = new (int x, int y, int z, int blockType)[27];
            int idx = 0;
            for (int y = 7; y <= 9; y++)
                for (int z = 7; z <= 9; z++)
                    for (int x = 7; x <= 9; x++)
                        blocks[idx++] = (x, y, z, 1);
            var section = CreateTestSection(blocks);

            // Sphere center at cube center (8.5, 8.5, 8.5), radius 2.0
            // Corner block center is at (7.5, 7.5, 7.5), distance = sqrt(3) = 1.732 < 2.0
            var hits = VoxelSphereQuery.Query(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 2.0f);

            if (hits.Count != 27)
                throw new Exception($"SphereQuery_3x3Cube: expected 27 hits for 3x3x3 cube within r=2, got {hits.Count}");

            // Count method should agree
            int count = VoxelSphereQuery.Count(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 2.0f);
            if (count != 27)
                throw new Exception($"SphereQuery_3x3Cube: Count() returned {count}, expected 27");

            // Results should be sorted by distance - center block first, corners last
            if (hits[0].Distance > hits[hits.Count - 1].Distance)
                throw new Exception("SphereQuery_3x3Cube: results should be sorted nearest-first");

            // The center block (8,8,8) should be first (distance 0)
            if (hits[0].X != 8 || hits[0].Y != 8 || hits[0].Z != 8)
                throw new Exception($"SphereQuery_3x3Cube: nearest hit should be center block (8,8,8), got ({hits[0].X},{hits[0].Y},{hits[0].Z})");
        }

        /// <summary>
        /// Test 3: Sphere excludes air blocks - only non-air blocks are returned.
        /// Mix of solid and air within the sphere radius.
        /// </summary>
        [TestMethod]
        public void SphereQuery_ExcludesAir_OnlySolidTest()
        {
            // Place 3 blocks with gaps (air between them)
            var section = CreateTestSection(
                (8, 8, 8, 1),   // stone at center
                (9, 8, 8, 3),   // dirt one block east
                (8, 8, 10, 5)); // grass two blocks north (just barely in range)

            // Sphere centered at (8.5, 8.5, 8.5), radius 3.0
            // (9.5 - 8.5)^2 = 1.0 for the dirt - well within range
            // (10.5 - 8.5)^2 = 4.0 for the grass z, total dist = sqrt(4) = 2.0 - within range
            // Air blocks at (8,8,9) etc. should NOT appear
            var hits = VoxelSphereQuery.Query(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 3.0f);

            if (hits.Count != 3)
                throw new Exception($"SphereQuery_ExcludesAir: expected 3 solid blocks, got {hits.Count}");

            // Verify no air blocks leaked through
            foreach (var hit in hits)
            {
                if (hit.BlockType == 0)
                    throw new Exception($"SphereQuery_ExcludesAir: air block returned at ({hit.X},{hit.Y},{hit.Z})");
            }
        }

        /// <summary>
        /// Test 4: Sphere with block filter - only returns blocks matching the filter.
        /// Tests type-specific queries (e.g., "find all stone within explosion radius").
        /// </summary>
        [TestMethod]
        public void SphereQuery_FilterByType_SelectiveTest()
        {
            // Mixed block types all near center
            var section = CreateTestSection(
                (8, 8, 8, 1),   // stone
                (9, 8, 8, 2),   // glass
                (8, 9, 8, 1),   // stone
                (8, 8, 9, 3),   // dirt
                (7, 8, 8, 1));  // stone

            // Query with filter: only stone (type 1)
            var stoneHits = VoxelSphereQuery.Query(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 3.0f,
                blockFilter: packed => PackedBlock.GetType(packed) == 1);

            if (stoneHits.Count != 3)
                throw new Exception($"SphereQuery_FilterByType: expected 3 stone blocks, got {stoneHits.Count}");

            foreach (var hit in stoneHits)
            {
                if (hit.BlockType != 1)
                    throw new Exception($"SphereQuery_FilterByType: non-stone block type {hit.BlockType} passed through filter");
            }

            // Count with same filter should agree
            int stoneCount = VoxelSphereQuery.Count(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 3.0f,
                blockFilter: packed => PackedBlock.GetType(packed) == 1);
            if (stoneCount != 3)
                throw new Exception($"SphereQuery_FilterByType: Count() returned {stoneCount}, expected 3");

            // Query for glass (type 2) - should get exactly 1
            var glassHits = VoxelSphereQuery.Query(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 3.0f,
                blockFilter: packed => PackedBlock.GetType(packed) == 2);
            if (glassHits.Count != 1)
                throw new Exception($"SphereQuery_FilterByType: expected 1 glass block, got {glassHits.Count}");
        }

        /// <summary>
        /// Test 5: Sphere query on empty section - returns 0 hits.
        /// </summary>
        [TestMethod]
        public void SphereQuery_EmptySection_ZeroHitsTest()
        {
            var section = new int[16 * 16 * 16]; // all air

            var hits = VoxelSphereQuery.Query(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 10.0f); // large radius, still 0 hits

            if (hits.Count != 0)
                throw new Exception($"SphereQuery_EmptySection: expected 0 hits in empty section, got {hits.Count}");

            int count = VoxelSphereQuery.Count(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), 10.0f);
            if (count != 0)
                throw new Exception($"SphereQuery_EmptySection: Count() returned {count}, expected 0");
        }

        // ===== DrawOrderKernels tests =====

        /// <summary>
        /// Test 6: Front-to-back sort - nearest section to camera is first in sorted order.
        /// Used for opaque rendering (early-Z rejection).
        /// </summary>
        [TestMethod]
        public void DrawOrder_FrontToBack_NearestFirstTest()
        {
            var entries = new[]
            {
                new SectionEntry { Coord = new SectionCoord(10, 0, 10) }, // far
                new SectionEntry { Coord = new SectionCoord(1, 0, 1) },   // near
                new SectionEntry { Coord = new SectionCoord(5, 0, 5) },   // mid
                new SectionEntry { Coord = new SectionCoord(3, 2, 3) },   // near-mid (elevated)
                new SectionEntry { Coord = new SectionCoord(8, 0, 2) },   // mid-far
            };

            float voxelSize = 1.0f;
            int sectionSize = 16;
            float baseY = 0f;
            // Camera near origin
            float camX = 0f, camY = 0f, camZ = 0f;

            var sorted = DrawOrderKernels.SortByDistance(entries, camX, camY, camZ,
                voxelSize, sectionSize, baseY, backToFront: false);

            if (sorted.Length != entries.Length)
                throw new Exception($"DrawOrder_FrontToBack: expected {entries.Length} indices, got {sorted.Length}");

            // Verify strictly ascending distances
            for (int i = 1; i < sorted.Length; i++)
            {
                float prevDist = ComputeSectionDistSq(entries[sorted[i - 1]].Coord, camX, camY, camZ, voxelSize, sectionSize, baseY);
                float currDist = ComputeSectionDistSq(entries[sorted[i]].Coord, camX, camY, camZ, voxelSize, sectionSize, baseY);
                if (currDist < prevDist)
                    throw new Exception($"DrawOrder_FrontToBack: index {i} (dist={currDist:F2}) is closer than index {i - 1} (dist={prevDist:F2}) - not front-to-back");
            }

            // The nearest section (1,0,1) should be first
            if (sorted[0] != 1)
                throw new Exception($"DrawOrder_FrontToBack: nearest section should be index 1 (coord 1,0,1), got index {sorted[0]}");
        }

        /// <summary>
        /// Test 7: Back-to-front sort - farthest section from camera is first.
        /// Used for transparent rendering (correct alpha blending).
        /// </summary>
        [TestMethod]
        public void DrawOrder_BackToFront_FarthestFirstTest()
        {
            var entries = new[]
            {
                new SectionEntry { Coord = new SectionCoord(10, 0, 10) }, // far
                new SectionEntry { Coord = new SectionCoord(1, 0, 1) },   // near
                new SectionEntry { Coord = new SectionCoord(5, 0, 5) },   // mid
            };

            float voxelSize = 1.0f;
            int sectionSize = 16;
            float baseY = 0f;
            float camX = 0f, camY = 0f, camZ = 0f;

            var sorted = DrawOrderKernels.SortByDistance(entries, camX, camY, camZ,
                voxelSize, sectionSize, baseY, backToFront: true);

            // Verify strictly descending distances
            for (int i = 1; i < sorted.Length; i++)
            {
                float prevDist = ComputeSectionDistSq(entries[sorted[i - 1]].Coord, camX, camY, camZ, voxelSize, sectionSize, baseY);
                float currDist = ComputeSectionDistSq(entries[sorted[i]].Coord, camX, camY, camZ, voxelSize, sectionSize, baseY);
                if (currDist > prevDist)
                    throw new Exception($"DrawOrder_BackToFront: index {i} (dist={currDist:F2}) is farther than index {i - 1} (dist={prevDist:F2}) - not back-to-front");
            }

            // The farthest section (10,0,10) should be first
            if (sorted[0] != 0)
                throw new Exception($"DrawOrder_BackToFront: farthest section should be index 0 (coord 10,0,10), got index {sorted[0]}");

            // The nearest section (1,0,1) should be last
            if (sorted[sorted.Length - 1] != 1)
                throw new Exception($"DrawOrder_BackToFront: nearest section should be last, index 1 (coord 1,0,1), got index {sorted[sorted.Length - 1]}");
        }

        /// <summary>
        /// Test 8: Camera at origin - verify distances are computed correctly.
        /// Hand-compute expected distances and verify the CPU reference matches.
        /// </summary>
        [TestMethod]
        public void DrawOrder_CameraAtOrigin_CorrectDistancesTest()
        {
            float voxelSize = 1.0f;
            int sectionSize = 16;
            float baseY = 0f;
            float camX = 0f, camY = 0f, camZ = 0f;

            var entries = new[]
            {
                new SectionEntry { Coord = new SectionCoord(0, 0, 0) },  // center at (8, 8, 8)
                new SectionEntry { Coord = new SectionCoord(1, 0, 0) },  // center at (24, 8, 8)
                new SectionEntry { Coord = new SectionCoord(0, 1, 0) },  // center at (8, 24, 8)
                new SectionEntry { Coord = new SectionCoord(0, 0, 1) },  // center at (8, 8, 24)
            };

            // Hand-computed: section center = coord * sectionSize * voxelSize + halfSection
            // halfSection = 16 * 1.0 * 0.5 = 8
            // Entry 0: center (8, 8, 8), distSq = 64 + 64 + 64 = 192
            // Entry 1: center (24, 8, 8), distSq = 576 + 64 + 64 = 704
            // Entry 2: center (8, 24, 8), distSq = 64 + 576 + 64 = 704
            // Entry 3: center (8, 8, 24), distSq = 64 + 64 + 576 = 704

            var sorted = DrawOrderKernels.SortByDistance(entries, camX, camY, camZ,
                voxelSize, sectionSize, baseY, backToFront: false);

            // Entry 0 should be nearest
            if (sorted[0] != 0)
                throw new Exception($"DrawOrder_CameraAtOrigin: entry 0 (distSq=192) should be first, got index {sorted[0]}");

            // Entries 1, 2, 3 are equidistant (distSq=704 each) - verify they're all after entry 0
            float nearestDist = ComputeSectionDistSq(entries[sorted[0]].Coord, camX, camY, camZ, voxelSize, sectionSize, baseY);
            float expectedNearest = 192f;
            if (MathF.Abs(nearestDist - expectedNearest) > 0.01f)
                throw new Exception($"DrawOrder_CameraAtOrigin: nearest distSq should be {expectedNearest}, got {nearestDist}");

            // All remaining entries should have distSq = 704
            for (int i = 1; i < sorted.Length; i++)
            {
                float dist = ComputeSectionDistSq(entries[sorted[i]].Coord, camX, camY, camZ, voxelSize, sectionSize, baseY);
                float expected = 704f;
                if (MathF.Abs(dist - expected) > 0.01f)
                    throw new Exception($"DrawOrder_CameraAtOrigin: entry {sorted[i]} distSq should be {expected}, got {dist}");
            }

            // Test with non-origin camera to verify offset works
            float camX2 = 24f, camY2 = 8f, camZ2 = 8f;
            var sorted2 = DrawOrderKernels.SortByDistance(entries, camX2, camY2, camZ2,
                voxelSize, sectionSize, baseY, backToFront: false);

            // Entry 1 center is (24, 8, 8), camera is at (24, 8, 8) - distSq = 0
            float distEntry1 = ComputeSectionDistSq(entries[1].Coord, camX2, camY2, camZ2, voxelSize, sectionSize, baseY);
            if (MathF.Abs(distEntry1) > 0.01f)
                throw new Exception($"DrawOrder_CameraAtOrigin: camera on section 1 center, distSq should be 0, got {distEntry1}");

            if (sorted2[0] != 1)
                throw new Exception($"DrawOrder_CameraAtOrigin: section 1 should be nearest when camera is at its center, got index {sorted2[0]}");
        }

        /// <summary>
        /// Helper: compute squared distance from camera to section center.
        /// Mirrors the exact math in DrawOrderKernels.SortByDistance.
        /// </summary>
        private static float ComputeSectionDistSq(SectionCoord coord,
            float camX, float camY, float camZ,
            float voxelSize, int sectionSize, float baseY)
        {
            float halfSection = sectionSize * voxelSize * 0.5f;
            float centerX = coord.Cx * sectionSize * voxelSize + halfSection;
            float centerY = baseY + coord.Sy * sectionSize * voxelSize + halfSection;
            float centerZ = coord.Cz * sectionSize * voxelSize + halfSection;
            float dx = centerX - camX;
            float dy = centerY - camY;
            float dz = centerZ - camZ;
            return dx * dx + dy * dy + dz * dz;
        }
    }
}
