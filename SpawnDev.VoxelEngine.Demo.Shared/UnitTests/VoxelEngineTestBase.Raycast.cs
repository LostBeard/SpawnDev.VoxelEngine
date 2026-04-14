using System.Numerics;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Physics;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Raycast tests: DDA voxel traversal with CPU reference comparison.
    // Tests the exact algorithm that both AubsCraft and Lost Spawns use for
    // block targeting, line of sight, projectile hits, and debug X-ray.
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>Create a 16x16x16 section filled with air, with specific blocks placed.</summary>
        private static int[] CreateTestSection(params (int x, int y, int z, int blockType)[] blocks)
        {
            int size = 16;
            var data = new int[size * size * size];
            foreach (var (x, y, z, bt) in blocks)
            {
                data[x + z * size + y * size * size] = PackedBlock.Pack(bt);
            }
            return data;
        }

        /// <summary>
        /// Test 1: Ray hits a single block. Verify correct hit position, face, and block type.
        /// </summary>
        [TestMethod]
        public void Raycast_SingleBlock_HitsCorrectFaceTest()
        {
            // Place a stone block at (8, 8, 8) in an empty section
            var section = CreateTestSection((8, 8, 8, 1)); // type 1 = stone

            // Ray from (8, 8, 0.5) looking in +Z direction - should hit -Z face of block at z=8
            var hit = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8.5f, 8.5f, 0.5f), Vector3.UnitZ, 20f);

            if (!hit.DidHit)
                throw new Exception("Raycast_SingleBlock: expected hit, got miss");
            if (hit.BlockX != 8 || hit.BlockY != 8 || hit.BlockZ != 8)
                throw new Exception($"Raycast_SingleBlock: wrong block ({hit.BlockX},{hit.BlockY},{hit.BlockZ}), expected (8,8,8)");
            if (hit.HitFace != VoxelMeshConstants.FaceNegZ)
                throw new Exception($"Raycast_SingleBlock: wrong face {hit.HitFace}, expected {VoxelMeshConstants.FaceNegZ} (-Z)");
            if (hit.BlockType != 1)
                throw new Exception($"Raycast_SingleBlock: wrong type {hit.BlockType}, expected 1");
            if (hit.Distance < 7f || hit.Distance > 8f)
                throw new Exception($"Raycast_SingleBlock: distance {hit.Distance} out of expected range 7-8");

            // Verify adjacent block is correct (one step in the face normal direction)
            if (hit.AdjacentZ != 7)
                throw new Exception($"Raycast_SingleBlock: adjacent Z={hit.AdjacentZ}, expected 7");
        }

        /// <summary>
        /// Test 2: Ray through empty space - no hit within max distance.
        /// </summary>
        [TestMethod]
        public void Raycast_EmptySection_NoHitTest()
        {
            var section = new int[16 * 16 * 16]; // all air

            var hit = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8f, 8f, 0.5f), Vector3.UnitZ, 20f);

            if (hit.DidHit)
                throw new Exception("Raycast_EmptySection: expected no hit, got hit");
        }

        /// <summary>
        /// Test 3: Ray through transparent block passes through, hits solid behind.
        /// Uses a block filter that treats type 2 as transparent.
        /// </summary>
        [TestMethod]
        public void Raycast_TransparentBlock_PassesThroughTest()
        {
            // Glass (type 2) at z=4, stone (type 1) at z=8
            var section = CreateTestSection(
                (8, 8, 4, 2),  // glass
                (8, 8, 8, 1)); // stone

            // Without filter: hits glass first
            var hitNoFilter = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8.5f, 8.5f, 0.5f), Vector3.UnitZ, 20f);

            if (!hitNoFilter.DidHit || hitNoFilter.BlockType != 2)
                throw new Exception($"Raycast_Transparent (no filter): expected glass hit, got type={hitNoFilter.BlockType}");

            // With filter: skip glass (type 2), hit stone
            var hitFiltered = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8.5f, 8.5f, 0.5f), Vector3.UnitZ, 20f,
                blockFilter: packed => PackedBlock.GetType(packed) != 2); // stop on non-glass

            if (!hitFiltered.DidHit || hitFiltered.BlockType != 1)
                throw new Exception($"Raycast_Transparent (filtered): expected stone hit, got type={hitFiltered.BlockType}");
            if (hitFiltered.BlockZ != 8)
                throw new Exception($"Raycast_Transparent (filtered): expected z=8, got z={hitFiltered.BlockZ}");
        }

        /// <summary>
        /// Test 4: Axis-aligned rays hit correct faces.
        /// Tests all 6 directions to verify face detection.
        /// </summary>
        [TestMethod]
        public void Raycast_AllSixDirections_CorrectFacesTest()
        {
            // Block at center (8,8,8)
            var section = CreateTestSection((8, 8, 8, 1));

            // Test each of the 6 directions
            var tests = new[]
            {
                (origin: new Vector3(0.5f, 8.5f, 8.5f), dir: Vector3.UnitX, expectedFace: VoxelMeshConstants.FaceNegX, name: "+X ray hits -X face"),
                (origin: new Vector3(15.5f, 8.5f, 8.5f), dir: -Vector3.UnitX, expectedFace: VoxelMeshConstants.FacePosX, name: "-X ray hits +X face"),
                (origin: new Vector3(8.5f, 0.5f, 8.5f), dir: Vector3.UnitY, expectedFace: VoxelMeshConstants.FaceNegY, name: "+Y ray hits -Y face"),
                (origin: new Vector3(8.5f, 15.5f, 8.5f), dir: -Vector3.UnitY, expectedFace: VoxelMeshConstants.FacePosY, name: "-Y ray hits +Y face"),
                (origin: new Vector3(8.5f, 8.5f, 0.5f), dir: Vector3.UnitZ, expectedFace: VoxelMeshConstants.FaceNegZ, name: "+Z ray hits -Z face"),
                (origin: new Vector3(8.5f, 8.5f, 15.5f), dir: -Vector3.UnitZ, expectedFace: VoxelMeshConstants.FacePosZ, name: "-Z ray hits +Z face"),
            };

            foreach (var (origin, dir, expectedFace, name) in tests)
            {
                var hit = VoxelRaycast.Cast(section, 16, 16, origin, dir, 20f);
                if (!hit.DidHit)
                    throw new Exception($"Raycast_AllSixDirections [{name}]: expected hit, got miss");
                if (hit.HitFace != expectedFace)
                    throw new Exception($"Raycast_AllSixDirections [{name}]: wrong face {hit.HitFace}, expected {expectedFace}");
            }
        }

        /// <summary>
        /// Test 5: Diagonal ray hits correct block and face.
        /// Non-axis-aligned ray at 45 degrees in XZ plane.
        /// </summary>
        [TestMethod]
        public void Raycast_DiagonalRay_HitsCorrectBlockTest()
        {
            // Line of blocks: (4,8,4), (6,8,6), (8,8,8)
            var section = CreateTestSection(
                (4, 8, 4, 10),
                (6, 8, 6, 20),
                (8, 8, 8, 30));

            // Diagonal ray from (0.5, 8.5, 0.5) in normalized (1,0,1) direction
            var dir = Vector3.Normalize(new Vector3(1, 0, 1));
            var hit = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(0.5f, 8.5f, 0.5f), dir, 30f);

            if (!hit.DidHit)
                throw new Exception("Raycast_Diagonal: expected hit, got miss");
            if (hit.BlockX != 4 || hit.BlockZ != 4)
                throw new Exception($"Raycast_Diagonal: expected first block (4,_,4), got ({hit.BlockX},_,{hit.BlockZ})");
            if (hit.BlockType != 10)
                throw new Exception($"Raycast_Diagonal: expected type 10, got {hit.BlockType}");
        }

        /// <summary>
        /// Test 6: Ray starting inside a solid block returns immediate hit at distance 0.
        /// </summary>
        [TestMethod]
        public void Raycast_InsideBlock_ImmediateHitTest()
        {
            var section = CreateTestSection((8, 8, 8, 1));

            // Start inside the block
            var hit = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8.5f, 8.5f, 8.5f), Vector3.UnitZ, 20f);

            if (!hit.DidHit)
                throw new Exception("Raycast_InsideBlock: expected hit, got miss");
            if (hit.BlockX != 8 || hit.BlockY != 8 || hit.BlockZ != 8)
                throw new Exception($"Raycast_InsideBlock: wrong block ({hit.BlockX},{hit.BlockY},{hit.BlockZ})");
            if (hit.Distance != 0f)
                throw new Exception($"Raycast_InsideBlock: expected distance 0, got {hit.Distance}");
        }

        /// <summary>
        /// Test 7: Ray hits multiple blocks in sequence - verify it returns the FIRST one.
        /// </summary>
        [TestMethod]
        public void Raycast_MultipleBlocks_HitsFirstTest()
        {
            // Three blocks in a row along Z: z=4, z=8, z=12
            var section = CreateTestSection(
                (8, 8, 4, 100),
                (8, 8, 8, 200),
                (8, 8, 12, 300));

            var hit = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8.5f, 8.5f, 0.5f), Vector3.UnitZ, 20f);

            if (!hit.DidHit)
                throw new Exception("Raycast_MultipleBlocks: expected hit, got miss");
            if (hit.BlockType != 100)
                throw new Exception($"Raycast_MultipleBlocks: expected first block type 100, got {hit.BlockType}");
            if (hit.BlockZ != 4)
                throw new Exception($"Raycast_MultipleBlocks: expected z=4, got z={hit.BlockZ}");
        }

        /// <summary>
        /// Test 8: Block with damage - verify damage level is returned.
        /// </summary>
        [TestMethod]
        public void Raycast_DamagedBlock_ReturnsDamageLevelTest()
        {
            int size = 16;
            var section = new int[size * size * size];
            // Place a block with type 42 and damage 7
            section[8 + 8 * size + 8 * size * size] = PackedBlock.Pack(42, 7);

            var hit = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8.5f, 8.5f, 0.5f), Vector3.UnitZ, 20f);

            if (!hit.DidHit)
                throw new Exception("Raycast_DamagedBlock: expected hit, got miss");
            if (hit.BlockType != 42)
                throw new Exception($"Raycast_DamagedBlock: expected type 42, got {hit.BlockType}");
            if (hit.DamageLevel != 7)
                throw new Exception($"Raycast_DamagedBlock: expected damage 7, got {hit.DamageLevel}");
        }

        /// <summary>
        /// Test 9: Max distance limit - block exists beyond max distance, should not be hit.
        /// </summary>
        [TestMethod]
        public void Raycast_MaxDistance_RespectsLimitTest()
        {
            // Block at z=14, ray max distance = 5 voxels (won't reach it from z=0.5)
            var section = CreateTestSection((8, 8, 14, 1));

            var hit = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8.5f, 8.5f, 0.5f), Vector3.UnitZ, 5f);

            if (hit.DidHit)
                throw new Exception("Raycast_MaxDistance: expected no hit (block beyond max distance), got hit");

            // Same ray with sufficient distance should hit
            var hit2 = VoxelRaycast.Cast(section, 16, 16,
                new Vector3(8.5f, 8.5f, 0.5f), Vector3.UnitZ, 20f);

            if (!hit2.DidHit)
                throw new Exception("Raycast_MaxDistance: expected hit with sufficient distance, got miss");
        }

        /// <summary>
        /// Test 10: Adjacent block coordinates for block placement.
        /// When placing a block, it goes at the adjacent position (one step along hit face normal).
        /// </summary>
        [TestMethod]
        public void Raycast_AdjacentBlock_CorrectForPlacementTest()
        {
            var section = CreateTestSection((8, 8, 8, 1));

            // Hit from each direction, verify adjacent block is correct for placement
            var tests = new[]
            {
                (origin: new Vector3(0.5f, 8.5f, 8.5f), dir: Vector3.UnitX,
                    adjX: 7, adjY: 8, adjZ: 8, name: "-X face placement"),
                (origin: new Vector3(8.5f, 0.5f, 8.5f), dir: Vector3.UnitY,
                    adjX: 8, adjY: 7, adjZ: 8, name: "-Y face placement"),
                (origin: new Vector3(8.5f, 8.5f, 0.5f), dir: Vector3.UnitZ,
                    adjX: 8, adjY: 8, adjZ: 7, name: "-Z face placement"),
                (origin: new Vector3(15.5f, 8.5f, 8.5f), dir: -Vector3.UnitX,
                    adjX: 9, adjY: 8, adjZ: 8, name: "+X face placement"),
                (origin: new Vector3(8.5f, 15.5f, 8.5f), dir: -Vector3.UnitY,
                    adjX: 8, adjY: 9, adjZ: 8, name: "+Y face placement"),
                (origin: new Vector3(8.5f, 8.5f, 15.5f), dir: -Vector3.UnitZ,
                    adjX: 8, adjY: 8, adjZ: 9, name: "+Z face placement"),
            };

            foreach (var (origin, dir, adjX, adjY, adjZ, name) in tests)
            {
                var hit = VoxelRaycast.Cast(section, 16, 16, origin, dir, 20f);
                if (!hit.DidHit)
                    throw new Exception($"Raycast_Adjacent [{name}]: expected hit, got miss");
                if (hit.AdjacentX != adjX || hit.AdjacentY != adjY || hit.AdjacentZ != adjZ)
                    throw new Exception($"Raycast_Adjacent [{name}]: adjacent ({hit.AdjacentX},{hit.AdjacentY},{hit.AdjacentZ}), expected ({adjX},{adjY},{adjZ})");
            }
        }
    }
}
