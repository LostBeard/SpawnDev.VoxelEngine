using System.Numerics;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Physics;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // AABB sweep collision tests: VoxelCollision sweep, overlap, and ground detection.
    // Tests the exact code path used for player movement, entity physics, and falling.
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>
        /// Test 1: AABB resting on solid floor - sweep downward is stopped at surface.
        /// Player-sized AABB (0.6 x 1.8 x 0.6) sitting on a 3x3 floor at y=0.
        /// A small downward velocity should collide immediately (time near 0).
        /// </summary>
        [TestMethod]
        public void Collision_RestingOnFloor_NoFallthroughTest()
        {
            // Build a 3x3 floor of stone at y=0, centered at x=7..9, z=7..9
            var section = CreateTestSection(
                (7, 0, 7, 1), (8, 0, 7, 1), (9, 0, 7, 1),
                (7, 0, 8, 1), (8, 0, 8, 1), (9, 0, 8, 1),
                (7, 0, 9, 1), (8, 0, 9, 1), (9, 0, 9, 1));

            // AABB: 0.6 wide, 1.8 tall, 0.6 deep. Min corner at (7.7, 1.0, 7.7)
            // so the AABB bottom face is exactly at y=1.0, sitting on top of the floor block (y=0 to y=1).
            Vector3 pos = new(7.7f, 1.0f, 7.7f);
            Vector3 size = new(0.6f, 1.8f, 0.6f);
            Vector3 velocity = new(0, -0.5f, 0); // falling downward

            var result = VoxelCollision.Sweep(section, 16, 16, pos, size, velocity);

            if (!result.Hit)
                throw new Exception("Collision_RestingOnFloor: expected collision with floor, got no hit");
            if (result.Time > 0.01f)
                throw new Exception($"Collision_RestingOnFloor: AABB is touching floor, time should be ~0, got {result.Time}");
            if (result.Normal != Vector3.UnitY)
                throw new Exception($"Collision_RestingOnFloor: normal should be +Y (upward from floor), got {result.Normal}");
            if (result.BlockY != 0)
                throw new Exception($"Collision_RestingOnFloor: should hit floor at y=0, got y={result.BlockY}");
        }

        /// <summary>
        /// Test 2: AABB moving into a wall - stopped at surface with time less than 1.
        /// Player walks east (+X) into a solid wall. Sweep must stop before penetrating.
        /// </summary>
        [TestMethod]
        public void Collision_WalkIntoWall_StoppedAtSurfaceTest()
        {
            // Wall of blocks at x=10, y=0..2, z=7..9
            var section = CreateTestSection(
                (10, 0, 7, 1), (10, 0, 8, 1), (10, 0, 9, 1),
                (10, 1, 7, 1), (10, 1, 8, 1), (10, 1, 9, 1),
                (10, 2, 7, 1), (10, 2, 8, 1), (10, 2, 9, 1));

            // AABB at (7.0, 0.0, 7.7), size (0.6, 1.8, 0.6), moving +X by 5 units
            Vector3 pos = new(7.0f, 0.0f, 7.7f);
            Vector3 size = new(0.6f, 1.8f, 0.6f);
            Vector3 velocity = new(5.0f, 0, 0);

            var result = VoxelCollision.Sweep(section, 16, 16, pos, size, velocity);

            if (!result.Hit)
                throw new Exception("Collision_WalkIntoWall: expected collision with wall, got no hit");
            // AABB right edge at 7.0 + 0.6 = 7.6, wall at x=10. Gap = 2.4 units. Velocity = 5.0.
            // Expected time = 2.4 / 5.0 = 0.48
            float expectedTime = (10.0f - (pos.X + size.X)) / velocity.X;
            if (MathF.Abs(result.Time - expectedTime) > 0.001f)
                throw new Exception($"Collision_WalkIntoWall: expected time ~{expectedTime:F4}, got {result.Time:F4}");
            if (result.Normal != -Vector3.UnitX)
                throw new Exception($"Collision_WalkIntoWall: normal should be -X (wall facing player), got {result.Normal}");
            if (result.Time >= 1f)
                throw new Exception($"Collision_WalkIntoWall: time must be < 1 (collision occurred), got {result.Time}");
            // Resolved position should place AABB flush against wall
            float resolvedRight = result.ResolvedPosition.X + size.X;
            if (MathF.Abs(resolvedRight - 10.0f) > 0.01f)
                throw new Exception($"Collision_WalkIntoWall: resolved right edge should be at 10.0, got {resolvedRight:F4}");
        }

        /// <summary>
        /// Test 3: AABB moving through completely empty space - no collision, time = 1.
        /// </summary>
        [TestMethod]
        public void Collision_EmptySpace_NoCollisionTest()
        {
            var section = new int[16 * 16 * 16]; // all air

            Vector3 pos = new(2.0f, 5.0f, 3.0f);
            Vector3 size = new(0.6f, 1.8f, 0.6f);
            Vector3 velocity = new(3.0f, -2.0f, 1.5f); // diagonal movement

            var result = VoxelCollision.Sweep(section, 16, 16, pos, size, velocity);

            if (result.Hit)
                throw new Exception("Collision_EmptySpace: expected no collision in empty section, got hit");
            if (result.Time != 1f)
                throw new Exception($"Collision_EmptySpace: no-collision time should be 1.0, got {result.Time}");
        }

        /// <summary>
        /// Test 4: Diagonal movement into wall corner - slides along the wall.
        /// Player moves diagonally (+X, 0, +Z) and hits a wall on the X axis.
        /// After collision, remaining velocity should preserve the Z component (slide along wall).
        /// </summary>
        [TestMethod]
        public void Collision_DiagonalIntoWall_SlidesAlongTest()
        {
            // Wall at x=10, z=5..10, y=0..1 - a long wall to slide along
            var section = CreateTestSection(
                (10, 0, 5, 1), (10, 0, 6, 1), (10, 0, 7, 1), (10, 0, 8, 1), (10, 0, 9, 1), (10, 0, 10, 1),
                (10, 1, 5, 1), (10, 1, 6, 1), (10, 1, 7, 1), (10, 1, 8, 1), (10, 1, 9, 1), (10, 1, 10, 1));

            // AABB approaching from left diagonally
            Vector3 pos = new(8.0f, 0.0f, 6.0f);
            Vector3 size = new(0.6f, 1.8f, 0.6f);
            Vector3 velocity = new(4.0f, 0, 3.0f); // diagonal: +X and +Z

            var result = VoxelCollision.Sweep(section, 16, 16, pos, size, velocity);

            if (!result.Hit)
                throw new Exception("Collision_DiagonalSlide: expected wall collision, got no hit");
            if (result.Normal != -Vector3.UnitX)
                throw new Exception($"Collision_DiagonalSlide: expected -X normal (wall), got {result.Normal}");

            // Remaining velocity should have X zeroed (slid off wall) but Z preserved
            if (MathF.Abs(result.RemainingVelocity.X) > 0.001f)
                throw new Exception($"Collision_DiagonalSlide: X velocity should be zeroed after slide, got {result.RemainingVelocity.X:F4}");
            if (MathF.Abs(result.RemainingVelocity.Z) < 0.01f)
                throw new Exception($"Collision_DiagonalSlide: Z velocity should be preserved for slide, got {result.RemainingVelocity.Z:F4}");
        }

        /// <summary>
        /// Test 5: Transparent block pass-through with filter.
        /// Water (type 2) does not block movement when the filter excludes it.
        /// Stone (type 1) behind the water still blocks.
        /// </summary>
        [TestMethod]
        public void Collision_TransparentFilter_PassesThroughTest()
        {
            // Water at x=6, stone wall at x=10
            var section = CreateTestSection(
                (6, 0, 8, 2), (6, 1, 8, 2),   // water (type 2) - should pass through
                (10, 0, 8, 1), (10, 1, 8, 1)); // stone (type 1) - should block

            Vector3 pos = new(3.0f, 0.0f, 7.7f);
            Vector3 size = new(0.6f, 1.8f, 0.6f);
            Vector3 velocity = new(9.0f, 0, 0);

            // Without filter: hits water first at x=6
            var hitAll = VoxelCollision.Sweep(section, 16, 16, pos, size, velocity);
            if (!hitAll.Hit)
                throw new Exception("Collision_TransparentFilter (no filter): expected hit on water");
            if (hitAll.BlockX != 6)
                throw new Exception($"Collision_TransparentFilter (no filter): expected block at x=6, got x={hitAll.BlockX}");

            // With filter: water (type 2) is not solid, passes through to stone at x=10
            var hitFiltered = VoxelCollision.Sweep(section, 16, 16, pos, size, velocity,
                solidFilter: packed => PackedBlock.GetType(packed) != 2);
            if (!hitFiltered.Hit)
                throw new Exception("Collision_TransparentFilter (filtered): expected hit on stone behind water");
            if (hitFiltered.BlockX != 10)
                throw new Exception($"Collision_TransparentFilter (filtered): expected block at x=10, got x={hitFiltered.BlockX}");
        }

        /// <summary>
        /// Test 6: IsOnGround - returns true when a block is directly below the AABB,
        /// false when the AABB is floating in air.
        /// </summary>
        [TestMethod]
        public void Collision_IsOnGround_DetectsFloorTest()
        {
            // Floor at y=0
            var section = CreateTestSection(
                (7, 0, 7, 1), (8, 0, 7, 1),
                (7, 0, 8, 1), (8, 0, 8, 1));

            Vector3 size = new(0.6f, 1.8f, 0.6f);

            // Case A: AABB sitting on floor (bottom at y=1.0, floor top at y=1.0)
            Vector3 onFloor = new(7.2f, 1.0f, 7.2f);
            bool grounded = VoxelCollision.IsOnGround(section, 16, 16, onFloor, size);
            if (!grounded)
                throw new Exception("Collision_IsOnGround: AABB on floor should be grounded, got false");

            // Case B: AABB floating 3 blocks above floor (bottom at y=4.0)
            Vector3 inAir = new(7.2f, 4.0f, 7.2f);
            bool airborne = VoxelCollision.IsOnGround(section, 16, 16, inAir, size);
            if (airborne)
                throw new Exception("Collision_IsOnGround: AABB in air should NOT be grounded, got true");

            // Case C: AABB slightly above floor (bottom at y=1.005, within epsilon)
            Vector3 nearFloor = new(7.2f, 1.005f, 7.2f);
            bool nearGrounded = VoxelCollision.IsOnGround(section, 16, 16, nearFloor, size);
            if (!nearGrounded)
                throw new Exception("Collision_IsOnGround: AABB within epsilon of floor should be grounded, got false");
        }

        /// <summary>
        /// Test 7: IsOverlapping - returns true when AABB intersects a solid block,
        /// false when AABB is in empty space. Tests partial overlap too.
        /// </summary>
        [TestMethod]
        public void Collision_IsOverlapping_DetectsIntersectionTest()
        {
            var section = CreateTestSection(
                (8, 4, 8, 1),  // solid block at (8,4,8)
                (8, 5, 8, 1)); // solid block at (8,5,8)

            Vector3 size = new(0.6f, 1.8f, 0.6f);

            // Case A: AABB clearly inside the block
            Vector3 inside = new(8.1f, 4.1f, 8.1f);
            if (!VoxelCollision.IsOverlapping(section, 16, 16, inside, size))
                throw new Exception("Collision_IsOverlapping: AABB inside solid block should overlap, got false");

            // Case B: AABB completely outside (far from the block)
            Vector3 outside = new(2.0f, 2.0f, 2.0f);
            if (VoxelCollision.IsOverlapping(section, 16, 16, outside, size))
                throw new Exception("Collision_IsOverlapping: AABB far from block should NOT overlap, got true");

            // Case C: AABB partially overlapping (edge case)
            // AABB right edge at 7.9 + 0.6 = 8.5, block starts at x=8. Overlap = 0.5 units.
            Vector3 partial = new(7.9f, 4.2f, 8.1f);
            if (!VoxelCollision.IsOverlapping(section, 16, 16, partial, size))
                throw new Exception("Collision_IsOverlapping: AABB partially overlapping block should overlap, got false");

            // Case D: AABB adjacent but not overlapping (just outside block boundary)
            // AABB right edge at 7.3 + 0.6 = 7.9 < 8.0 (block starts at x=8)
            Vector3 adjacent = new(7.3f, 4.2f, 8.1f);
            if (VoxelCollision.IsOverlapping(section, 16, 16, adjacent, size))
                throw new Exception("Collision_IsOverlapping: AABB adjacent (not touching) should NOT overlap, got true");

            // Case E: IsOverlapping with filter - only type 5 is solid, type 1 should pass through
            if (VoxelCollision.IsOverlapping(section, 16, 16, inside, size,
                solidFilter: packed => PackedBlock.GetType(packed) == 5))
                throw new Exception("Collision_IsOverlapping (filter): type 1 blocks should not be solid with type-5 filter, got true");
        }
    }
}
