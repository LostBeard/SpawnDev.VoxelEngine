using System.Numerics;
using System.Runtime.CompilerServices;

namespace SpawnDev.VoxelEngine.Physics
{
    /// <summary>
    /// AABB-voxel collision detection and response.
    /// Sweep test: given an AABB + velocity, finds the first block collision
    /// and computes a slide response along the colliding surface.
    ///
    /// Used for: player movement (can't walk through walls), falling (ground detection),
    /// jumping (ceiling detection), entity physics.
    ///
    /// Algorithm:
    /// 1. Expand AABB by velocity to get the broadphase region
    /// 2. Check all blocks within the broadphase region
    /// 3. For each solid block, compute swept AABB vs AABB intersection
    /// 4. Find the earliest collision time and normal
    /// 5. Apply slide response: zero velocity component along collision normal
    ///
    /// Operates in section-local coordinates. Caller handles cross-section boundaries.
    /// </summary>
    public static class VoxelCollision
    {
        /// <summary>
        /// Result of an AABB sweep test against the voxel grid.
        /// </summary>
        public struct SweepResult
        {
            /// <summary>Whether any collision occurred.</summary>
            public bool Hit;

            /// <summary>Parametric time of first collision (0-1 along velocity vector).</summary>
            public float Time;

            /// <summary>Surface normal of the collision (axis-aligned).</summary>
            public Vector3 Normal;

            /// <summary>Remaining velocity after slide response.</summary>
            public Vector3 RemainingVelocity;

            /// <summary>Position of the AABB after sliding to the collision point.</summary>
            public Vector3 ResolvedPosition;

            /// <summary>Block coordinates of the colliding block.</summary>
            public int BlockX, BlockY, BlockZ;

            /// <summary>No collision result.</summary>
            public static readonly SweepResult None = new() { Hit = false, Time = 1f };
        }

        /// <summary>
        /// Sweep an AABB along a velocity vector through the voxel grid.
        /// Returns the first collision and slide response.
        ///
        /// The AABB is defined by its minimum corner (position) and size (half-extents * 2).
        /// </summary>
        /// <param name="blocks">Section voxel data (PackedBlock format, flat array).</param>
        /// <param name="sizeXZ">Section size XZ.</param>
        /// <param name="sizeY">Section size Y.</param>
        /// <param name="position">AABB minimum corner in section-local space.</param>
        /// <param name="size">AABB size (width, height, depth).</param>
        /// <param name="velocity">Movement vector for this frame.</param>
        /// <param name="solidFilter">Returns true for blocks that should be solid. Null = all non-air are solid.</param>
        public static SweepResult Sweep(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            Vector3 position, Vector3 size, Vector3 velocity,
            Func<int, bool>? solidFilter = null)
        {
            if (velocity == Vector3.Zero)
                return SweepResult.None;

            // Broadphase: find all blocks the AABB could possibly touch during movement
            Vector3 endPos = position + velocity;
            Vector3 broadMin = Vector3.Min(position, endPos);
            Vector3 broadMax = Vector3.Max(position + size, endPos + size);

            int minBX = Math.Max(0, (int)MathF.Floor(broadMin.X));
            int minBY = Math.Max(0, (int)MathF.Floor(broadMin.Y));
            int minBZ = Math.Max(0, (int)MathF.Floor(broadMin.Z));
            int maxBX = Math.Min(sizeXZ - 1, (int)MathF.Floor(broadMax.X));
            int maxBY = Math.Min(sizeY - 1, (int)MathF.Floor(broadMax.Y));
            int maxBZ = Math.Min(sizeXZ - 1, (int)MathF.Floor(broadMax.Z));

            float earliestTime = 1f;
            Vector3 earliestNormal = Vector3.Zero;
            int hitBX = 0, hitBY = 0, hitBZ = 0;
            bool anyHit = false;

            // Test each potential block
            for (int by = minBY; by <= maxBY; by++)
            {
                for (int bz = minBZ; bz <= maxBZ; bz++)
                {
                    for (int bx = minBX; bx <= maxBX; bx++)
                    {
                        int packed = blocks[bx + bz * sizeXZ + by * sizeXZ * sizeXZ];
                        if (PackedBlock.IsAir(packed)) continue;
                        if (solidFilter != null && !solidFilter(packed)) continue;

                        // Block AABB: min = (bx, by, bz), max = (bx+1, by+1, bz+1)
                        float t = SweptAABB(
                            position, size, velocity,
                            new Vector3(bx, by, bz), Vector3.One,
                            out Vector3 normal);

                        if (t < earliestTime)
                        {
                            earliestTime = t;
                            earliestNormal = normal;
                            hitBX = bx; hitBY = by; hitBZ = bz;
                            anyHit = true;
                        }
                    }
                }
            }

            if (!anyHit)
                return SweepResult.None;

            // Slide response: move to collision point, zero velocity along normal
            Vector3 resolvedPos = position + velocity * earliestTime;
            Vector3 remaining = velocity * (1f - earliestTime);

            // Remove velocity component along collision normal
            float dot = Vector3.Dot(remaining, earliestNormal);
            remaining -= earliestNormal * dot;

            return new SweepResult
            {
                Hit = true,
                Time = earliestTime,
                Normal = earliestNormal,
                RemainingVelocity = remaining,
                ResolvedPosition = resolvedPos,
                BlockX = hitBX,
                BlockY = hitBY,
                BlockZ = hitBZ,
            };
        }

        /// <summary>
        /// Simple overlap test: is the AABB overlapping any solid blocks?
        /// Used for ground detection (is player standing on something?).
        /// </summary>
        public static bool IsOverlapping(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            Vector3 position, Vector3 size,
            Func<int, bool>? solidFilter = null)
        {
            int minBX = Math.Max(0, (int)MathF.Floor(position.X));
            int minBY = Math.Max(0, (int)MathF.Floor(position.Y));
            int minBZ = Math.Max(0, (int)MathF.Floor(position.Z));
            int maxBX = Math.Min(sizeXZ - 1, (int)MathF.Floor(position.X + size.X));
            int maxBY = Math.Min(sizeY - 1, (int)MathF.Floor(position.Y + size.Y));
            int maxBZ = Math.Min(sizeXZ - 1, (int)MathF.Floor(position.Z + size.Z));

            for (int by = minBY; by <= maxBY; by++)
                for (int bz = minBZ; bz <= maxBZ; bz++)
                    for (int bx = minBX; bx <= maxBX; bx++)
                    {
                        int packed = blocks[bx + bz * sizeXZ + by * sizeXZ * sizeXZ];
                        if (PackedBlock.IsAir(packed)) continue;
                        if (solidFilter != null && !solidFilter(packed)) continue;

                        // Check actual AABB overlap (not just grid cell)
                        if (position.X + size.X > bx && position.X < bx + 1 &&
                            position.Y + size.Y > by && position.Y < by + 1 &&
                            position.Z + size.Z > bz && position.Z < bz + 1)
                            return true;
                    }

            return false;
        }

        /// <summary>
        /// Ground check: is there a solid block directly below the AABB within epsilon distance?
        /// </summary>
        public static bool IsOnGround(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            Vector3 position, Vector3 size,
            float epsilon = 0.01f,
            Func<int, bool>? solidFilter = null)
        {
            // Check a thin slab below the AABB
            Vector3 checkPos = position - new Vector3(0, epsilon, 0);
            Vector3 checkSize = new(size.X, epsilon, size.Z);
            return IsOverlapping(blocks, sizeXZ, sizeY, checkPos, checkSize, solidFilter);
        }

        /// <summary>
        /// Swept AABB vs static AABB intersection.
        /// Returns parametric time (0-1) of first contact, and the collision normal.
        /// Returns 1.0 if no collision within the movement.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SweptAABB(
            Vector3 posA, Vector3 sizeA, Vector3 vel,
            Vector3 posB, Vector3 sizeB,
            out Vector3 normal)
        {
            normal = Vector3.Zero;

            // Compute entry and exit distances per axis
            // Minkowski sum: expand B by A's size
            float xInvEntry, yInvEntry, zInvEntry;
            float xInvExit, yInvExit, zInvExit;

            if (vel.X > 0)
            {
                xInvEntry = posB.X - (posA.X + sizeA.X);
                xInvExit = (posB.X + sizeB.X) - posA.X;
            }
            else
            {
                xInvEntry = (posB.X + sizeB.X) - posA.X;
                xInvExit = posB.X - (posA.X + sizeA.X);
            }

            if (vel.Y > 0)
            {
                yInvEntry = posB.Y - (posA.Y + sizeA.Y);
                yInvExit = (posB.Y + sizeB.Y) - posA.Y;
            }
            else
            {
                yInvEntry = (posB.Y + sizeB.Y) - posA.Y;
                yInvExit = posB.Y - (posA.Y + sizeA.Y);
            }

            if (vel.Z > 0)
            {
                zInvEntry = posB.Z - (posA.Z + sizeA.Z);
                zInvExit = (posB.Z + sizeB.Z) - posA.Z;
            }
            else
            {
                zInvEntry = (posB.Z + sizeB.Z) - posA.Z;
                zInvExit = posB.Z - (posA.Z + sizeA.Z);
            }

            // Parametric times
            float xEntry, yEntry, zEntry;
            float xExit, yExit, zExit;

            if (vel.X == 0) { xEntry = float.NegativeInfinity; xExit = float.PositiveInfinity; }
            else { xEntry = xInvEntry / vel.X; xExit = xInvExit / vel.X; }

            if (vel.Y == 0) { yEntry = float.NegativeInfinity; yExit = float.PositiveInfinity; }
            else { yEntry = yInvEntry / vel.Y; yExit = yInvExit / vel.Y; }

            if (vel.Z == 0) { zEntry = float.NegativeInfinity; zExit = float.PositiveInfinity; }
            else { zEntry = zInvEntry / vel.Z; zExit = zInvExit / vel.Z; }

            float entryTime = MathF.Max(xEntry, MathF.Max(yEntry, zEntry));
            float exitTime = MathF.Min(xExit, MathF.Min(yExit, zExit));

            // No collision
            if (entryTime > exitTime || entryTime < 0 || entryTime > 1)
                return 1f;

            // Determine collision normal (which axis entered last = collision face)
            if (xEntry >= yEntry && xEntry >= zEntry)
                normal = vel.X > 0 ? -Vector3.UnitX : Vector3.UnitX;
            else if (yEntry >= xEntry && yEntry >= zEntry)
                normal = vel.Y > 0 ? -Vector3.UnitY : Vector3.UnitY;
            else
                normal = vel.Z > 0 ? -Vector3.UnitZ : Vector3.UnitZ;

            return entryTime;
        }
    }
}
