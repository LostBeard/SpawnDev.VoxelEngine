using System.Numerics;
using System.Runtime.CompilerServices;

namespace SpawnDev.VoxelEngine.Physics
{
    /// <summary>
    /// Fast ray-voxel intersection using DDA (Digital Differential Analyzer).
    /// Same algorithm Teardown uses for its entire renderer.
    ///
    /// Traverses a voxel grid along a ray, testing each voxel in order.
    /// Returns the first solid block hit with exact face, position, and distance.
    ///
    /// Based on Amanatides and Woo (1987) "A Fast Voxel Traversal Algorithm for Ray Tracing."
    ///
    /// Operates on section-local coordinates (0 to sectionSize-1 per axis).
    /// For world-space rays, the caller transforms ray origin into section-local space
    /// and calls this per-section along the ray path.
    ///
    /// Both CPU and GPU versions use the same algorithm.
    /// CPU version serves as the reference for GPU kernel verification.
    /// </summary>
    public static class VoxelRaycast
    {
        /// <summary>
        /// Cast a ray through a section's voxel grid. Returns the first solid block hit.
        ///
        /// Algorithm:
        /// 1. Find the starting voxel from the ray origin
        /// 2. Calculate tMax (parametric distance to next voxel boundary) per axis
        /// 3. Calculate tDelta (parametric distance between boundaries) per axis
        /// 4. Step to the nearest boundary, test that voxel
        /// 5. Repeat until hit, out of bounds, or max distance exceeded
        /// </summary>
        /// <param name="blocks">Section voxel data in PackedBlock format. Flat array indexed as [x + z * sizeXZ + y * sizeXZ * sizeXZ].</param>
        /// <param name="sizeXZ">Section size along X and Z axes (typically 16).</param>
        /// <param name="sizeY">Section size along Y axis (typically 16).</param>
        /// <param name="rayOrigin">Ray origin in section-local space (0-based, fractional positions allowed).</param>
        /// <param name="rayDir">Ray direction (must be normalized).</param>
        /// <param name="maxDistance">Maximum traversal distance in voxel units.</param>
        /// <param name="blockFilter">Optional filter - return true for blocks the ray should stop at. Null = stop at any non-air block.</param>
        public static RaycastHit Cast(
            ReadOnlySpan<int> blocks,
            int sizeXZ,
            int sizeY,
            Vector3 rayOrigin,
            Vector3 rayDir,
            float maxDistance,
            Func<int, bool>? blockFilter = null)
        {
            // Current voxel position (integer grid coordinates)
            int x = (int)MathF.Floor(rayOrigin.X);
            int y = (int)MathF.Floor(rayOrigin.Y);
            int z = (int)MathF.Floor(rayOrigin.Z);

            // Step direction per axis (+1 or -1)
            int stepX = rayDir.X >= 0 ? 1 : -1;
            int stepY = rayDir.Y >= 0 ? 1 : -1;
            int stepZ = rayDir.Z >= 0 ? 1 : -1;

            // Distance along ray to cross one voxel boundary per axis
            float tDeltaX = rayDir.X != 0 ? MathF.Abs(1f / rayDir.X) : float.MaxValue;
            float tDeltaY = rayDir.Y != 0 ? MathF.Abs(1f / rayDir.Y) : float.MaxValue;
            float tDeltaZ = rayDir.Z != 0 ? MathF.Abs(1f / rayDir.Z) : float.MaxValue;

            // Distance to the NEXT voxel boundary from the current position
            float tMaxX = GetTMax(rayOrigin.X, rayDir.X, stepX);
            float tMaxY = GetTMax(rayOrigin.Y, rayDir.Y, stepY);
            float tMaxZ = GetTMax(rayOrigin.Z, rayDir.Z, stepZ);

            // Face that was crossed to enter the current voxel (updated during traversal)
            // Initialize to -1 (origin is inside the first voxel)
            int hitFace = -1;

            // Check starting voxel (ray might start inside a solid block)
            if (IsInBounds(x, y, z, sizeXZ, sizeY))
            {
                int packed = GetBlock(blocks, x, y, z, sizeXZ);
                if (ShouldStop(packed, blockFilter))
                {
                    return BuildHit(x, y, z, hitFace, packed, 0f, rayOrigin, rayDir, stepX, stepY, stepZ);
                }
            }

            // Traverse the grid
            float t = 0f;
            int maxSteps = sizeXZ + sizeY + sizeXZ + 4; // generous upper bound to prevent infinite loops
            for (int step = 0; step < maxSteps; step++)
            {
                // Step to the nearest axis boundary
                if (tMaxX < tMaxY)
                {
                    if (tMaxX < tMaxZ)
                    {
                        t = tMaxX;
                        x += stepX;
                        tMaxX += tDeltaX;
                        hitFace = stepX > 0 ? Meshing.VoxelMeshConstants.FaceNegX : Meshing.VoxelMeshConstants.FacePosX;
                    }
                    else
                    {
                        t = tMaxZ;
                        z += stepZ;
                        tMaxZ += tDeltaZ;
                        hitFace = stepZ > 0 ? Meshing.VoxelMeshConstants.FaceNegZ : Meshing.VoxelMeshConstants.FacePosZ;
                    }
                }
                else
                {
                    if (tMaxY < tMaxZ)
                    {
                        t = tMaxY;
                        y += stepY;
                        tMaxY += tDeltaY;
                        hitFace = stepY > 0 ? Meshing.VoxelMeshConstants.FaceNegY : Meshing.VoxelMeshConstants.FacePosY;
                    }
                    else
                    {
                        t = tMaxZ;
                        z += stepZ;
                        tMaxZ += tDeltaZ;
                        hitFace = stepZ > 0 ? Meshing.VoxelMeshConstants.FaceNegZ : Meshing.VoxelMeshConstants.FacePosZ;
                    }
                }

                // Check distance limit
                if (t > maxDistance)
                    return RaycastHit.None;

                // Check bounds
                if (!IsInBounds(x, y, z, sizeXZ, sizeY))
                    return RaycastHit.None;

                // Test voxel
                int packed = GetBlock(blocks, x, y, z, sizeXZ);
                if (ShouldStop(packed, blockFilter))
                {
                    return BuildHit(x, y, z, hitFace, packed, t, rayOrigin, rayDir, stepX, stepY, stepZ);
                }
            }

            return RaycastHit.None;
        }

        /// <summary>
        /// Cast a ray through multiple sections along its path.
        /// Transforms world-space ray into each section's local space and tests in order.
        /// </summary>
        /// <param name="getSection">Function that returns section block data for a given section coordinate, or null if not loaded.</param>
        /// <param name="config">Engine configuration (voxel size, section size, base Y).</param>
        /// <param name="worldOrigin">Ray origin in world space.</param>
        /// <param name="worldDir">Ray direction in world space (must be normalized).</param>
        /// <param name="maxWorldDistance">Maximum distance in world units.</param>
        /// <param name="blockFilter">Optional block filter.</param>
        public static RaycastHit CastWorld(
            Func<SectionCoord, int[]?> getSection,
            VoxelEngineConfig config,
            Vector3 worldOrigin,
            Vector3 worldDir,
            float maxWorldDistance,
            Func<int, bool>? blockFilter = null)
        {
            float vs = config.VoxelSize;
            int ss = config.SectionSize;
            float maxVoxelDist = maxWorldDistance / vs;

            // Convert world origin to voxel-space
            Vector3 voxelOrigin = new(
                worldOrigin.X / vs,
                (worldOrigin.Y - config.BaseY) / vs,
                worldOrigin.Z / vs);

            // Walk sections along the ray path
            // Start with the section containing the origin
            float distTraveled = 0f;
            Vector3 currentOrigin = voxelOrigin;
            int maxSectionSteps = (int)(maxVoxelDist / ss) + 2;

            for (int sStep = 0; sStep < maxSectionSteps; sStep++)
            {
                // Which section are we in?
                int cx = (int)MathF.Floor(currentOrigin.X / ss);
                int sy = (int)MathF.Floor(currentOrigin.Y / ss);
                int cz = (int)MathF.Floor(currentOrigin.Z / ss);
                var coord = new SectionCoord(cx, sy, cz);

                // Get section data
                var sectionData = getSection(coord);
                if (sectionData != null)
                {
                    // Transform to section-local coordinates
                    Vector3 localOrigin = new(
                        currentOrigin.X - cx * ss,
                        currentOrigin.Y - sy * ss,
                        currentOrigin.Z - cz * ss);

                    float remainingDist = maxVoxelDist - distTraveled;
                    // Limit to section diagonal to avoid overshooting into next section
                    float sectionMaxDist = MathF.Min(remainingDist, ss * 1.8f); // sqrt(3) * ss

                    var hit = Cast(sectionData, ss, ss, localOrigin, worldDir, sectionMaxDist, blockFilter);
                    if (hit.DidHit)
                    {
                        // Convert back to world space
                        float worldDist = (distTraveled + hit.Distance) * vs;
                        return new RaycastHit
                        {
                            DidHit = true,
                            HitPosition = worldOrigin + worldDir * worldDist,
                            BlockX = hit.BlockX + cx * ss,
                            BlockY = hit.BlockY + sy * ss,
                            BlockZ = hit.BlockZ + cz * ss,
                            HitFace = hit.HitFace,
                            BlockType = hit.BlockType,
                            DamageLevel = hit.DamageLevel,
                            Distance = worldDist,
                            AdjacentX = hit.AdjacentX + cx * ss,
                            AdjacentY = hit.AdjacentY + sy * ss,
                            AdjacentZ = hit.AdjacentZ + cz * ss,
                        };
                    }
                }

                // Advance to the next section boundary
                // Find the exit point of the ray from the current section bounds
                float exitT = GetSectionExitT(currentOrigin, worldDir, cx * ss, sy * ss, cz * ss, ss);
                if (exitT <= 0 || float.IsInfinity(exitT))
                    exitT = ss; // fallback: advance one section length

                distTraveled += exitT;
                if (distTraveled >= maxVoxelDist)
                    return RaycastHit.None;

                // Move origin to just past the section boundary
                currentOrigin += worldDir * (exitT + 0.001f);
            }

            return RaycastHit.None;
        }

        /// <summary>Calculate parametric distance from the section entry to exit.</summary>
        private static float GetSectionExitT(Vector3 origin, Vector3 dir, float minX, float minY, float minZ, int size)
        {
            float maxX = minX + size;
            float maxY = minY + size;
            float maxZ = minZ + size;

            float txExit = dir.X > 0 ? (maxX - origin.X) / dir.X :
                           dir.X < 0 ? (minX - origin.X) / dir.X : float.MaxValue;
            float tyExit = dir.Y > 0 ? (maxY - origin.Y) / dir.Y :
                           dir.Y < 0 ? (minY - origin.Y) / dir.Y : float.MaxValue;
            float tzExit = dir.Z > 0 ? (maxZ - origin.Z) / dir.Z :
                           dir.Z < 0 ? (minZ - origin.Z) / dir.Z : float.MaxValue;

            return MathF.Min(txExit, MathF.Min(tyExit, tzExit));
        }

        /// <summary>
        /// Calculate initial tMax for one axis.
        /// tMax = distance along the ray to the FIRST voxel boundary crossing on this axis.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float GetTMax(float origin, float dir, int step)
        {
            if (dir == 0) return float.MaxValue;

            // Next boundary position
            float boundary = step > 0 ? MathF.Floor(origin) + 1f : MathF.Floor(origin);
            return (boundary - origin) / dir;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsInBounds(int x, int y, int z, int sizeXZ, int sizeY)
        {
            return (uint)x < (uint)sizeXZ && (uint)y < (uint)sizeY && (uint)z < (uint)sizeXZ;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetBlock(ReadOnlySpan<int> blocks, int x, int y, int z, int sizeXZ)
        {
            return blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldStop(int packed, Func<int, bool>? filter)
        {
            if (PackedBlock.IsAir(packed)) return false;
            if (filter != null) return filter(packed);
            return true; // stop at any non-air block
        }

        private static RaycastHit BuildHit(int x, int y, int z, int hitFace, int packed, float t,
            Vector3 rayOrigin, Vector3 rayDir, int stepX, int stepY, int stepZ)
        {
            // Face normal directions for adjacent block calculation
            int adjX = x, adjY = y, adjZ = z;
            if (hitFace >= 0)
            {
                switch (hitFace)
                {
                    case Meshing.VoxelMeshConstants.FacePosX: adjX = x + 1; break;
                    case Meshing.VoxelMeshConstants.FaceNegX: adjX = x - 1; break;
                    case Meshing.VoxelMeshConstants.FacePosZ: adjZ = z + 1; break;
                    case Meshing.VoxelMeshConstants.FaceNegZ: adjZ = z - 1; break;
                    case Meshing.VoxelMeshConstants.FacePosY: adjY = y + 1; break;
                    case Meshing.VoxelMeshConstants.FaceNegY: adjY = y - 1; break;
                }
            }

            return new RaycastHit
            {
                DidHit = true,
                HitPosition = rayOrigin + rayDir * t,
                BlockX = x,
                BlockY = y,
                BlockZ = z,
                HitFace = hitFace,
                BlockType = PackedBlock.GetType(packed),
                DamageLevel = PackedBlock.GetDamage(packed),
                Distance = t,
                AdjacentX = adjX,
                AdjacentY = adjY,
                AdjacentZ = adjZ,
            };
        }
    }
}
