using System.Numerics;

namespace SpawnDev.VoxelEngine.Culling
{
    /// <summary>
    /// Culling pipeline orchestrator.
    /// Integrates all culling stages in order:
    ///
    /// 1. Graph BFS (CPU) - VisibilityGraph eliminates underground sections
    /// 2. Frustum cull (GPU) - FrustumCullKernels tests AABB vs frustum planes
    /// 3. Fog cull (GPU) - sections beyond fog distance rejected
    /// 4. Face masking - FaceMaskService skips back-facing quad groups
    ///
    /// Hi-Z occlusion is reserved but NOT implemented in v1.0 (Geordi recommendation).
    /// Graph visibility + frustum handles 95%+ of occlusion for voxel worlds.
    ///
    /// Input: all loaded sections (SectionEntry[])
    /// Output: visible section indices + per-section face masks
    /// </summary>
    public class CullingPipeline
    {
        private readonly VoxelEngineConfig _config;

        /// <summary>Visibility graph for underground culling.</summary>
        public bool EnableGraphVisibility { get; set; } = true;

        /// <summary>Frustum culling.</summary>
        public bool EnableFrustumCull { get; set; } = true;

        /// <summary>Fog distance culling.</summary>
        public bool EnableFogCull { get; set; } = true;

        /// <summary>Face masking (skip back-facing quad groups).</summary>
        public bool EnableFaceMasking { get; set; } = true;

        /// <summary>Fog distance in world units.</summary>
        public float FogDistance { get; set; } = 256f;

        /// <summary>Statistics from the last cull pass.</summary>
        public CullStats LastStats { get; private set; } = new();

        public CullingPipeline(VoxelEngineConfig config)
        {
            _config = config;
        }

        /// <summary>
        /// Run the full culling pipeline on CPU.
        /// Returns indices of visible sections and per-section visible face masks.
        /// </summary>
        /// <param name="sections">All loaded sections.</param>
        /// <param name="cameraPos">Camera position in world space.</param>
        /// <param name="cameraForward">Camera forward direction (normalized).</param>
        /// <param name="frustumPlanes">6 frustum planes (24 floats: nx,ny,nz,d per plane).</param>
        /// <param name="getConnectivity">Connectivity lookup for graph visibility.</param>
        public CullOutput Cull(
            ReadOnlySpan<SectionEntry> sections,
            Vector3 cameraPos,
            Vector3 cameraForward,
            float[] frustumPlanes,
            Func<SectionCoord, long>? getConnectivity = null)
        {
            int totalSections = sections.Length;
            var visibleIndices = new List<int>(totalSections);
            var visibleFaceMasks = new List<int>(totalSections); // 6-bit mask per section

            // Stage 1: Graph BFS visibility
            HashSet<SectionCoord>? graphVisible = null;
            int graphCulled = 0;
            if (EnableGraphVisibility && getConnectivity != null)
            {
                // Find camera's section
                var camSection = WorldToSection(cameraPos);
                graphVisible = VisibilityGraph.ComputeVisibleSections(
                    camSection, getConnectivity, _config.DrawDistance);
            }

            for (int i = 0; i < totalSections; i++)
            {
                var section = sections[i];

                // Stage 1: Graph visibility check
                if (graphVisible != null && !graphVisible.Contains(section.Coord))
                {
                    graphCulled++;
                    continue;
                }

                // Stage 2: Fog distance check
                if (EnableFogCull)
                {
                    var sectionCenter = GetSectionCenter(section.Coord);
                    float distSq = Vector3.DistanceSquared(cameraPos, sectionCenter);
                    if (distSq > FogDistance * FogDistance)
                    {
                        continue;
                    }
                }

                // Stage 3: Frustum cull (AABB vs 6 planes)
                if (EnableFrustumCull && frustumPlanes != null)
                {
                    var min = section.Coord.WorldMin(_config.VoxelSize, _config.SectionSize, _config.BaseY);
                    var max = section.Coord.WorldMax(_config.VoxelSize, _config.SectionSize, _config.BaseY);

                    if (!AABBInFrustum(min, max, frustumPlanes))
                    {
                        continue;
                    }
                }

                // Stage 4: Face masking
                int faceMask = 0x3F; // all 6 faces visible by default
                if (EnableFaceMasking)
                {
                    var sectionCenter = GetSectionCenter(section.Coord);
                    var toSection = Vector3.Normalize(sectionCenter - cameraPos);
                    faceMask = ComputeFaceMask(toSection);
                }

                visibleIndices.Add(i);
                visibleFaceMasks.Add(faceMask);
            }

            LastStats = new CullStats
            {
                TotalSections = totalSections,
                GraphCulled = graphCulled,
                VisibleSections = visibleIndices.Count,
                CullRatio = 1f - (float)visibleIndices.Count / Math.Max(1, totalSections),
            };

            return new CullOutput
            {
                VisibleIndices = visibleIndices.ToArray(),
                FaceMasks = visibleFaceMasks.ToArray(),
            };
        }

        private SectionCoord WorldToSection(Vector3 worldPos)
        {
            float vs = _config.VoxelSize;
            int ss = _config.SectionSize;
            return new SectionCoord(
                (int)MathF.Floor(worldPos.X / (ss * vs)),
                (int)MathF.Floor((worldPos.Y - _config.BaseY) / (ss * vs)),
                (int)MathF.Floor(worldPos.Z / (ss * vs)));
        }

        private Vector3 GetSectionCenter(SectionCoord coord)
        {
            float half = _config.SectionSize * _config.VoxelSize * 0.5f;
            return coord.WorldMin(_config.VoxelSize, _config.SectionSize, _config.BaseY) + new Vector3(half);
        }

        /// <summary>Test AABB against 6 frustum planes. Returns true if any part is inside.</summary>
        private static bool AABBInFrustum(Vector3 min, Vector3 max, float[] planes)
        {
            for (int i = 0; i < 6; i++)
            {
                int p = i * 4;
                float nx = planes[p], ny = planes[p + 1], nz = planes[p + 2], d = planes[p + 3];

                // Test the positive vertex (corner most aligned with plane normal)
                float px = nx >= 0 ? max.X : min.X;
                float py = ny >= 0 ? max.Y : min.Y;
                float pz = nz >= 0 ? max.Z : min.Z;

                if (nx * px + ny * py + nz * pz + d < 0)
                    return false; // fully outside this plane
            }
            return true;
        }

        /// <summary>Compute 6-bit face visibility mask from camera-to-section direction.</summary>
        private static int ComputeFaceMask(Vector3 toSection)
        {
            int mask = 0;
            // A face is visible if the camera could see it (dot product with face normal > threshold)
            float threshold = -0.1f; // slight margin for edge cases

            if (toSection.X > threshold) mask |= (1 << 0); // +X face visible (camera is in +X direction)
            if (toSection.X < -threshold) mask |= (1 << 1); // -X
            if (toSection.Z > threshold) mask |= (1 << 2); // +Z
            if (toSection.Z < -threshold) mask |= (1 << 3); // -Z
            if (toSection.Y > threshold) mask |= (1 << 4); // +Y
            if (toSection.Y < -threshold) mask |= (1 << 5); // -Y

            return mask == 0 ? 0x3F : mask; // fallback: show all if inside section
        }
    }

    /// <summary>Output from the culling pipeline.</summary>
    public class CullOutput
    {
        /// <summary>Indices into the SectionEntry array for visible sections.</summary>
        public int[] VisibleIndices { get; init; } = Array.Empty<int>();

        /// <summary>Per-visible-section 6-bit face masks (parallel with VisibleIndices).</summary>
        public int[] FaceMasks { get; init; } = Array.Empty<int>();

        /// <summary>Number of visible sections.</summary>
        public int VisibleCount => VisibleIndices.Length;
    }

    /// <summary>Statistics from the last cull pass.</summary>
    public struct CullStats
    {
        public int TotalSections;
        public int GraphCulled;
        public int VisibleSections;
        public float CullRatio;

        public override string ToString() =>
            $"Visible: {VisibleSections}/{TotalSections} ({CullRatio:P0} culled, {GraphCulled} graph)";
    }
}
