namespace SpawnDev.VoxelEngine
{
    /// <summary>
    /// Section lifecycle state machine.
    /// Tracks the state of each loaded section from first request through GPU readiness.
    ///
    /// States:
    ///   UNLOADED -> LOADING -> MESHING -> GPU_READY -> VISIBLE -> CACHED
    ///
    /// UNLOADED: Not in memory. No block data, no mesh.
    /// LOADING: Block data being loaded from provider (async I/O).
    /// MESHING: Block data loaded, GPU mesh generation in progress.
    /// GPU_READY: Mesh uploaded to GPU, ready to draw but not yet visible (outside frustum or LOD).
    /// VISIBLE: Currently drawn. In the IndirectDrawBuffer.
    /// CACHED: Was visible, now outside draw distance. Mesh retained for quick re-show.
    /// </summary>
    public enum SectionState
    {
        Unloaded,
        Loading,
        Meshing,
        GpuReady,
        Visible,
        Cached,
    }

    /// <summary>
    /// Manages the lifecycle of all loaded sections.
    /// Coordinates between block data providers, meshing pipeline, GPU buffers,
    /// and the culling/rendering system.
    ///
    /// The ChunkManager does NOT own the block data or GPU buffers - it orchestrates
    /// the flow between subsystems that do.
    /// </summary>
    public class ChunkManager
    {
        private readonly Dictionary<SectionCoord, ManagedSection> _sections = new();
        private readonly VoxelEngineConfig _config;

        /// <summary>Total number of managed sections.</summary>
        public int SectionCount => _sections.Count;

        /// <summary>Number of sections in each state.</summary>
        public int LoadingCount { get; private set; }
        public int MeshingCount { get; private set; }
        public int GpuReadyCount { get; private set; }
        public int VisibleCount { get; private set; }
        public int CachedCount { get; private set; }

        public ChunkManager(VoxelEngineConfig config)
        {
            _config = config;
        }

        /// <summary>
        /// Request a section to be loaded. Transitions from UNLOADED to LOADING.
        /// Returns false if already loaded or at capacity.
        /// </summary>
        public bool RequestLoad(SectionCoord coord)
        {
            if (_sections.ContainsKey(coord))
                return false;

            if (_sections.Count >= _config.MaxLoadedSections)
                return false;

            _sections[coord] = new ManagedSection
            {
                Coord = coord,
                State = SectionState.Loading,
            };
            LoadingCount++;
            return true;
        }

        /// <summary>
        /// Mark a section's block data as loaded. Transitions LOADING -> MESHING.
        /// </summary>
        public bool MarkLoaded(SectionCoord coord, int[] blockData)
        {
            if (!_sections.TryGetValue(coord, out var section) || section.State != SectionState.Loading)
                return false;

            section.State = SectionState.Meshing;
            section.BlockData = blockData;
            LoadingCount--;
            MeshingCount++;
            return true;
        }

        /// <summary>
        /// Mark a section's mesh as uploaded to GPU. Transitions MESHING -> GPU_READY.
        /// </summary>
        public bool MarkMeshed(SectionCoord coord, int quadOffset, int quadCount)
        {
            if (!_sections.TryGetValue(coord, out var section) || section.State != SectionState.Meshing)
                return false;

            section.State = SectionState.GpuReady;
            section.QuadOffset = quadOffset;
            section.QuadCount = quadCount;
            MeshingCount--;
            GpuReadyCount++;
            return true;
        }

        /// <summary>
        /// Mark a section as visible (in the draw buffer). Transitions GPU_READY/CACHED -> VISIBLE.
        /// </summary>
        public bool MarkVisible(SectionCoord coord)
        {
            if (!_sections.TryGetValue(coord, out var section))
                return false;

            if (section.State == SectionState.GpuReady)
                GpuReadyCount--;
            else if (section.State == SectionState.Cached)
                CachedCount--;
            else
                return false;

            section.State = SectionState.Visible;
            VisibleCount++;
            return true;
        }

        /// <summary>
        /// Mark a section as cached (was visible, now outside view). Transitions VISIBLE -> CACHED.
        /// </summary>
        public bool MarkCached(SectionCoord coord)
        {
            if (!_sections.TryGetValue(coord, out var section) || section.State != SectionState.Visible)
                return false;

            section.State = SectionState.Cached;
            VisibleCount--;
            CachedCount++;
            return true;
        }

        /// <summary>
        /// Unload a section entirely. Transitions any state -> UNLOADED (removed from dictionary).
        /// </summary>
        public bool Unload(SectionCoord coord)
        {
            if (!_sections.TryGetValue(coord, out var section))
                return false;

            switch (section.State)
            {
                case SectionState.Loading: LoadingCount--; break;
                case SectionState.Meshing: MeshingCount--; break;
                case SectionState.GpuReady: GpuReadyCount--; break;
                case SectionState.Visible: VisibleCount--; break;
                case SectionState.Cached: CachedCount--; break;
            }

            _sections.Remove(coord);
            return true;
        }

        /// <summary>Get the state of a section, or null if not loaded.</summary>
        public SectionState? GetState(SectionCoord coord)
        {
            return _sections.TryGetValue(coord, out var s) ? s.State : null;
        }

        /// <summary>Get section data, or null if not loaded.</summary>
        public ManagedSection? GetSection(SectionCoord coord)
        {
            return _sections.TryGetValue(coord, out var s) ? s : null;
        }

        /// <summary>Get all sections in a specific state.</summary>
        public IEnumerable<ManagedSection> GetSectionsInState(SectionState state)
        {
            return _sections.Values.Where(s => s.State == state);
        }

        /// <summary>Get all managed sections.</summary>
        public IEnumerable<ManagedSection> AllSections => _sections.Values;
    }

    /// <summary>
    /// Internal tracking data for a managed section.
    /// </summary>
    public class ManagedSection
    {
        public SectionCoord Coord { get; set; }
        public SectionState State { get; set; }
        public int[]? BlockData { get; set; }
        public int QuadOffset { get; set; }
        public int QuadCount { get; set; }
        public int LodLevel { get; set; }
        public float DistanceSq { get; set; }
        public long Connectivity { get; set; } = ~0L; // default: fully connected
    }
}
