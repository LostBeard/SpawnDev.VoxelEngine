namespace SpawnDev.VoxelEngine.Caching
{
    /// <summary>
    /// Interface for chunk data persistence.
    /// Consumers implement this to provide block data from their storage backend
    /// (network, file system, IndexedDB, procedural generation).
    ///
    /// The VoxelEngine calls LoadSectionAsync when a section enters draw distance
    /// and SaveSectionAsync when a modified section is evicted from memory.
    ///
    /// Implementations:
    /// - AubsCraft: network fetch from Minecraft server via SignalR
    /// - Lost Spawns: IndexedDB cache + procedural heightmap generation
    /// - Demo/tests: in-memory dictionary
    /// </summary>
    public interface IChunkCacheProvider
    {
        /// <summary>
        /// Load block data for a section. Returns null if the section doesn't exist
        /// (e.g., beyond world border). The engine will treat null as all-air.
        /// </summary>
        /// <param name="coord">Section coordinates to load.</param>
        /// <returns>Block data in PackedBlock format (flat array, sizeXZ * sizeXZ * sizeY), or null.</returns>
        Task<int[]?> LoadSectionAsync(SectionCoord coord);

        /// <summary>
        /// Save modified block data for a section.
        /// Called when a section with modifications is evicted from memory,
        /// or periodically for auto-save.
        /// </summary>
        /// <param name="coord">Section coordinates.</param>
        /// <param name="blocks">Block data to persist.</param>
        Task SaveSectionAsync(SectionCoord coord, int[] blocks);

        /// <summary>
        /// Check if a section exists without loading it.
        /// Used for world border detection and draw distance optimization.
        /// </summary>
        /// <param name="coord">Section coordinates.</param>
        /// <returns>True if the section has data (even if all air).</returns>
        Task<bool> ExistsAsync(SectionCoord coord);

        /// <summary>
        /// Batch-load multiple sections. Default implementation loads sequentially.
        /// Override for backends that support batch queries (database, network batch).
        /// </summary>
        async Task<Dictionary<SectionCoord, int[]>> LoadBatchAsync(IEnumerable<SectionCoord> coords)
        {
            var result = new Dictionary<SectionCoord, int[]>();
            foreach (var coord in coords)
            {
                var data = await LoadSectionAsync(coord);
                if (data != null) result[coord] = data;
            }
            return result;
        }
    }

    /// <summary>
    /// In-memory chunk cache for testing and demos.
    /// No persistence - data is lost when the process exits.
    /// </summary>
    public class InMemoryChunkCache : IChunkCacheProvider
    {
        private readonly Dictionary<SectionCoord, int[]> _data = new();

        public Task<int[]?> LoadSectionAsync(SectionCoord coord)
        {
            _data.TryGetValue(coord, out var data);
            return Task.FromResult(data);
        }

        public Task SaveSectionAsync(SectionCoord coord, int[] blocks)
        {
            _data[coord] = blocks;
            return Task.CompletedTask;
        }

        public Task<bool> ExistsAsync(SectionCoord coord)
        {
            return Task.FromResult(_data.ContainsKey(coord));
        }

        /// <summary>Pre-populate sections for testing.</summary>
        public void Set(SectionCoord coord, int[] blocks) => _data[coord] = blocks;

        /// <summary>Number of cached sections.</summary>
        public int Count => _data.Count;

        /// <summary>Clear all cached data.</summary>
        public void Clear() => _data.Clear();
    }
}
