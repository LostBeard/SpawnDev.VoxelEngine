namespace SpawnDev.VoxelEngine.Buffers
{
    /// <summary>
    /// Fixed-size bucket allocator for GPU vertex data.
    /// Backed by a single large buffer divided into N equal buckets.
    /// O(1) alloc via pop from free stack, O(1) free via push to free stack.
    ///
    /// Design based on Nick McDonald's vertex pooling (nickmcd.me/2021/04/04/high-performance-voxel-engine/).
    /// Measured 45% frame time improvement over per-chunk VBOs.
    ///
    /// Each chunk uses 1-6 buckets (one per face orientation group for face masking).
    /// When the pool is full, eviction by priority score (distSq * vertexCount) frees buckets.
    /// </summary>
    public class VertexPool
    {
        /// <summary>Total number of buckets in the pool.</summary>
        public int BucketCount { get; }

        /// <summary>Size of each bucket in elements (e.g., floats or packed quads).</summary>
        public int BucketSize { get; }

        /// <summary>Total capacity in elements.</summary>
        public int TotalCapacity => BucketCount * BucketSize;

        /// <summary>Number of currently allocated buckets.</summary>
        public int AllocatedCount => BucketCount - _freeStack.Count;

        /// <summary>Number of available buckets.</summary>
        public int FreeCount => _freeStack.Count;

        /// <summary>Whether any buckets are available.</summary>
        public bool HasFree => _freeStack.Count > 0;

        // Free bucket indices - LIFO stack for cache locality
        private readonly Stack<int> _freeStack;

        // Tracks which chunk owns which buckets
        private readonly Dictionary<long, ChunkAllocation> _allocations;

        /// <summary>
        /// Create a vertex pool with the given number of fixed-size buckets.
        /// </summary>
        /// <param name="bucketCount">Number of buckets (derived from device maxBufferSize)</param>
        /// <param name="bucketSize">Elements per bucket (enough for worst-case chunk at target LOD)</param>
        public VertexPool(int bucketCount, int bucketSize)
        {
            BucketCount = bucketCount;
            BucketSize = bucketSize;
            _freeStack = new Stack<int>(bucketCount);
            _allocations = new Dictionary<long, ChunkAllocation>();

            // Initialize free stack (push in reverse so lowest indices pop first)
            for (int i = bucketCount - 1; i >= 0; i--)
            {
                _freeStack.Push(i);
            }
        }

        /// <summary>
        /// Allocate a bucket for a chunk. Returns the bucket index and byte offset into the pool buffer.
        /// Returns -1 if no buckets available (caller should evict then retry).
        /// </summary>
        public int Allocate(long chunkKey)
        {
            if (_freeStack.Count == 0) return -1;

            int bucketIndex = _freeStack.Pop();

            if (!_allocations.TryGetValue(chunkKey, out var allocation))
            {
                allocation = new ChunkAllocation { ChunkKey = chunkKey, BucketIndices = new List<int>() };
                _allocations[chunkKey] = allocation;
            }

            allocation.BucketIndices.Add(bucketIndex);
            return bucketIndex;
        }

        /// <summary>
        /// Allocate multiple buckets for a chunk (e.g., 6 for face masking groups).
        /// Returns array of bucket indices, or null if not enough free.
        /// </summary>
        public int[]? AllocateMultiple(long chunkKey, int count)
        {
            if (_freeStack.Count < count) return null;

            var indices = new int[count];
            for (int i = 0; i < count; i++)
            {
                indices[i] = _freeStack.Pop();
            }

            if (!_allocations.TryGetValue(chunkKey, out var allocation))
            {
                allocation = new ChunkAllocation { ChunkKey = chunkKey, BucketIndices = new List<int>() };
                _allocations[chunkKey] = allocation;
            }

            allocation.BucketIndices.AddRange(indices);
            return indices;
        }

        /// <summary>
        /// Free all buckets owned by a chunk. Returns them to the pool.
        /// </summary>
        public bool Free(long chunkKey)
        {
            if (!_allocations.TryGetValue(chunkKey, out var allocation)) return false;

            foreach (var idx in allocation.BucketIndices)
            {
                _freeStack.Push(idx);
            }

            _allocations.Remove(chunkKey);
            return true;
        }

        /// <summary>
        /// Get the element offset for a bucket index (for use in draw commands).
        /// </summary>
        public int GetBucketOffset(int bucketIndex) => bucketIndex * BucketSize;

        /// <summary>
        /// Get all bucket indices allocated to a chunk.
        /// </summary>
        public IReadOnlyList<int>? GetChunkBuckets(long chunkKey)
        {
            return _allocations.TryGetValue(chunkKey, out var allocation) ? allocation.BucketIndices : null;
        }

        /// <summary>
        /// Check if a chunk has an allocation.
        /// </summary>
        public bool HasAllocation(long chunkKey) => _allocations.ContainsKey(chunkKey);

        /// <summary>
        /// Get all allocated chunk keys (for eviction scoring).
        /// </summary>
        public IEnumerable<long> AllocatedChunks => _allocations.Keys;

        /// <summary>
        /// Get allocation info for a chunk.
        /// </summary>
        public ChunkAllocation? GetAllocation(long chunkKey)
        {
            return _allocations.TryGetValue(chunkKey, out var allocation) ? allocation : null;
        }

        /// <summary>
        /// Evict the chunk with the highest eviction score.
        /// Score function takes chunkKey and returns priority (higher = evict first).
        /// Returns the evicted chunkKey, or null if pool is empty.
        /// </summary>
        public long? EvictHighestScore(Func<long, float> scoreFunction)
        {
            if (_allocations.Count == 0) return null;

            long bestKey = 0;
            float bestScore = float.MinValue;
            bool found = false;

            foreach (var kvp in _allocations)
            {
                float score = scoreFunction(kvp.Key);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestKey = kvp.Key;
                    found = true;
                }
            }

            if (!found) return null;

            Free(bestKey);
            return bestKey;
        }

        /// <summary>
        /// Evict chunks until at least 'needed' buckets are free.
        /// Returns count of chunks evicted.
        /// </summary>
        public int EvictUntilFree(int needed, Func<long, float> scoreFunction, int maxEvictions = 10)
        {
            int evicted = 0;
            while (_freeStack.Count < needed && evicted < maxEvictions && _allocations.Count > 0)
            {
                var key = EvictHighestScore(scoreFunction);
                if (key == null) break;
                evicted++;
            }
            return evicted;
        }

        /// <summary>
        /// Create a standard eviction score function using camera position.
        /// Score = distanceSq * bucketCount (far chunks with many buckets evict first).
        /// </summary>
        public static Func<long, float> CreateDistanceScorer(
            float cameraX, float cameraZ,
            Func<long, (float x, float z)> chunkCenterFunc,
            VertexPool pool)
        {
            return chunkKey =>
            {
                var (cx, cz) = chunkCenterFunc(chunkKey);
                float dx = cx - cameraX;
                float dz = cz - cameraZ;
                float distSq = dx * dx + dz * dz;
                int buckets = pool.GetAllocation(chunkKey)?.BucketIndices.Count ?? 1;
                return distSq * buckets;
            };
        }
    }

    /// <summary>
    /// Tracks which buckets are allocated to a chunk.
    /// </summary>
    public class ChunkAllocation
    {
        public long ChunkKey { get; set; }
        public List<int> BucketIndices { get; set; } = new();
    }
}
