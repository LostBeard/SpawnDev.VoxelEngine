using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Buffers;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // VertexPool tests - pure data structure, no GPU needed
    public partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task VertexPool_AllocAndFree_Roundtrip() => await RunTest(async accelerator =>
        {
            var pool = new VertexPool(100, 1024);
            if (pool.FreeCount != 100) throw new Exception($"Expected 100 free, got {pool.FreeCount}");

            int bucket = pool.Allocate(1);
            if (bucket < 0) throw new Exception("Allocate returned -1 on non-full pool");
            if (pool.FreeCount != 99) throw new Exception($"After alloc: expected 99 free, got {pool.FreeCount}");
            if (pool.AllocatedCount != 1) throw new Exception($"Expected 1 allocated, got {pool.AllocatedCount}");

            pool.Free(1);
            if (pool.FreeCount != 100) throw new Exception($"After free: expected 100 free, got {pool.FreeCount}");
            if (pool.AllocatedCount != 0) throw new Exception($"Expected 0 allocated, got {pool.AllocatedCount}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VertexPool_AllocMultiple_6BucketsPerChunk() => await RunTest(async accelerator =>
        {
            var pool = new VertexPool(100, 1024);

            // Allocate 6 buckets for face masking (one per face direction)
            var buckets = pool.AllocateMultiple(42, 6);
            if (buckets == null) throw new Exception("AllocateMultiple returned null");
            if (buckets.Length != 6) throw new Exception($"Expected 6 buckets, got {buckets.Length}");
            if (pool.FreeCount != 94) throw new Exception($"Expected 94 free, got {pool.FreeCount}");

            // All bucket indices should be unique
            var unique = new HashSet<int>(buckets);
            if (unique.Count != 6) throw new Exception("Bucket indices are not unique");

            // GetChunkBuckets should return the same indices
            var retrieved = pool.GetChunkBuckets(42);
            if (retrieved == null || retrieved.Count != 6)
                throw new Exception($"GetChunkBuckets returned {retrieved?.Count ?? 0}, expected 6");

            // Free returns all 6
            pool.Free(42);
            if (pool.FreeCount != 100) throw new Exception($"After free: expected 100 free, got {pool.FreeCount}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VertexPool_ExhaustAndRecover() => await RunTest(async accelerator =>
        {
            var pool = new VertexPool(10, 512);

            // Exhaust all buckets
            for (int i = 0; i < 10; i++)
            {
                int bucket = pool.Allocate(i);
                if (bucket < 0) throw new Exception($"Alloc failed at iteration {i}");
            }

            if (pool.HasFree) throw new Exception("Pool should be empty");

            // Next alloc should fail
            int fail = pool.Allocate(99);
            if (fail != -1) throw new Exception($"Expected -1 from full pool, got {fail}");

            // Free one and alloc again
            pool.Free(5);
            if (!pool.HasFree) throw new Exception("Pool should have 1 free after freeing chunk 5");

            int recovered = pool.Allocate(99);
            if (recovered < 0) throw new Exception("Alloc should succeed after free");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VertexPool_Eviction_FarthestChunkFirst() => await RunTest(async accelerator =>
        {
            var pool = new VertexPool(10, 512);

            // Allocate chunks at known distances from origin
            // Chunk key encodes position: key = cx * 1000 + cz
            pool.Allocate(2_002); // chunk at (2, 2), dist^2 = 8
            pool.Allocate(5_005); // chunk at (5, 5), dist^2 = 50
            pool.Allocate(1_001); // chunk at (1, 1), dist^2 = 2
            pool.Allocate(10_010); // chunk at (10, 10), dist^2 = 200

            // Evict the farthest chunk (camera at origin)
            var scorer = VertexPool.CreateDistanceScorer(0, 0,
                key => ((float)(key / 1000), (float)(key % 1000)), pool);

            var evicted = pool.EvictHighestScore(scorer);
            if (evicted != 10_010)
                throw new Exception($"Expected chunk 10010 (farthest) to be evicted, got {evicted}");

            // Next eviction should be (5,5)
            evicted = pool.EvictHighestScore(scorer);
            if (evicted != 5_005)
                throw new Exception($"Expected chunk 5005 to be evicted next, got {evicted}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VertexPool_EvictUntilFree_MultipleEvictions() => await RunTest(async accelerator =>
        {
            var pool = new VertexPool(10, 512);

            // Fill pool: 5 chunks, 2 buckets each
            for (int i = 0; i < 5; i++)
            {
                pool.AllocateMultiple(i, 2);
            }

            if (pool.FreeCount != 0) throw new Exception("Pool should be full");

            // Need 4 free buckets - should evict 2 chunks (2 buckets each)
            var scorer = VertexPool.CreateDistanceScorer(0, 0,
                key => ((float)(key + 1) * 10, (float)(key + 1) * 10), pool);

            int evictCount = pool.EvictUntilFree(4, scorer);
            if (evictCount < 2) throw new Exception($"Expected at least 2 evictions, got {evictCount}");
            if (pool.FreeCount < 4) throw new Exception($"Expected at least 4 free, got {pool.FreeCount}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VertexPool_GetBucketOffset_Correct() => await RunTest(async accelerator =>
        {
            var pool = new VertexPool(10, 1024);

            int bucket0 = pool.Allocate(1);
            int bucket1 = pool.Allocate(2);

            int offset0 = pool.GetBucketOffset(bucket0);
            int offset1 = pool.GetBucketOffset(bucket1);

            // Offsets should be bucket_index * bucket_size
            if (offset0 != bucket0 * 1024)
                throw new Exception($"Offset0: expected {bucket0 * 1024}, got {offset0}");
            if (offset1 != bucket1 * 1024)
                throw new Exception($"Offset1: expected {bucket1 * 1024}, got {offset1}");

            // Offsets should not overlap
            if (Math.Abs(offset1 - offset0) < 1024)
                throw new Exception($"Offsets overlap: {offset0} and {offset1}, bucket size 1024");

            await Task.CompletedTask;
        });
    }
}
