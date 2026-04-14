using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Buffers;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Buffer compaction tests: plan correctness, GPU copy integrity, offset updates.
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>
        /// Test 1: Plan with no gaps - compaction not needed.
        /// </summary>
        [TestMethod]
        public void Compaction_NoGaps_NotNeededTest()
        {
            var compaction = new BufferCompaction();
            var entries = new[]
            {
                new SectionEntry { Coord = new SectionCoord(0,0,0), QuadOffset = 0, QuadCount = 100 },
                new SectionEntry { Coord = new SectionCoord(1,0,0), QuadOffset = 100, QuadCount = 50 },
                new SectionEntry { Coord = new SectionCoord(2,0,0), QuadOffset = 150, QuadCount = 75 },
            };

            var plan = compaction.Plan(entries, totalBufferUsed: 225);

            if (plan.ShouldCompact)
                throw new Exception("Compaction_NoGaps: should not need compaction (no gaps)");
            if (plan.MovesRequired != 0)
                throw new Exception($"Compaction_NoGaps: expected 0 moves, got {plan.MovesRequired}");
            if (plan.QuadsFreed != 0)
                throw new Exception($"Compaction_NoGaps: expected 0 freed, got {plan.QuadsFreed}");
        }

        /// <summary>
        /// Test 2: Plan with gap in the middle - segments should shift left.
        /// </summary>
        [TestMethod]
        public void Compaction_MiddleGap_CorrectPlanTest()
        {
            var compaction = new BufferCompaction { FragmentationThreshold = 0.1f };
            // Section at 0-99, gap at 100-199 (removed section), section at 200-249
            var entries = new[]
            {
                new SectionEntry { Coord = new SectionCoord(0,0,0), QuadOffset = 0, QuadCount = 100 },
                new SectionEntry { Coord = new SectionCoord(2,0,0), QuadOffset = 200, QuadCount = 50 },
            };

            var plan = compaction.Plan(entries, totalBufferUsed: 250);

            if (!plan.ShouldCompact)
                throw new Exception("Compaction_MiddleGap: should need compaction (40% fragmented)");
            if (plan.CompactedSize != 150)
                throw new Exception($"Compaction_MiddleGap: compacted size {plan.CompactedSize}, expected 150");
            if (plan.QuadsFreed != 100)
                throw new Exception($"Compaction_MiddleGap: freed {plan.QuadsFreed}, expected 100");

            // First segment stays at 0, second moves from 200 to 100
            if (plan.Segments[0].DestOffset != 0)
                throw new Exception($"Compaction_MiddleGap: seg0 dest={plan.Segments[0].DestOffset}, expected 0");
            if (plan.Segments[1].DestOffset != 100)
                throw new Exception($"Compaction_MiddleGap: seg1 dest={plan.Segments[1].DestOffset}, expected 100");
        }

        /// <summary>
        /// Test 3: Plan with leading gap - all segments shift left.
        /// </summary>
        [TestMethod]
        public void Compaction_LeadingGap_AllShiftTest()
        {
            var compaction = new BufferCompaction { FragmentationThreshold = 0.1f };
            // Gap at 0-99 (removed first section), sections at 100-149 and 150-199
            var entries = new[]
            {
                new SectionEntry { Coord = new SectionCoord(1,0,0), QuadOffset = 100, QuadCount = 50 },
                new SectionEntry { Coord = new SectionCoord(2,0,0), QuadOffset = 150, QuadCount = 50 },
            };

            var plan = compaction.Plan(entries, totalBufferUsed: 200);

            if (!plan.ShouldCompact)
                throw new Exception("Compaction_LeadingGap: should need compaction");
            if (plan.MovesRequired != 2)
                throw new Exception($"Compaction_LeadingGap: expected 2 moves, got {plan.MovesRequired}");
            if (plan.Segments[0].DestOffset != 0)
                throw new Exception($"Compaction_LeadingGap: seg0 dest={plan.Segments[0].DestOffset}, expected 0");
            if (plan.Segments[1].DestOffset != 50)
                throw new Exception($"Compaction_LeadingGap: seg1 dest={plan.Segments[1].DestOffset}, expected 50");
        }

        /// <summary>
        /// Test 4: GPU compaction - verify data integrity after GPU copy.
        /// Creates a buffer with known data, introduces a gap, compacts, verifies.
        /// </summary>
        [TestMethod]
        public async Task Compaction_GpuCopy_DataIntegrityTest() => await RunTest(async accelerator =>
        {
            // Create buffer: [seg0: 10 quads] [gap: 10] [seg1: 10 quads]
            int bufferSize = 30;
            long[] initialData = new long[bufferSize];

            // Seg0 at offset 0: quads with value 100+i
            for (int i = 0; i < 10; i++)
                initialData[i] = 100 + i;

            // Gap at offset 10-19 (old data, will be overwritten)
            for (int i = 10; i < 20; i++)
                initialData[i] = -1; // garbage

            // Seg1 at offset 20: quads with value 200+i
            for (int i = 0; i < 10; i++)
                initialData[20 + i] = 200 + i;

            using var gpuBuffer = accelerator.Allocate1D(initialData);

            // Plan: seg0 stays at 0, seg1 moves from 20 to 10
            var entries = new[]
            {
                new SectionEntry { Coord = new SectionCoord(0,0,0), QuadOffset = 0, QuadCount = 10 },
                new SectionEntry { Coord = new SectionCoord(1,0,0), QuadOffset = 20, QuadCount = 10 },
            };

            var compaction = new BufferCompaction { FragmentationThreshold = 0.1f };
            var plan = compaction.Plan(entries, totalBufferUsed: 30);

            if (!plan.ShouldCompact)
                throw new Exception("Compaction_GpuCopy: plan should trigger compaction");

            // Execute GPU compaction
            await BufferCompaction.ExecuteAsync(accelerator, gpuBuffer, plan);

            // Read back and verify
            var result = await gpuBuffer.CopyToHostAsync<long>();

            // Seg0 (0-9): should be unchanged
            for (int i = 0; i < 10; i++)
            {
                if (result[i] != 100 + i)
                    throw new Exception($"Compaction_GpuCopy: seg0[{i}] = {result[i]}, expected {100 + i}");
            }

            // Seg1 (10-19): should now contain values 200+i (moved from 20-29)
            for (int i = 0; i < 10; i++)
            {
                if (result[10 + i] != 200 + i)
                    throw new Exception($"Compaction_GpuCopy: seg1[{i}] at offset {10 + i} = {result[10 + i]}, expected {200 + i}");
            }
        });

        /// <summary>
        /// Test 5: Fragmentation calculation.
        /// </summary>
        [TestMethod]
        public void Compaction_FragmentationCalc_CorrectTest()
        {
            var entries = new[]
            {
                new SectionEntry { QuadCount = 50 },
                new SectionEntry { QuadCount = 30 },
            };
            // 80 live quads in 200 buffer = 60% fragmented
            float frag = BufferCompaction.GetFragmentation(entries, 200);
            if (MathF.Abs(frag - 0.6f) > 0.01f)
                throw new Exception($"Compaction_FragCalc: expected 0.6, got {frag}");

            // 80 live quads in 80 buffer = 0% fragmented
            float frag2 = BufferCompaction.GetFragmentation(entries, 80);
            if (MathF.Abs(frag2) > 0.01f)
                throw new Exception($"Compaction_FragCalc: expected 0.0, got {frag2}");
        }

        /// <summary>
        /// Test 6: Multiple gaps - verify all segments compact correctly.
        /// </summary>
        [TestMethod]
        public async Task Compaction_MultipleGaps_AllCompactedTest() => await RunTest(async accelerator =>
        {
            // Layout: [A:5] [gap:5] [B:5] [gap:10] [C:5] = 30 total, 15 live
            int bufferSize = 30;
            long[] data = new long[bufferSize];

            for (int i = 0; i < 5; i++) data[i] = 1000 + i;        // A at 0
            for (int i = 0; i < 5; i++) data[10 + i] = 2000 + i;   // B at 10
            for (int i = 0; i < 5; i++) data[25 + i] = 3000 + i;   // C at 25

            using var gpuBuffer = accelerator.Allocate1D(data);

            var entries = new[]
            {
                new SectionEntry { Coord = new SectionCoord(0,0,0), QuadOffset = 0, QuadCount = 5 },
                new SectionEntry { Coord = new SectionCoord(1,0,0), QuadOffset = 10, QuadCount = 5 },
                new SectionEntry { Coord = new SectionCoord(2,0,0), QuadOffset = 25, QuadCount = 5 },
            };

            var compaction = new BufferCompaction { FragmentationThreshold = 0.1f };
            var plan = compaction.Plan(entries, 30);

            if (plan.CompactedSize != 15)
                throw new Exception($"Compaction_MultiGap: compacted {plan.CompactedSize}, expected 15");

            await BufferCompaction.ExecuteAsync(accelerator, gpuBuffer, plan);

            var result = await gpuBuffer.CopyToHostAsync<long>();

            // A at 0-4, B at 5-9, C at 10-14
            for (int i = 0; i < 5; i++)
            {
                if (result[i] != 1000 + i)
                    throw new Exception($"Compaction_MultiGap: A[{i}] = {result[i]}, expected {1000 + i}");
                if (result[5 + i] != 2000 + i)
                    throw new Exception($"Compaction_MultiGap: B[{i}] = {result[5 + i]}, expected {2000 + i}");
                if (result[10 + i] != 3000 + i)
                    throw new Exception($"Compaction_MultiGap: C[{i}] = {result[10 + i]}, expected {3000 + i}");
            }

            // Verify offset updates
            var offsets = new Dictionary<SectionCoord, int>();
            BufferCompaction.ApplyOffsets(plan, (coord, newOffset) => offsets[coord] = newOffset);

            // A stays at 0 (no move), B moves to 5, C moves to 10
            if (offsets.ContainsKey(new SectionCoord(0, 0, 0)))
                throw new Exception("Compaction_MultiGap: A should not have moved");
            if (!offsets.TryGetValue(new SectionCoord(1, 0, 0), out int bOffset) || bOffset != 5)
                throw new Exception($"Compaction_MultiGap: B offset {bOffset}, expected 5");
            if (!offsets.TryGetValue(new SectionCoord(2, 0, 0), out int cOffset) || cOffset != 10)
                throw new Exception($"Compaction_MultiGap: C offset {cOffset}, expected 10");
        });
    }
}
