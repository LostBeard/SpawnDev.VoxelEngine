using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Buffers;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // IndirectDrawBuffer tests: O(1) add/remove, end-swap integrity, capacity limits.
    // These are CPU-only tests (no GPU needed) - the buffer is a CPU-side data structure
    // that gets uploaded to GPU each frame.
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>
        /// Test 1: Add commands, verify count and data integrity.
        /// </summary>
        [TestMethod]
        public void IndirectDraw_AddCommands_CountAndDataCorrectTest()
        {
            var buf = new IndirectDrawBuffer(100);

            var s0 = new SectionCoord(0, 0, 0);
            var s1 = new SectionCoord(1, 0, 0);
            var s2 = new SectionCoord(0, 1, 0);

            int slot0 = buf.Add(s0, quadOffset: 0, quadCount: 50, sectionIndex: 0);
            int slot1 = buf.Add(s1, quadOffset: 50, quadCount: 100, sectionIndex: 1);
            int slot2 = buf.Add(s2, quadOffset: 150, quadCount: 25, sectionIndex: 2);

            if (buf.Count != 3)
                throw new Exception($"IndirectDraw_Add: expected count 3, got {buf.Count}");
            if (slot0 != 0 || slot1 != 1 || slot2 != 2)
                throw new Exception($"IndirectDraw_Add: unexpected slots {slot0},{slot1},{slot2}");

            // Verify command data
            var cmd0 = buf.GetCommandAt(0);
            if (cmd0.VertexCount != 300 || cmd0.FirstVertex != 0)
                throw new Exception($"IndirectDraw_Add: cmd0 vertices={cmd0.VertexCount} first={cmd0.FirstVertex}, expected 300/0");

            var cmd1 = buf.GetCommandAt(1);
            if (cmd1.VertexCount != 600 || cmd1.FirstVertex != 300)
                throw new Exception($"IndirectDraw_Add: cmd1 vertices={cmd1.VertexCount} first={cmd1.FirstVertex}, expected 600/300");
        }

        /// <summary>
        /// Test 2: Remove middle entry - end-swap preserves data integrity.
        /// </summary>
        [TestMethod]
        public void IndirectDraw_RemoveMiddle_EndSwapIntegrityTest()
        {
            var buf = new IndirectDrawBuffer(100);

            var s0 = new SectionCoord(0, 0, 0);
            var s1 = new SectionCoord(1, 0, 0);
            var s2 = new SectionCoord(2, 0, 0);

            buf.Add(s0, 0, 10, 0);   // slot 0: 10 quads
            buf.Add(s1, 10, 20, 1);  // slot 1: 20 quads (will be removed)
            buf.Add(s2, 30, 30, 2);  // slot 2: 30 quads (will be swapped to slot 1)

            // Remove s1 (middle) - s2 should move to slot 1
            bool removed = buf.Remove(s1);
            if (!removed)
                throw new Exception("IndirectDraw_RemoveMiddle: Remove returned false");
            if (buf.Count != 2)
                throw new Exception($"IndirectDraw_RemoveMiddle: expected count 2, got {buf.Count}");

            // s1 should no longer be present
            if (buf.Contains(s1))
                throw new Exception("IndirectDraw_RemoveMiddle: removed section still present");

            // s2 should now be at slot 1 (swapped from slot 2)
            int s2Slot = buf.GetSlot(s2);
            if (s2Slot != 1)
                throw new Exception($"IndirectDraw_RemoveMiddle: s2 at slot {s2Slot}, expected 1 (end-swap)");

            // s2's command data should be intact after swap
            var cmd = buf.GetCommandAt(1);
            if (cmd.VertexCount != 180) // 30 quads * 6 verts
                throw new Exception($"IndirectDraw_RemoveMiddle: swapped cmd vertices={cmd.VertexCount}, expected 180");

            // s0 should be unchanged
            if (buf.GetSlot(s0) != 0)
                throw new Exception("IndirectDraw_RemoveMiddle: s0 slot changed unexpectedly");
        }

        /// <summary>
        /// Test 3: Remove last entry - no swap needed.
        /// </summary>
        [TestMethod]
        public void IndirectDraw_RemoveLast_NoSwapTest()
        {
            var buf = new IndirectDrawBuffer(100);

            var s0 = new SectionCoord(0, 0, 0);
            var s1 = new SectionCoord(1, 0, 0);

            buf.Add(s0, 0, 10, 0);
            buf.Add(s1, 10, 20, 1);

            buf.Remove(s1); // remove last - no swap
            if (buf.Count != 1)
                throw new Exception($"IndirectDraw_RemoveLast: expected count 1, got {buf.Count}");
            if (buf.GetSlot(s0) != 0)
                throw new Exception("IndirectDraw_RemoveLast: s0 slot changed");
        }

        /// <summary>
        /// Test 4: Capacity limit - returns -1 when full.
        /// </summary>
        [TestMethod]
        public void IndirectDraw_CapacityFull_RejectsAddTest()
        {
            var buf = new IndirectDrawBuffer(3);

            buf.Add(new SectionCoord(0, 0, 0), 0, 10, 0);
            buf.Add(new SectionCoord(1, 0, 0), 10, 10, 1);
            buf.Add(new SectionCoord(2, 0, 0), 20, 10, 2);

            int result = buf.Add(new SectionCoord(3, 0, 0), 30, 10, 3);
            if (result != -1)
                throw new Exception($"IndirectDraw_CapacityFull: expected -1, got {result}");
            if (buf.Count != 3)
                throw new Exception($"IndirectDraw_CapacityFull: count should still be 3, got {buf.Count}");
        }

        /// <summary>
        /// Test 5: Duplicate add updates in place.
        /// </summary>
        [TestMethod]
        public void IndirectDraw_DuplicateAdd_UpdatesInPlaceTest()
        {
            var buf = new IndirectDrawBuffer(100);

            var s0 = new SectionCoord(5, 3, 7);
            buf.Add(s0, 0, 10, 0);

            // Add same section again with different data
            int slot = buf.Add(s0, 100, 50, 42);
            if (buf.Count != 1)
                throw new Exception($"IndirectDraw_DuplicateAdd: expected count 1, got {buf.Count}");

            var cmd = buf.GetCommandAt(slot);
            if (cmd.VertexCount != 300 || cmd.FirstVertex != 600)
                throw new Exception($"IndirectDraw_DuplicateAdd: expected 300 verts at 600, got {cmd.VertexCount} at {cmd.FirstVertex}");
        }

        /// <summary>
        /// Test 6: Add, remove all, add again - clean state.
        /// </summary>
        [TestMethod]
        public void IndirectDraw_AddRemoveAdd_CleanStateTest()
        {
            var buf = new IndirectDrawBuffer(100);

            for (int i = 0; i < 10; i++)
                buf.Add(new SectionCoord(i, 0, 0), i * 10, 10, i);

            if (buf.Count != 10)
                throw new Exception($"IndirectDraw_AddRemoveAdd: expected 10, got {buf.Count}");

            // Remove all
            for (int i = 0; i < 10; i++)
                buf.Remove(new SectionCoord(i, 0, 0));

            if (buf.Count != 0)
                throw new Exception($"IndirectDraw_AddRemoveAdd: expected 0 after remove all, got {buf.Count}");

            // Add new entries
            buf.Add(new SectionCoord(99, 99, 99), 0, 5, 0);
            if (buf.Count != 1)
                throw new Exception($"IndirectDraw_AddRemoveAdd: expected 1 after re-add, got {buf.Count}");

            var cmd = buf.GetCommandAt(0);
            if (cmd.VertexCount != 30)
                throw new Exception($"IndirectDraw_AddRemoveAdd: expected 30 verts, got {cmd.VertexCount}");
        }

        /// <summary>
        /// Test 7: Commands span is correct for GPU upload.
        /// </summary>
        [TestMethod]
        public void IndirectDraw_CommandsSpan_CorrectSizeTest()
        {
            var buf = new IndirectDrawBuffer(100);

            buf.Add(new SectionCoord(0, 0, 0), 0, 10, 0);
            buf.Add(new SectionCoord(1, 0, 0), 10, 20, 1);
            buf.Add(new SectionCoord(2, 0, 0), 30, 30, 2);

            var span = buf.Commands;
            if (span.Length != 3)
                throw new Exception($"IndirectDraw_CommandsSpan: expected length 3, got {span.Length}");

            // Verify ActiveSizeBytes
            int expectedBytes = 3 * IndirectDrawBuffer.DrawCommandSizeBytes; // 3 * 16 = 48
            if (buf.ActiveSizeBytes != expectedBytes)
                throw new Exception($"IndirectDraw_CommandsSpan: expected {expectedBytes} bytes, got {buf.ActiveSizeBytes}");
        }

        /// <summary>
        /// Test 8: Stress test - many adds and removes maintain consistency.
        /// </summary>
        [TestMethod]
        public void IndirectDraw_StressAddRemove_ConsistencyTest()
        {
            var buf = new IndirectDrawBuffer(256);

            // Add 200 sections
            for (int i = 0; i < 200; i++)
                buf.Add(new SectionCoord(i % 16, i / 16, 0), i * 5, 5, i);

            if (buf.Count != 200)
                throw new Exception($"IndirectDraw_Stress: expected 200, got {buf.Count}");

            // Remove every other section (100 removes, each triggers end-swap)
            for (int i = 0; i < 200; i += 2)
                buf.Remove(new SectionCoord(i % 16, i / 16, 0));

            if (buf.Count != 100)
                throw new Exception($"IndirectDraw_Stress: expected 100 after removes, got {buf.Count}");

            // Verify all remaining sections are findable and have correct data
            for (int i = 1; i < 200; i += 2)
            {
                var coord = new SectionCoord(i % 16, i / 16, 0);
                if (!buf.Contains(coord))
                    throw new Exception($"IndirectDraw_Stress: section ({coord}) missing after removes");

                int slot = buf.GetSlot(coord);
                var cmd = buf.GetCommandAt(slot);
                if (cmd.VertexCount != 30) // 5 quads * 6
                    throw new Exception($"IndirectDraw_Stress: section ({coord}) verts={cmd.VertexCount}, expected 30");
            }

            // Verify no removed sections remain
            for (int i = 0; i < 200; i += 2)
            {
                var coord = new SectionCoord(i % 16, i / 16, 0);
                if (buf.Contains(coord))
                    throw new Exception($"IndirectDraw_Stress: removed section ({coord}) still present");
            }
        }
    }
}
