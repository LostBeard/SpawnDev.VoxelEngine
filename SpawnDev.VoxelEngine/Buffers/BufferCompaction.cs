using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.VoxelEngine.Buffers
{
    /// <summary>
    /// GPU-accelerated buffer compaction for the shared PackedQuad buffer.
    ///
    /// When sections are unloaded, their quad data leaves gaps in the buffer.
    /// Compaction closes these gaps by copying live segments to the front,
    /// then updates all SectionEntry.QuadOffset values to reflect new positions.
    ///
    /// Triggered when fragmentation exceeds a threshold (wasted space / total used > 30%).
    ///
    /// Algorithm:
    /// 1. Sort live segments by current offset (already ordered by insertion)
    /// 2. Compute compacted offsets via exclusive prefix sum of segment sizes
    /// 3. GPU kernel copies each segment from old offset to new offset
    /// 4. Update SectionEntry.QuadOffset for each moved section
    ///
    /// The copy kernel is dispatched on GPU - one thread per quad in the moved segment.
    /// For large buffers (1M+ quads), this is significantly faster than CPU memcpy.
    /// </summary>
    public class BufferCompaction
    {
        /// <summary>
        /// Describes a live segment in the quad buffer that needs to be compacted.
        /// </summary>
        public struct Segment
        {
            /// <summary>Current offset in the quad buffer.</summary>
            public int SourceOffset;

            /// <summary>Number of quads in this segment.</summary>
            public int Count;

            /// <summary>Section coordinate that owns this segment.</summary>
            public SectionCoord Owner;

            /// <summary>New offset after compaction (computed by Plan).</summary>
            public int DestOffset;
        }

        /// <summary>
        /// Result of planning a compaction pass.
        /// Contains the segments to move and the new total used size.
        /// </summary>
        public class CompactionPlan
        {
            /// <summary>Segments sorted by source offset, with dest offsets computed.</summary>
            public Segment[] Segments { get; init; } = Array.Empty<Segment>();

            /// <summary>Number of segments that actually need to move (destOffset != sourceOffset).</summary>
            public int MovesRequired { get; init; }

            /// <summary>Total quads after compaction (no gaps).</summary>
            public int CompactedSize { get; init; }

            /// <summary>Bytes freed by compaction.</summary>
            public int QuadsFreed { get; init; }

            /// <summary>Whether compaction is worth doing (enough fragmentation).</summary>
            public bool ShouldCompact { get; init; }
        }

        /// <summary>Fragmentation threshold above which compaction is triggered (0-1).</summary>
        public float FragmentationThreshold { get; set; } = 0.3f;

        /// <summary>
        /// Plan a compaction pass. Analyzes the current section entries and determines
        /// which segments need to move and where.
        ///
        /// This is CPU-side planning. The actual data movement happens on GPU.
        /// </summary>
        /// <param name="entries">All loaded section entries with current QuadOffset/QuadCount.</param>
        /// <param name="totalBufferUsed">Current high-water mark of the quad buffer.</param>
        public CompactionPlan Plan(IEnumerable<SectionEntry> entries, int totalBufferUsed)
        {
            // Build sorted segment list
            var segments = entries
                .Where(e => e.QuadCount > 0)
                .Select(e => new Segment
                {
                    SourceOffset = e.QuadOffset,
                    Count = e.QuadCount,
                    Owner = e.Coord,
                    DestOffset = 0,
                })
                .OrderBy(s => s.SourceOffset)
                .ToArray();

            if (segments.Length == 0)
            {
                return new CompactionPlan
                {
                    Segments = segments,
                    MovesRequired = 0,
                    CompactedSize = 0,
                    QuadsFreed = 0,
                    ShouldCompact = false,
                };
            }

            // Compute compacted offsets (exclusive prefix sum of segment sizes)
            int cursor = 0;
            for (int i = 0; i < segments.Length; i++)
            {
                segments[i].DestOffset = cursor;
                cursor += segments[i].Count;
            }

            int compactedSize = cursor;
            int quadsFreed = totalBufferUsed - compactedSize;
            int movesRequired = 0;

            for (int i = 0; i < segments.Length; i++)
            {
                if (segments[i].DestOffset != segments[i].SourceOffset)
                    movesRequired++;
            }

            float fragmentation = totalBufferUsed > 0
                ? (float)quadsFreed / totalBufferUsed
                : 0f;

            return new CompactionPlan
            {
                Segments = segments,
                MovesRequired = movesRequired,
                CompactedSize = compactedSize,
                QuadsFreed = quadsFreed,
                ShouldCompact = fragmentation >= FragmentationThreshold && movesRequired > 0,
            };
        }

        /// <summary>
        /// Execute compaction on the GPU quad buffer.
        /// Copies each segment from source to dest offset using a GPU kernel.
        /// Segments are processed front-to-back so moves never overwrite unread data
        /// (dest is always <= source for every segment when sorted by source offset).
        /// </summary>
        /// <param name="accelerator">GPU accelerator.</param>
        /// <param name="quadBuffer">The shared PackedQuad buffer (long[] for packed quads).</param>
        /// <param name="plan">Compaction plan from Plan().</param>
        public static async Task ExecuteAsync(
            Accelerator accelerator,
            MemoryBuffer1D<long, Stride1D.Dense> quadBuffer,
            CompactionPlan plan)
        {
            if (!plan.ShouldCompact || plan.MovesRequired == 0)
                return;

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, int, int, int>(CompactCopyKernel);

            // Process segments front-to-back (safe: dest <= source when sorted)
            foreach (var seg in plan.Segments)
            {
                if (seg.DestOffset == seg.SourceOffset)
                    continue; // no move needed

                kernel((Index1D)seg.Count, quadBuffer.View, seg.SourceOffset, seg.DestOffset, seg.Count);
            }

            await accelerator.SynchronizeAsync();
        }

        /// <summary>
        /// GPU kernel: copy one quad from source to dest offset.
        /// Each thread copies one quad (one long value).
        /// </summary>
        static void CompactCopyKernel(Index1D index, ArrayView<long> buffer, int srcOffset, int dstOffset, int count)
        {
            if (index < count)
            {
                buffer[dstOffset + index] = buffer[srcOffset + index];
            }
        }

        /// <summary>
        /// Apply the compaction plan to section entries, updating their QuadOffset values.
        /// Call this AFTER ExecuteAsync completes.
        /// </summary>
        /// <param name="plan">The executed compaction plan.</param>
        /// <param name="updateOffset">Callback to update each section's QuadOffset.</param>
        public static void ApplyOffsets(CompactionPlan plan, Action<SectionCoord, int> updateOffset)
        {
            foreach (var seg in plan.Segments)
            {
                if (seg.DestOffset != seg.SourceOffset)
                {
                    updateOffset(seg.Owner, seg.DestOffset);
                }
            }
        }

        /// <summary>
        /// Calculate current fragmentation ratio.
        /// </summary>
        /// <param name="entries">All loaded section entries.</param>
        /// <param name="totalBufferUsed">Current high-water mark.</param>
        /// <returns>Fragmentation ratio (0 = no gaps, 1 = all gaps).</returns>
        public static float GetFragmentation(IEnumerable<SectionEntry> entries, int totalBufferUsed)
        {
            if (totalBufferUsed <= 0) return 0;
            int liveQuads = entries.Sum(e => e.QuadCount);
            return 1f - (float)liveQuads / totalBufferUsed;
        }
    }
}
