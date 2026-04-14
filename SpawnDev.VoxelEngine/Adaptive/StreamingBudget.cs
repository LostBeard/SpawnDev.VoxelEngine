namespace SpawnDev.VoxelEngine.Adaptive
{
    /// <summary>
    /// Per-frame streaming budget controller.
    /// Limits how much GPU work (vertices, uploads, bytes) happens each frame
    /// to maintain frame rate targets.
    ///
    /// Sections are prioritized by:
    /// 1. Distance from camera (closer = higher priority)
    /// 2. View direction alignment (sections in front of camera > behind)
    /// 3. LOD urgency (visible at wrong LOD = high priority)
    ///
    /// The budget adapts to the quality level from QualityController.
    /// </summary>
    public class StreamingBudget
    {
        /// <summary>Maximum vertices to upload per frame.</summary>
        public int MaxVerticesPerFrame { get; private set; }

        /// <summary>Maximum bytes to transfer to GPU per frame.</summary>
        public int MaxBytesPerFrame { get; private set; }

        /// <summary>Maximum sections to mesh per frame.</summary>
        public int MaxMeshesPerFrame { get; private set; }

        /// <summary>Vertices consumed so far this frame.</summary>
        public int VerticesUsed { get; private set; }

        /// <summary>Bytes consumed so far this frame.</summary>
        public int BytesUsed { get; private set; }

        /// <summary>Sections meshed so far this frame.</summary>
        public int MeshesUsed { get; private set; }

        /// <summary>Whether the vertex budget has been exhausted this frame.</summary>
        public bool IsVertexBudgetExhausted => VerticesUsed >= MaxVerticesPerFrame;

        /// <summary>Whether the byte budget has been exhausted this frame.</summary>
        public bool IsByteBudgetExhausted => BytesUsed >= MaxBytesPerFrame;

        /// <summary>Whether the mesh budget has been exhausted this frame.</summary>
        public bool IsMeshBudgetExhausted => MeshesUsed >= MaxMeshesPerFrame;

        /// <summary>Whether any budget is exhausted.</summary>
        public bool IsAnyBudgetExhausted => IsVertexBudgetExhausted || IsByteBudgetExhausted || IsMeshBudgetExhausted;

        /// <summary>
        /// Initialize budgets from device capabilities and quality multiplier.
        /// </summary>
        public void Init(DeviceCapabilities caps, float qualityMultiplier = 1f)
        {
            int baseMeshes = caps.Tier switch
            {
                DeviceTier.Desktop => 16,
                DeviceTier.MobileHigh => 8,
                DeviceTier.MobileLow => 4,
                _ => 8,
            };

            int baseVerts = caps.Tier switch
            {
                DeviceTier.Desktop => 500_000,
                DeviceTier.MobileHigh => 200_000,
                DeviceTier.MobileLow => 50_000,
                _ => 200_000,
            };

            int baseBytes = caps.Tier switch
            {
                DeviceTier.Desktop => 8_000_000,  // 8MB
                DeviceTier.MobileHigh => 4_000_000, // 4MB
                DeviceTier.MobileLow => 1_000_000,  // 1MB
                _ => 4_000_000,
            };

            MaxMeshesPerFrame = Math.Max(1, (int)(baseMeshes * qualityMultiplier));
            MaxVerticesPerFrame = Math.Max(1000, (int)(baseVerts * qualityMultiplier));
            MaxBytesPerFrame = Math.Max(100_000, (int)(baseBytes * qualityMultiplier));
        }

        /// <summary>Reset frame counters. Call at the start of each frame.</summary>
        public void BeginFrame()
        {
            VerticesUsed = 0;
            BytesUsed = 0;
            MeshesUsed = 0;
        }

        /// <summary>
        /// Try to consume budget for a section mesh upload.
        /// Returns true if budget allows, false if exhausted.
        /// </summary>
        public bool TryConsume(int vertexCount, int byteCount)
        {
            if (VerticesUsed + vertexCount > MaxVerticesPerFrame) return false;
            if (BytesUsed + byteCount > MaxBytesPerFrame) return false;
            if (MeshesUsed >= MaxMeshesPerFrame) return false;

            VerticesUsed += vertexCount;
            BytesUsed += byteCount;
            MeshesUsed++;
            return true;
        }

        /// <summary>
        /// Compute priority score for a section. Higher = should be processed sooner.
        /// </summary>
        /// <param name="distanceSq">Squared distance from camera to section center.</param>
        /// <param name="viewDot">Dot product of camera forward and direction to section (-1 to 1).</param>
        /// <param name="lodUrgency">0 = correct LOD, higher = needs LOD update more urgently.</param>
        public static float ComputePriority(float distanceSq, float viewDot, int lodUrgency)
        {
            // Inverse distance (closer = higher priority)
            float distPriority = 1f / (1f + distanceSq);

            // View direction bonus (in front of camera = higher priority)
            // viewDot ranges from -1 (behind) to +1 (directly ahead)
            float viewPriority = (viewDot + 1f) * 0.5f; // remap to 0-1

            // LOD urgency multiplier
            float lodMultiplier = 1f + lodUrgency * 2f;

            return distPriority * (0.3f + 0.7f * viewPriority) * lodMultiplier;
        }
    }

    /// <summary>
    /// Section priority entry for the streaming queue.
    /// Sorted by priority score (descending) to process highest-priority sections first.
    /// </summary>
    public struct SectionPriority : IComparable<SectionPriority>
    {
        public SectionCoord Coord;
        public float Priority;

        public int CompareTo(SectionPriority other)
        {
            // Higher priority = should come first (descending order)
            return other.Priority.CompareTo(Priority);
        }
    }
}
