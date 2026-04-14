using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Adaptive
{
    /// <summary>
    /// Device tier classification based on queried hardware capabilities.
    /// Budgets and quality settings are derived from the tier, not hardcoded.
    /// </summary>
    public enum DeviceTier
    {
        /// <summary>Desktop GPU (CUDA, OpenCL, high-end WebGPU). Full quality.</summary>
        Desktop,

        /// <summary>High-end mobile / Quest 3S / M-series Mac. Reduced budgets.</summary>
        MobileHigh,

        /// <summary>Low-end mobile / integrated GPU. Minimum viable quality.</summary>
        MobileLow,
    }

    /// <summary>
    /// Queries device hardware limits at initialization and derives all engine budgets.
    /// No hardcoded limits - everything adapts to the actual GPU.
    ///
    /// Quest 3S (Adreno 740): 8GB shared, ~256MB buffer, 10 storage bindings.
    /// Desktop (RTX 4090): 24GB VRAM, 2GB+ buffer, 16+ storage bindings.
    ///
    /// Usage:
    ///   var caps = DeviceCapabilities.Query(accelerator, config);
    ///   int vertexBudget = caps.MaxVertexCount;
    ///   int sectionBudget = caps.MaxLoadedSections;
    /// </summary>
    public class DeviceCapabilities
    {
        /// <summary>Device tier classification.</summary>
        public DeviceTier Tier { get; init; }

        /// <summary>Maximum buffer size in bytes reported by the device.</summary>
        public long MaxBufferSize { get; init; }

        /// <summary>Maximum storage buffer bindings per shader stage.</summary>
        public int MaxStorageBufferBindings { get; init; }

        /// <summary>Maximum number of vertices the vertex pool can hold.</summary>
        public int MaxVertexCount { get; init; }

        /// <summary>Maximum number of sections that can be loaded.</summary>
        public int MaxLoadedSections { get; init; }

        /// <summary>Maximum number of quads across all sections.</summary>
        public int MaxTotalQuads { get; init; }

        /// <summary>Maximum number of indirect draw commands.</summary>
        public int MaxDrawCommands { get; init; }

        /// <summary>Number of vertex pool buckets (power-of-2 sizes).</summary>
        public int PoolBucketCount { get; init; }

        /// <summary>Whether ASTC texture compression is supported (hardware decompression).</summary>
        public bool SupportsASTC { get; init; }

        /// <summary>Whether shared memory is available (not available on WebGL).</summary>
        public bool SupportsSharedMemory { get; init; }

        /// <summary>Whether atomic operations are available.</summary>
        public bool SupportsAtomics { get; init; }

        /// <summary>Whether barriers are available (required for multi-pass algorithms).</summary>
        public bool SupportsBarriers { get; init; }

        /// <summary>Accelerator type for backend-specific decisions.</summary>
        public AcceleratorType AcceleratorType { get; init; }

        /// <summary>
        /// Query device capabilities and derive engine budgets.
        /// Call once at initialization with the active accelerator.
        /// </summary>
        public static DeviceCapabilities Query(Accelerator accelerator, VoxelEngineConfig config)
        {
            var type = accelerator.AcceleratorType;
            long maxBuffer = accelerator.MemorySize;
            int maxBindings = accelerator.MaxStorageBufferBindings;

            // Classify tier based on accelerator type and buffer size
            var tier = ClassifyTier(type, maxBuffer);

            // Derive budgets from tier and config
            int maxSections = tier switch
            {
                DeviceTier.Desktop => Math.Min(config.MaxLoadedSections, 8192),
                DeviceTier.MobileHigh => Math.Min(config.MaxLoadedSections, 2048),
                DeviceTier.MobileLow => Math.Min(config.MaxLoadedSections, 512),
                _ => 1024,
            };

            int maxQuads = tier switch
            {
                DeviceTier.Desktop => Math.Min(config.MaxTotalQuads, 4_000_000),
                DeviceTier.MobileHigh => Math.Min(config.MaxTotalQuads, 1_000_000),
                DeviceTier.MobileLow => Math.Min(config.MaxTotalQuads, 250_000),
                _ => 500_000,
            };

            // Vertex count = quads * 6 (2 triangles per quad)
            int maxVertices = maxQuads * 6;

            // Draw commands = one per section (or per face group with face masking)
            int maxDrawCommands = tier switch
            {
                DeviceTier.Desktop => maxSections * 6, // per-face group
                DeviceTier.MobileHigh => maxSections,   // per-section
                DeviceTier.MobileLow => maxSections,
                _ => maxSections,
            };

            // Pool buckets: power-of-2 from 64 to max section quad count
            int poolBuckets = tier switch
            {
                DeviceTier.Desktop => 12,     // 64 to 131072
                DeviceTier.MobileHigh => 10,  // 64 to 32768
                DeviceTier.MobileLow => 8,    // 64 to 8192
                _ => 10,
            };

            // Feature detection
            bool hasSharedMem = type != AcceleratorType.OpenCL || true; // WebGL handled by AcceleratorType
            bool hasAtomics = true; // All backends except WebGL
            bool hasBarriers = true;

            // WebGL lacks shared memory, atomics, and barriers
            if (IsWebGL(accelerator))
            {
                hasSharedMem = false;
                hasAtomics = false;
                hasBarriers = false;
            }

            return new DeviceCapabilities
            {
                Tier = tier,
                MaxBufferSize = maxBuffer,
                MaxStorageBufferBindings = maxBindings,
                MaxVertexCount = maxVertices,
                MaxLoadedSections = maxSections,
                MaxTotalQuads = maxQuads,
                MaxDrawCommands = maxDrawCommands,
                PoolBucketCount = poolBuckets,
                SupportsASTC = false, // Set by consumer after WebGPU adapter feature query
                SupportsSharedMemory = hasSharedMem,
                SupportsAtomics = hasAtomics,
                SupportsBarriers = hasBarriers,
                AcceleratorType = type,
            };
        }

        private static DeviceTier ClassifyTier(AcceleratorType type, long maxBuffer)
        {
            // Desktop backends
            if (type == AcceleratorType.Cuda || type == AcceleratorType.OpenCL)
                return DeviceTier.Desktop;

            if (type == AcceleratorType.CPU)
                return DeviceTier.Desktop;

            // Browser backends - classify by available memory
            // Quest 3S: maxBuffer ~256MB. Desktop Chrome: maxBuffer ~2GB+.
            if (maxBuffer >= 1_000_000_000L) // 1GB+
                return DeviceTier.Desktop;

            if (maxBuffer >= 128_000_000L) // 128MB+
                return DeviceTier.MobileHigh;

            return DeviceTier.MobileLow;
        }

        private static bool IsWebGL(Accelerator accelerator)
        {
            // WebGL accelerator type detection
            return accelerator.GetType().Name.Contains("WebGL", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Human-readable summary for debug overlay.</summary>
        public override string ToString()
        {
            return $"[{Tier}] Buffer={MaxBufferSize / 1_000_000}MB, Sections={MaxLoadedSections}, " +
                   $"Quads={MaxTotalQuads:N0}, Bindings={MaxStorageBufferBindings}, " +
                   $"SharedMem={SupportsSharedMemory}, Atomics={SupportsAtomics}";
        }
    }
}
