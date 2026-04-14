using Microsoft.Extensions.DependencyInjection;
using SpawnDev.VoxelEngine.Adaptive;
using SpawnDev.VoxelEngine.Buffers;
using SpawnDev.VoxelEngine.Culling;
using SpawnDev.VoxelEngine.LOD;

namespace SpawnDev.VoxelEngine
{
    /// <summary>
    /// DI service registration for SpawnDev.VoxelEngine.
    ///
    /// Usage in Program.cs:
    ///   builder.Services.AddVoxelEngine(config => {
    ///       config.VoxelSize = 0.5f;  // Lost Spawns
    ///       config.DrawDistance = 16;
    ///   });
    ///
    /// All engine services are registered as singletons.
    /// The accelerator must be provided separately (it's created by ILGPU initialization).
    /// </summary>
    public static class VoxelEngineServiceExtensions
    {
        /// <summary>
        /// Register all VoxelEngine services with the DI container.
        /// </summary>
        /// <param name="services">The service collection.</param>
        /// <param name="configure">Optional configuration callback.</param>
        public static IServiceCollection AddVoxelEngine(
            this IServiceCollection services,
            Action<VoxelEngineConfig>? configure = null)
        {
            var config = new VoxelEngineConfig();
            configure?.Invoke(config);

            // Core configuration
            services.AddSingleton(config);

            // Block registry
            services.AddSingleton<BlockRegistry>();

            // Chunk management
            services.AddSingleton(sp => new ChunkManager(sp.GetRequiredService<VoxelEngineConfig>()));

            // Adaptive systems
            services.AddSingleton<QualityController>();
            services.AddSingleton<StreamingBudget>();
            services.AddSingleton<ThermalManager>();

            // LOD
            services.AddSingleton<LODSelector>();

            // Culling
            services.AddSingleton(sp => new CullingPipeline(sp.GetRequiredService<VoxelEngineConfig>()));

            // Buffers (indirect draw buffer capacity set after DeviceCapabilities query)
            services.AddSingleton(sp =>
            {
                var cfg = sp.GetRequiredService<VoxelEngineConfig>();
                return new IndirectDrawBuffer(cfg.MaxLoadedSections * 6); // per-face-group capacity
            });

            return services;
        }

        /// <summary>
        /// Register VoxelEngine with AubsCraft defaults.
        /// </summary>
        public static IServiceCollection AddVoxelEngineAubsCraft(this IServiceCollection services)
        {
            return services.AddVoxelEngine(config =>
            {
                config.VoxelSize = 1.0f;
                config.SectionSize = 16;
                config.DrawDistance = 16;
                config.MaxLoadedSections = 4096;
                config.MaxTotalQuads = 1_000_000;
                config.BaseY = -64f;
                config.SectionsPerColumn = 24;
            });
        }

        /// <summary>
        /// Register VoxelEngine with Lost Spawns defaults.
        /// </summary>
        public static IServiceCollection AddVoxelEngineLostSpawns(this IServiceCollection services)
        {
            return services.AddVoxelEngine(config =>
            {
                config.VoxelSize = 0.5f;
                config.SectionSize = 16;
                config.DrawDistance = 24;
                config.MaxLoadedSections = 8192;
                config.MaxTotalQuads = 2_000_000;
                config.BaseY = 0f;
                config.SectionsPerColumn = 32; // 256m / 0.5m / 16 = 32 sections
            });
        }
    }
}
