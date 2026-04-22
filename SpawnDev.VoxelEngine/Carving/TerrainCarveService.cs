using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.VoxelEngine.SDF;

namespace SpawnDev.VoxelEngine.Carving
{
    /// <summary>
    /// Default <see cref="ITerrainCarve"/> implementation. One instance per
    /// accelerator; compiles the CSG kernel once and reuses it for every call.
    ///
    /// Mode -&gt; blend radius mapping:
    /// - Dig:     k = 1.0  (crisp edge, small visible rounding)
    /// - Fill:    k = 1.0  (same, mirrored for additive)
    /// - Explode: k = 3.0  (wide smoothing, blast-like rim)
    /// </summary>
    public sealed class TerrainCarveService : ITerrainCarve
    {
        private readonly Accelerator _accelerator;
        private readonly Action<
            Index3D, ArrayView<short>,
            float, float, float,
            float,
            int,
            float,
            float, float, float,
            float,
            int> _modifyKernel;

        public TerrainCarveService(Accelerator accelerator)
        {
            _accelerator = accelerator;
            _modifyKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<short>,
                float, float, float,
                float,
                int,
                float,
                float, float, float,
                float,
                int>(SdfNoiseKernels.ModifySdfSphereKernel);
        }

        public (int kernelMode, float blendRadius) ResolveKernelParams(CarveMode mode)
        {
            return mode switch
            {
                CarveMode.Dig => (0, 1.0f),
                CarveMode.Fill => (1, 1.0f),
                CarveMode.Explode => (0, 3.0f),
                _ => (0, 1.0f),
            };
        }

        public bool SphereIntersectsChunk(
            Vector3 chunkWorldMin,
            float voxelSize,
            int chunkSize,
            Vector3 worldCenter,
            float radius,
            CarveMode mode)
        {
            if (radius <= 0f) return false;

            var (_, blend) = ResolveKernelParams(mode);

            // Expand the chunk AABB by (radius + blend + 1) on all sides. The kernel's
            // early-reject uses the same margin, so this matches the "would dispatch
            // modify any voxels" test.
            float expand = radius + blend + 1f;
            float extent = chunkSize * voxelSize;
            Vector3 chunkMax = chunkWorldMin + new Vector3(extent);

            if (worldCenter.X < chunkWorldMin.X - expand) return false;
            if (worldCenter.Y < chunkWorldMin.Y - expand) return false;
            if (worldCenter.Z < chunkWorldMin.Z - expand) return false;
            if (worldCenter.X > chunkMax.X + expand) return false;
            if (worldCenter.Y > chunkMax.Y + expand) return false;
            if (worldCenter.Z > chunkMax.Z + expand) return false;
            return true;
        }

        public bool ApplySphereToChunk(
            MemoryBuffer1D<short, Stride1D.Dense> sdfValues,
            Vector3 chunkWorldMin,
            float voxelSize,
            int chunkSize,
            Vector3 worldCenter,
            float radius,
            CarveMode mode)
        {
            if (sdfValues is null) throw new ArgumentNullException(nameof(sdfValues));
            if (voxelSize <= 0f) throw new ArgumentOutOfRangeException(nameof(voxelSize));
            if (chunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSize));

            long expectedLen = (long)chunkSize * chunkSize * chunkSize;
            if (sdfValues.Length < expectedLen)
            {
                throw new ArgumentException(
                    $"SDF buffer has {sdfValues.Length} elements but chunkSize {chunkSize} needs {expectedLen}.",
                    nameof(sdfValues));
            }

            if (!SphereIntersectsChunk(chunkWorldMin, voxelSize, chunkSize, worldCenter, radius, mode))
                return false;

            var (kernelMode, blend) = ResolveKernelParams(mode);

            _modifyKernel(
                new Index3D(chunkSize, chunkSize, chunkSize),
                sdfValues.View,
                worldCenter.X, worldCenter.Y, worldCenter.Z,
                radius,
                kernelMode,
                blend,
                chunkWorldMin.X, chunkWorldMin.Y, chunkWorldMin.Z,
                voxelSize,
                chunkSize);
            return true;
        }

        public async Task<bool> ApplySphereToChunkAsync(
            MemoryBuffer1D<short, Stride1D.Dense> sdfValues,
            Vector3 chunkWorldMin,
            float voxelSize,
            int chunkSize,
            Vector3 worldCenter,
            float radius,
            CarveMode mode)
        {
            var dispatched = ApplySphereToChunk(
                sdfValues, chunkWorldMin, voxelSize, chunkSize,
                worldCenter, radius, mode);
            if (dispatched)
                await _accelerator.SynchronizeAsync();
            return dispatched;
        }
    }
}
