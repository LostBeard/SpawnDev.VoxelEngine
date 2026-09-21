using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.VoxelEngine.Destruction
{
    /// <summary>
    /// GPU block-column carve/fill for the LostSpawns-style byte[] world.
    /// Uploads a column to PackedBlock ints, runs DestroyKernel or FillKernel,
    /// copies the dirty Y slab back to the byte[] cache. CPU DestroyInSphereBytes
    /// remains the test oracle.
    /// </summary>
    public sealed class BlockColumnCarveService : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, SphereOpParams> _destroyKernel;
        private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, SphereOpParams> _fillKernel;
        private MemoryBuffer1D<int, Stride1D.Dense>? _columnBuffer;
        private MemoryBuffer1D<int, Stride1D.Dense>? _counterBuffer;
        private int _columnCapacity;
        private int[]? _scratch;

        public BlockColumnCarveService(Accelerator accelerator)
        {
            _accelerator = accelerator;
            _destroyKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, SphereOpParams>(
                ExplosionKernels.DestroyKernel);
            _fillKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, SphereOpParams>(
                ExplosionKernels.FillKernel);
            _counterBuffer = accelerator.Allocate1D(new int[] { 0 });
        }

        /// <summary>
        /// Carve a sphere in one column. Center is column-local voxel space.
        /// Mutates <paramref name="blocks"/> for the dirty Y range. Returns destroyed count.
        /// </summary>
        public async Task<int> DestroySphereAsync(
            byte[] blocks, int sizeXZ, int sizeY,
            float centerX, float centerY, float centerZ, float radius)
        {
            if (radius <= 0) return 0;
            if (!TryBuildParams(sizeXZ, sizeY, centerX, centerY, centerZ, radius, packedBlock: 0, out var p, out int threads))
                return 0;

            EnsureColumn(blocks.Length);
            UploadColumn(blocks);
            _counterBuffer!.CopyFromCPU(new int[] { 0 });
            _destroyKernel((Index1D)threads, _columnBuffer!.View, _counterBuffer.View, p);
            await _accelerator.SynchronizeAsync();
            var count = await _counterBuffer.CopyToHostAsync();
            int destroyed = count[0];
            if (destroyed > 0)
                await DownloadSlabAsync(blocks, sizeXZ, p.MinY, p.MinY + p.RangeY - 1);
            return destroyed;
        }

        /// <summary>
        /// Fill air cells in a sphere. Returns cells written.
        /// </summary>
        public async Task<int> FillSphereAsync(
            byte[] blocks, int sizeXZ, int sizeY,
            float centerX, float centerY, float centerZ, float radius,
            int packedBlock)
        {
            if (radius <= 0 || (packedBlock & 0xFFF) == 0) return 0;
            if (!TryBuildParams(sizeXZ, sizeY, centerX, centerY, centerZ, radius, packedBlock, out var p, out int threads))
                return 0;

            EnsureColumn(blocks.Length);
            UploadColumn(blocks);
            _counterBuffer!.CopyFromCPU(new int[] { 0 });
            _fillKernel((Index1D)threads, _columnBuffer!.View, _counterBuffer.View, p);
            await _accelerator.SynchronizeAsync();
            var count = await _counterBuffer.CopyToHostAsync();
            int filled = count[0];
            if (filled > 0)
                await DownloadSlabAsync(blocks, sizeXZ, p.MinY, p.MinY + p.RangeY - 1);
            return filled;
        }

        private static bool TryBuildParams(
            int sizeXZ, int sizeY,
            float centerX, float centerY, float centerZ, float radius,
            int packedBlock,
            out SphereOpParams p, out int threads)
        {
            p = default;
            threads = 0;
            int minX = Math.Max(0, (int)MathF.Floor(centerX - radius));
            int minY = Math.Max(0, (int)MathF.Floor(centerY - radius));
            int minZ = Math.Max(0, (int)MathF.Floor(centerZ - radius));
            int maxX = Math.Min(sizeXZ - 1, (int)MathF.Floor(centerX + radius));
            int maxY = Math.Min(sizeY - 1, (int)MathF.Floor(centerY + radius));
            int maxZ = Math.Min(sizeXZ - 1, (int)MathF.Floor(centerZ + radius));
            int rangeX = maxX - minX + 1;
            int rangeY = maxY - minY + 1;
            int rangeZ = maxZ - minZ + 1;
            if (rangeX <= 0 || rangeY <= 0 || rangeZ <= 0) return false;
            p = new SphereOpParams
            {
                CenterX = centerX, CenterY = centerY, CenterZ = centerZ,
                RadiusSq = radius * radius,
                PackedBlock = packedBlock,
                SizeXZ = sizeXZ, SizeY = sizeY,
                MinX = minX, MinY = minY, MinZ = minZ,
                RangeX = rangeX, RangeY = rangeY, RangeZ = rangeZ,
            };
            threads = rangeX * rangeY * rangeZ;
            return true;
        }

        private void EnsureColumn(int length)
        {
            if (_columnBuffer != null && _columnCapacity >= length) return;
            _columnBuffer?.Dispose();
            _columnCapacity = length;
            _columnBuffer = _accelerator.Allocate1D<int>(length);
        }

        private void UploadColumn(byte[] blocks)
        {
            if (_scratch == null || _scratch.Length != blocks.Length)
                _scratch = new int[blocks.Length];
            for (int i = 0; i < blocks.Length; i++)
            {
                byte b = blocks[i];
                _scratch[i] = b != 0 ? PackedBlock.Pack(b) : 0;
            }
            _columnBuffer!.View.SubView(0, blocks.Length).CopyFromCPU(_scratch);
        }

        private async Task DownloadSlabAsync(byte[] blocks, int sizeXZ, int minY, int maxY)
        {
            int xz = sizeXZ * sizeXZ;
            int start = minY * xz;
            int len = (maxY - minY + 1) * xz;
            // Full-column readback then apply slab - SubView.CopyFrom between GPU buffers
            // is unreliable on WebGL; dig columns are 65KB so this is acceptable for carve.
            var host = await _columnBuffer!.CopyToHostAsync();
            for (int i = 0; i < len; i++)
                blocks[start + i] = (byte)(host[start + i] & 0xFFF);
        }

        public void Dispose()
        {
            _columnBuffer?.Dispose();
            _counterBuffer?.Dispose();
        }
    }
}
