using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    /// <summary>
    /// Base class for voxel engine tests. Backend-specific test classes inherit this
    /// and implement CreateAcceleratorAsync to select their backend.
    /// </summary>
    public abstract partial class VoxelEngineTestBase : IDisposable
    {
        public abstract string BackendName { get; }

        private Context? _cachedContext;
        private Accelerator? _cachedAccelerator;

        /// <summary>
        /// Creates the ILGPU context and accelerator for this backend.
        /// Override in backend-specific classes.
        /// Default implementation uses preferred accelerator.
        /// </summary>
        protected virtual async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var context = await Context.CreateAsync(builder => builder.AllAcceleratorsAsync());
            var accelerator = await SpawnDev.ILGPU.SpawnDevContextExtensions.CreatePreferredAcceleratorAsync(context);
            if (accelerator == null)
            {
                context.Dispose();
                throw new Exception($"No accelerator available for backend: {BackendName}");
            }
            return (context, accelerator);
        }

        /// <summary>
        /// Run a test with the cached accelerator. Creates it on first use.
        /// </summary>
        protected async Task RunTest(Func<Accelerator, Task> test)
        {
            if (_cachedAccelerator == null)
            {
                var (ctx, acc) = await CreateAcceleratorAsync();
                _cachedContext = ctx;
                _cachedAccelerator = acc;
            }
            await test(_cachedAccelerator);
        }

        public void Dispose()
        {
            _cachedAccelerator?.Dispose();
            _cachedContext?.Dispose();
            _cachedAccelerator = null;
            _cachedContext = null;
        }
    }

    // Face culling tests - verify GPU output matches CPU reference exactly
    public partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task FaceCull_SingleBlock_Exactly6Faces() => await RunTest(async accelerator =>
        {
            // A single block in empty space must produce exactly 6 visible faces
            int chunkXZ = 16;
            int height = 32; // keep small for fast tests
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.SingleBlock(chunkXZ, height);

            // CPU reference - naive count
            int naiveFaces = FaceCullCpuReference.CountVisibleFacesNaive(blocks, paddedXZ, height);
            if (naiveFaces != 6)
                throw new Exception($"Naive reference expected 6 faces for single block, got {naiveFaces}");

            // CPU reference - bitwise
            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            int bitwiseFaces = FaceCullCpuReference.CountVisibleFaces(cpuFaceMasks);
            if (bitwiseFaces != naiveFaces)
                throw new Exception($"Bitwise CPU reference got {bitwiseFaces} faces, naive got {naiveFaces}");

            // GPU kernel
            var gpuFaces = await RunFaceCullOnGpu(accelerator, blocks, paddedXZ, height);
            int gpuFaceCount = FaceCullCpuReference.CountVisibleFaces(gpuFaces);
            if (gpuFaceCount != naiveFaces)
                throw new Exception($"GPU got {gpuFaceCount} faces, CPU reference got {naiveFaces}");

            // Verify every mask matches exactly
            VerifyMasksMatch(cpuFaceMasks, gpuFaces, paddedXZ);
        });

        [TestMethod]
        public async Task FaceCull_FlatTerrain_MatchesCpuReference() => await RunTest(async accelerator =>
        {
            int chunkXZ = 16;
            int height = 32;
            int solidHeight = 4;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.FlatTerrain(chunkXZ, height, solidHeight);

            int naiveFaces = FaceCullCpuReference.CountVisibleFacesNaive(blocks, paddedXZ, height);
            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            int cpuFaces = FaceCullCpuReference.CountVisibleFaces(cpuFaceMasks);

            if (cpuFaces != naiveFaces)
                throw new Exception($"CPU bitwise ({cpuFaces}) != naive ({naiveFaces}) for flat terrain");

            var gpuFaces = await RunFaceCullOnGpu(accelerator, blocks, paddedXZ, height);
            int gpuFaceCount = FaceCullCpuReference.CountVisibleFaces(gpuFaces);

            if (gpuFaceCount != naiveFaces)
                throw new Exception($"GPU ({gpuFaceCount}) != CPU ({naiveFaces}) for flat terrain h={solidHeight}");

            VerifyMasksMatch(cpuFaceMasks, gpuFaces, paddedXZ);
        });

        [TestMethod]
        public async Task FaceCull_SolidCube_OnlyExteriorFaces() => await RunTest(async accelerator =>
        {
            int chunkXZ = 8;
            int height = 8;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.SolidCube(chunkXZ, height);

            // Solid 8x8x8 cube: exterior faces only
            // Top: 64, Bottom: 64, Front/Back/Left/Right: 4 sides x 8x8 = 256
            // Total: 64 + 64 + 256 = 384
            int naiveFaces = FaceCullCpuReference.CountVisibleFacesNaive(blocks, paddedXZ, height);

            var gpuFaces = await RunFaceCullOnGpu(accelerator, blocks, paddedXZ, height);
            int gpuFaceCount = FaceCullCpuReference.CountVisibleFaces(gpuFaces);

            if (gpuFaceCount != naiveFaces)
                throw new Exception($"Solid cube: GPU ({gpuFaceCount}) != CPU ({naiveFaces})");
        });

        [TestMethod]
        public async Task FaceCull_Checkerboard_WorstCase() => await RunTest(async accelerator =>
        {
            int chunkXZ = 8;
            int height = 8;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.Checkerboard(chunkXZ, height);

            int naiveFaces = FaceCullCpuReference.CountVisibleFacesNaive(blocks, paddedXZ, height);
            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            int cpuFaces = FaceCullCpuReference.CountVisibleFaces(cpuFaceMasks);

            if (cpuFaces != naiveFaces)
                throw new Exception($"Checkerboard: bitwise ({cpuFaces}) != naive ({naiveFaces})");

            var gpuFaces = await RunFaceCullOnGpu(accelerator, blocks, paddedXZ, height);
            int gpuFaceCount = FaceCullCpuReference.CountVisibleFaces(gpuFaces);

            if (gpuFaceCount != naiveFaces)
                throw new Exception($"Checkerboard: GPU ({gpuFaceCount}) != CPU ({naiveFaces})");

            VerifyMasksMatch(cpuFaceMasks, gpuFaces, paddedXZ);
        });

        [TestMethod]
        public async Task FaceCull_HollowBox_InteriorFacesHidden() => await RunTest(async accelerator =>
        {
            int chunkXZ = 8;
            int height = 8;
            int paddedXZ = chunkXZ + 2;
            var blocks = TestChunkGenerator.HollowBox(chunkXZ, height, 2);

            int naiveFaces = FaceCullCpuReference.CountVisibleFacesNaive(blocks, paddedXZ, height);

            var gpuFaces = await RunFaceCullOnGpu(accelerator, blocks, paddedXZ, height);
            int gpuFaceCount = FaceCullCpuReference.CountVisibleFaces(gpuFaces);

            if (gpuFaceCount != naiveFaces)
                throw new Exception($"Hollow box: GPU ({gpuFaceCount}) != CPU ({naiveFaces})");

            // Hollow box must have MORE faces than solid cube (interior surfaces)
            var solidBlocks = TestChunkGenerator.SolidCube(chunkXZ, height);
            int solidFaces = FaceCullCpuReference.CountVisibleFacesNaive(solidBlocks, paddedXZ, height);
            if (naiveFaces <= solidFaces)
                throw new Exception($"Hollow box ({naiveFaces}) should have more faces than solid ({solidFaces})");
        });

        [TestMethod]
        public async Task FaceCull_EmptyChunk_ZeroFaces() => await RunTest(async accelerator =>
        {
            int chunkXZ = 8;
            int height = 16;
            int paddedXZ = chunkXZ + 2;
            var blocks = new int[paddedXZ * paddedXZ * height]; // all air

            var gpuFaces = await RunFaceCullOnGpu(accelerator, blocks, paddedXZ, height);
            int gpuFaceCount = FaceCullCpuReference.CountVisibleFaces(gpuFaces);

            if (gpuFaceCount != 0)
                throw new Exception($"Empty chunk should have 0 faces, got {gpuFaceCount}");
        });

        /// <summary>
        /// Run the full face culling pipeline on GPU: occupancy -> face cull -> readback.
        /// </summary>
        private async Task<long[]> RunFaceCullOnGpu(Accelerator accelerator, int[] paddedBlocks, int paddedXZ, int height)
        {
            int innerXZ = paddedXZ - 2;
            int innerCount = innerXZ * innerXZ;

            // Allocate GPU buffers
            using var gpuBlocks = accelerator.Allocate1D(paddedBlocks);
            using var gpuOccupancy = accelerator.Allocate1D<long>(paddedXZ * paddedXZ);
            using var gpuFaceMasks = accelerator.Allocate1D<long>(innerCount * 6);

            // Load kernels
            var occupancyKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<int>, ArrayView<long>, int, int>(
                FaceCullKernels.BuildOccupancyKernel);

            var faceCullKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<long>, ArrayView<long>, int, int>(
                FaceCullKernels.FaceCullKernel);

            // Dispatch
            occupancyKernel(new Index2D(paddedXZ, paddedXZ),
                gpuBlocks.View, gpuOccupancy.View, paddedXZ, height);

            faceCullKernel(new Index2D(paddedXZ, paddedXZ),
                gpuOccupancy.View, gpuFaceMasks.View, paddedXZ, height);

            // Readback
            var result = await gpuFaceMasks.CopyToHostAsync();
            return result;
        }

        /// <summary>
        /// Verify every face mask matches between CPU and GPU output.
        /// </summary>
        private void VerifyMasksMatch(long[] cpuMasks, long[] gpuMasks, int paddedXZ)
        {
            if (cpuMasks.Length != gpuMasks.Length)
                throw new Exception($"Mask array length mismatch: CPU={cpuMasks.Length}, GPU={gpuMasks.Length}");

            int innerXZ = paddedXZ - 2;
            int innerCount = innerXZ * innerXZ;
            string[] faceNames = { "+X", "-X", "+Z", "-Z", "+Y", "-Y" };

            for (int i = 0; i < cpuMasks.Length; i++)
            {
                if (cpuMasks[i] != gpuMasks[i])
                {
                    int face = i / innerCount;
                    int innerIdx = i % innerCount;
                    int x = innerIdx % innerXZ;
                    int z = innerIdx / innerXZ;
                    throw new Exception(
                        $"Mask mismatch at face={faceNames[face]} x={x} z={z}: " +
                        $"CPU=0x{cpuMasks[i]:X16}, GPU=0x{gpuMasks[i]:X16}");
                }
            }
        }
    }
}
