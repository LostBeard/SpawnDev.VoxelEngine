using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Demo.Shared.UnitTests;

namespace SpawnDev.VoxelEngine.DemoConsole
{
    public class CudaTests : VoxelEngineTestBase
    {
        public override string BackendName => "CUDA";
        protected override string ExpectedAcceleratorTypeName => "Cuda";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var context = Context.Create(b => b.AllAccelerators().EnableAlgorithms());
            var devices = context.GetCudaDevices();
            if (devices.Count == 0)
            {
                context.Dispose();
                throw new UnsupportedTestException("No CUDA devices found");
            }
            var accelerator = devices[0].CreateAccelerator(context);
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }
    }
}
