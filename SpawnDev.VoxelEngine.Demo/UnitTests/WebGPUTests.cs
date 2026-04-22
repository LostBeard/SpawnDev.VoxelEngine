using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Demo.Shared.UnitTests;

namespace SpawnDev.VoxelEngine.Demo.UnitTests
{
    public class WebGPUTests : VoxelEngineTestBase
    {
        public override string BackendName => "WebGPU";

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create().EnableAlgorithms();
            builder.EnableWebGPUAlgorithms();
            await builder.WebGPU();
            var context = builder.ToContext();
            var devices = context.GetWebGPUDevices();
            if (devices.Count == 0)
            {
                context.Dispose();
                throw new UnsupportedTestException("No WebGPU devices found");
            }
            var accelerator = await devices[0].CreateAcceleratorAsync(context, null);
            return (context, accelerator);
        }
    }
}
