using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Demo.Shared.UnitTests;

namespace SpawnDev.VoxelEngine.Demo.UnitTests
{
    public class WebGLTests : VoxelEngineTestBase
    {
        public override string BackendName => "WebGL";

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create().EnableAlgorithms();
            await builder.WebGL();
            var context = builder.ToContext();
            var devices = context.GetWebGLDevices();
            if (devices.Count == 0)
            {
                context.Dispose();
                throw new UnsupportedTestException("No WebGL2 devices found");
            }
            var accelerator = devices[0].CreateAccelerator(context);
            return (context, accelerator);
        }
    }
}
