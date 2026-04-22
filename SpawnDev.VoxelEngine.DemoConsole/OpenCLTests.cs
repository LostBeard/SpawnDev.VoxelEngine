using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Demo.Shared.UnitTests;

namespace SpawnDev.VoxelEngine.DemoConsole
{
    public class OpenCLTests : VoxelEngineTestBase
    {
        public override string BackendName => "OpenCL";
        // CLAccelerator is the ILGPU type name (note: CL, not OpenCL).
        protected override string ExpectedAcceleratorTypeName => "CL";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var context = Context.Create(b => b.AllAccelerators().EnableAlgorithms());
            var devices = context.GetCLDevices();
            if (devices.Count == 0)
            {
                context.Dispose();
                throw new UnsupportedTestException("No OpenCL devices found");
            }
            var accelerator = devices[0].CreateAccelerator(context);
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }
    }
}
