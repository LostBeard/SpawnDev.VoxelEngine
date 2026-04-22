using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.VoxelEngine.Demo.Shared.UnitTests;

namespace SpawnDev.VoxelEngine.DemoConsole
{
    public class CPUTests : VoxelEngineTestBase
    {
        public override string BackendName => "CPU";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var context = Context.Create(b => b.CPU().EnableAlgorithms());
            var accelerator = context.GetCPUDevice(0).CreateCPUAccelerator(context);
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }
    }
}
