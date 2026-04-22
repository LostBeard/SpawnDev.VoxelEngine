using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.Wasm;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Demo.Shared.UnitTests;

namespace SpawnDev.VoxelEngine.Demo.UnitTests
{
    public class WasmTests : VoxelEngineTestBase
    {
        public override string BackendName => "Wasm";

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create()
                .EnableAlgorithms()
                .EnableWasmAlgorithms()
                .Wasm();
            var context = builder.ToContext();
            var accelerator = await context.CreateWasmAcceleratorAsync();
            return (context, accelerator);
        }
    }
}
