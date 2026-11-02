using ILGPU.Runtime;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>
        /// The GPU meshers refuse a device they cannot run on (MeshingSupport), instead of meshing garbage: on WebGL with
        /// ILGPU 5.4.14 every quad took slot 0 from the emulated Atomic.Add (the 2026-11-02 sweep: 28 WebGL failures of the
        /// old pipeline's tests). Runs on every lane: a device with atomics builds all three, one without gets
        /// NotSupportedException naming the mesher from each.
        /// </summary>
        [TestMethod]
        public async Task Meshers_RefuseADeviceWithoutAtomics_AndBuildOnOneWith() => await RunTest(async accelerator =>
        {
            await Task.CompletedTask;
            var builds = new (string Name, Func<IDisposable> Make)[]
            {
                (nameof(ColumnMesher), () => new ColumnMesher(accelerator, 2)),
                (nameof(HeightmapMesher), () => new HeightmapMesher(accelerator, 2)),
                (nameof(VoxelMeshPipeline), () => new VoxelMeshPipeline(accelerator)),
            };
            bool supported = MeshingSupport.IsSupported(accelerator);
            foreach (var (name, make) in builds)
            {
                if (supported)
                {
                    using var mesher = make();
                }
                else
                {
                    try
                    {
                        using var mesher = make();
                        throw new Exception($"{name} was built on {accelerator.AcceleratorType}, a device without atomics: it would mesh garbage");
                    }
                    catch (NotSupportedException ex) when (ex.Message.Contains(name)) { }
                }
            }
        });
    }
}
