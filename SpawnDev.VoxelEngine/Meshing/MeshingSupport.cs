using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// What the GPU meshers (<see cref="ColumnMesher"/>, <see cref="HeightmapMesher"/>, <see cref="VoxelMeshPipeline"/>) and the
    /// GPU frustum cull (<see cref="Culling.FrustumCullKernels"/>) need from a device: atomics (every output quad or chunk takes
    /// its slot from <c>Atomic.Add</c>'s return value, and face masks are cleared with 64-bit <c>Atomic.And</c>), stores to
    /// computed addresses, and 64-bit integers.
    ///
    /// WebGL2 has none of the atomics: ILGPU's WebGL emulation returns 0 from <c>Atomic.Add</c>, so every quad landed in slot 0
    /// (ILGPU 5.4.14; 5.4.15 refuses the kernel instead). The meshers check at construction and throw
    /// <see cref="NotSupportedException"/>, so a WebGL device fails loudly instead of meshing garbage. Use WebGPU (or a desktop
    /// backend). An atomic-free path (flags, a global scan, a scatter) is possible on WebGL and is not written yet
    /// (TJ 2026-11-02: later).
    /// </summary>
    public static class MeshingSupport
    {
        /// <summary>The device features the GPU meshers need.</summary>
        public static AcceleratorRequirements Requirements => new()
        {
            RequiresAtomics = true,
            RequiresScatterStores = true,
            RequiresInt64 = true,
        };

        /// <summary>True when <paramref name="accelerator"/> can run the GPU meshers.</summary>
        public static bool IsSupported(Accelerator accelerator) => accelerator.Device.Satisfies(Requirements);

        /// <summary>Throws <see cref="NotSupportedException"/> when <paramref name="accelerator"/> cannot run <paramref name="what"/>.</summary>
        public static void Require(Accelerator accelerator, string what)
        {
            if (!IsSupported(accelerator))
                throw new NotSupportedException($"{what} needs atomics, scatter stores and 64-bit integers; {accelerator.GetType().Name} ({accelerator.AcceleratorType}) has not. Use WebGPU or a desktop backend.");
        }
    }
}
