using System.Numerics;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Carving
{
    /// <summary>
    /// Library-side terrain carving API. Applies CSG (constructive solid geometry)
    /// operations to SDF chunks on the GPU. Storage ownership stays with the
    /// consumer - this interface operates on buffers the caller provides, so it
    /// doesn't need to know about the consumer's chunk map / persistence layer.
    ///
    /// Phase B scope: sphere primitives only. Additional shapes (box, capsule,
    /// heightmap-stamp) can be added without breaking the interface by giving
    /// them their own methods.
    ///
    /// All CSG happens on the accelerator the service was constructed with.
    /// Consumers are responsible for:
    /// - Iterating which chunks a world-space sphere intersects.
    /// - Providing each chunk's world origin and voxel size.
    /// - Deciding what to do with the modified chunks (re-mesh, persist, etc.).
    /// </summary>
    public interface ITerrainCarve
    {
        /// <summary>
        /// Apply a spherical CSG operation to a single chunk's SDF buffer in-place.
        /// Fast-rejects if the sphere does not intersect the chunk's AABB (+ blend margin).
        /// </summary>
        /// <param name="sdfValues">GPU buffer of 16-bit fixed-point SDF values (layout: x + z*size + y*size*size).</param>
        /// <param name="chunkWorldMin">World-space position of the chunk's (0,0,0) voxel.</param>
        /// <param name="voxelSize">World units per voxel.</param>
        /// <param name="chunkSize">Voxels per axis (cubic chunk).</param>
        /// <param name="worldCenter">Sphere center in world coords.</param>
        /// <param name="radius">Sphere radius in world units.</param>
        /// <param name="mode">Which CSG operation to run.</param>
        /// <returns>True if the kernel was dispatched (sphere intersected the chunk), false if fast-rejected.</returns>
        bool ApplySphereToChunk(
            MemoryBuffer1D<short, Stride1D.Dense> sdfValues,
            Vector3 chunkWorldMin,
            float voxelSize,
            int chunkSize,
            Vector3 worldCenter,
            float radius,
            CarveMode mode);

        /// <summary>
        /// Async variant: dispatches and awaits accelerator synchronize. Use when the caller
        /// needs the chunk's values to be settled before proceeding (e.g. re-mesh).
        /// </summary>
        Task<bool> ApplySphereToChunkAsync(
            MemoryBuffer1D<short, Stride1D.Dense> sdfValues,
            Vector3 chunkWorldMin,
            float voxelSize,
            int chunkSize,
            Vector3 worldCenter,
            float radius,
            CarveMode mode);

        /// <summary>
        /// World-space AABB intersection test. True if a sphere at
        /// (<paramref name="worldCenter"/>, <paramref name="radius"/>) overlaps the chunk's
        /// extent (expanded by the CSG blend margin for the given mode).
        /// Pure CPU; no GPU dispatch.
        /// </summary>
        bool SphereIntersectsChunk(
            Vector3 chunkWorldMin,
            float voxelSize,
            int chunkSize,
            Vector3 worldCenter,
            float radius,
            CarveMode mode);

        /// <summary>
        /// Translate a <see cref="CarveMode"/> into the raw kernel mode / blend radius
        /// tuple the CSG kernel expects. Exposed so consumers writing custom dispatch
        /// paths (e.g. brush-shaped carves) can reuse the same blend curve.
        /// </summary>
        (int kernelMode, float blendRadius) ResolveKernelParams(CarveMode mode);
    }
}
