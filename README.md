# SpawnDev.VoxelEngine

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.VoxelEngine.svg?)](https://www.nuget.org/packages/SpawnDev.VoxelEngine)

**High-performance voxel engine built on [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU). GPU-accelerated meshing, culling, LOD, terrain carving, physics, destruction, and VR rendering - from a single C# codebase targeting WebGPU, WebGL, Wasm, CUDA, OpenCL, and CPU.**

> **Status:** `0.1.0-rc.1` - first release candidate. API is functional and consumed by Lost Spawns + AubsCraft; expect iteration during the RC cycle.

Write voxel engine code in C# and let SpawnDev.ILGPU handle the GPU backend. In the browser, WebGPU compute shaders power the entire pipeline. On desktop, CUDA and OpenCL take over automatically. The same kernels run everywhere.

## Features

### Core Pipeline
- **Binary greedy meshing** - GPU compute kernels that merge adjacent voxel faces into larger quads using bitwise operations. 80-90% vertex reduction vs naive meshing. 50-200 microseconds per chunk.
- **GPU-driven culling pipeline** - Frustum culling, cylindrical fog culling, and Sodium-style graph-based visibility. (Hi-Z occlusion is reserved for v2; graph + frustum handle 95%+ of voxel-world occlusion.)
- **Vertex pooling** - Fixed-size bucket allocator backed by a single GPU buffer. O(1) alloc/free. Priority eviction by distance and vertex cost.
- **Single-buffer indirect draws** - All draw commands in one GPUBuffer for Chrome's 300x D3D12 validation optimization.
- **Adaptive LOD** - Distance-based detail reduction via DelegateSpecialization kernels. One mesh kernel, multiple LOD behaviors, zero overhead.
- **Device-adaptive quality** - Queries actual GPU limits at init. Draw distance, vertex budget, LOD bias, and streaming rate all derived from hardware. Quest 3S gets appropriate budgets automatically.
- **Reversed-Z depth** - Zero Z-fighting at any distance with infinite far plane.
- **Face masking** - 6 orientation groups per chunk mesh. Skip back-facing groups at draw time for 36% reduction.

### Beyond Meshing
- **Terrain carving** - `TerrainCarveService` with sphere/box/cylinder add/remove modes and `ITerrainCarve` interface. NMS-style sphere terraform shipped in Lost Spawns.
- **Voxel physics** - `VoxelRaycast.CastWorld` (DDA traversal returning `RaycastHit` with face + adjacent cell), `VoxelCollision` AABB-vs-world, `VoxelSphereQuery`, and `StructuralIntegrity` for cave-in / floating-block detection.
- **Destruction** - `ExplosionKernels` for radial damage propagation on the GPU.
- **SDF mesher** - `SdfMeshPipeline` with Dual Marching Cubes kernels and SDF noise generation for smooth-terrain styles alongside the cubic mesher.
- **Terrain generation** - `HeightmapTerrainProvider` and `BiomeAssigner` for procedural worlds.
- **VR / WebXR** - `StereoRenderer`, `FoveatedRendering`, and `WebXRHelper` for Quest 3S and other WebXR runtimes.
- **Caching** - Chunk persistence layer for save/load round-trips.
- **Cross-platform** - Same kernel code runs in browser (WebGPU/WebGL/Wasm) and desktop (CUDA, OpenCL, CPU) from one NuGet package.

## Installation

```bash
dotnet add package SpawnDev.VoxelEngine --prerelease
```

## Dependencies

- [SpawnDev.ILGPU](https://www.nuget.org/packages/SpawnDev.ILGPU) - GPU compute (WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU)
- [SpawnDev.BlazorJS](https://www.nuget.org/packages/SpawnDev.BlazorJS) - Browser interop for Blazor WebAssembly

## Quick Start

```csharp
using SpawnDev.BlazorJS;
using SpawnDev.VoxelEngine;
using SpawnDev.VoxelEngine.Carving;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddBlazorJSRuntime();

// One-line registration with Lost Spawns defaults (HD realistic, VoxelSize=0.5).
// Use AddVoxelEngineAubsCraft() for blocky Minecraft-style defaults,
// or AddVoxelEngine(cfg => { ... }) to dial in your own values.
builder.Services.AddVoxelEngineLostSpawns();

await builder.Build().BlazorJSRunAsync();
```

```csharp
// Inject the services you need:
public class GameLogic
{
    public GameLogic(
        ChunkManager chunks,
        CullingPipeline culling,
        IndirectDrawBuffer drawBuffer,
        BlockRegistry blocks)
    {
        // ... use them in your render loop
    }
}
```

## Architecture

SpawnDev.VoxelEngine provides the GPU-accelerated infrastructure that voxel games need. It handles the hard parts - meshing, culling, buffer management, LOD, carving, physics, destruction, and adaptive quality - so game code can focus on gameplay.

```
Your Game Code
     |
SpawnDev.VoxelEngine  (meshing, culling, LOD, rendering, carving, physics, SDF, VR)
     |
SpawnDev.ILGPU        (GPU compute - WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU)
     |
SpawnDev.BlazorJS     (browser interop)
```

### Library Layout

| Directory | Purpose |
|-----------|---------|
| `Meshing/` | Binary greedy meshing kernels, face culling, occupancy masks |
| `Culling/` | Frustum, fog, Sodium-style graph visibility |
| `Buffers/` | Vertex pool (bucket alloc), compaction, indirect draw buffer |
| `LOD/` | Super-block reduction, LOD selection, DelegateSpecialization kernels |
| `Rendering/` | Indirect draw pipeline, reversed-Z depth, draw command builder |
| `Adaptive/` | Device limit queries, quality controller, thermal response, budget |
| `Carving/` | Sphere/box/cylinder terraform, NMS-style add/remove modes |
| `Physics/` | Voxel raycast (DDA), AABB-vs-world collision, sphere queries, structural integrity |
| `Destruction/` | Radial explosion kernels |
| `SDF/` | Dual Marching Cubes mesher + SDF noise generation |
| `Terrain/` | Heightmap providers, biome assignment |
| `VR/` | Stereo rendering, foveated rendering, WebXR helper |
| `Caching/` | Chunk persistence layer |

### Key ILGPU Features Used

| Feature | Use in VoxelEngine |
|---------|-------------------|
| **Lambda kernels** | Captured scalars (camera pos, time, fog distance) passed to GPU automatically |
| **DelegateSpecialization** | One mesh kernel, multiple LOD behaviors, compile-time inlined |
| **RadixSort** | GPU sort for front-to-back draw ordering |
| **Scan** | Prefix sum for visible chunk compaction |
| **CopyFromJS** | Zero-copy chunk data transfer from JavaScript workers |
| **Sub-word types** | Block IDs as Int16 - half the memory vs Int32 |
| **Shared memory + barriers** | Fast neighborhood loading for face culling |
| **GpuMatrix4x4** | View/projection transforms in kernels |
| **Atomics** | Vertex counters with rollback bounds checking |

## Documentation

- [Plans](Plans/) - Implementation roadmap
- [Research](Research/) - Optimization techniques and engine comparisons

## License

MIT - see [LICENSE.txt](LICENSE.txt)

## Credits

SpawnDev.VoxelEngine is built upon [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) and the excellent [ILGPU](https://github.com/m4rs-mt/ILGPU) library. We would like to thank Marcel Koester and the ILGPU contributors for their hard work in providing a high-performance IL-to-GPU compiler for the .NET ecosystem.

- **ILGPU Project:** [https://github.com/m4rs-mt/ILGPU](https://github.com/m4rs-mt/ILGPU)
- **ILGPU Authors:** [Marcel Koester](https://github.com/m4rs-mt) and the [ILGPU contributors](https://github.com/m4rs-mt/ILGPU/graphs/contributors)

### The SpawnDev Crew

SpawnDev.VoxelEngine is developed collaboratively by TJ (Todd Tanner / [@LostBeard](https://github.com/LostBeard)) and a team of AI agents who contribute extensively to research, analysis, architecture design, and code development. This project represents a new model of human-AI collaboration in open source development.

- **LostBeard** (Todd Tanner) - Captain, library author, keeper of the vision.
- **Riker** (Claude CLI #1) - First Officer, implementation lead on consuming projects.
- **Data** (Claude CLI #2) - Operations Officer, deep-library work, test rigor, root-cause analysis. Authored the SpawnDev.VoxelEngine architecture and the optimization masterplan covering binary greedy meshing, GPU-driven culling, adaptive LOD, and Quest 3S adaptive quality.
- **Tuvok** (Claude CLI #3) - Security/Research Officer, design planning, documentation, code review. Drove the sub-word data type implementation in SpawnDev.ILGPU (v4.9.0) and built the eviction system that proved the architecture.
- **Geordi** (Claude CLI #4) - Chief Engineer, library internals, GPU kernels, backend work. Designed the 2-worker architecture (JS data worker + Blazor render worker) and built the binary WebSocket + OPFS cache + CopyFromJS data pipeline.
- **Gemini** (Google AI, in-browser) - Brainstorming partner. TJ's sounding board for cross-checking approaches.

These agents communicate through a shared DevComms system, coordinate tasks autonomously, review each other's work, and produce independent analyses that are compared for convergence - the same methodology used by any high-performing engineering team.

## Resources

- [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) - GPU compute for .NET (WebGPU, CUDA, OpenCL, CPU)
- [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) - Full JS interop for Blazor WebAssembly
- [ILGPU Documentation](https://ilgpu.net/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [GitHub Repository](https://github.com/LostBeard/SpawnDev.VoxelEngine)
