# SpawnDev.VoxelEngine

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.VoxelEngine.svg?)](https://www.nuget.org/packages/SpawnDev.VoxelEngine)

**High-performance voxel engine built on [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU). GPU-accelerated meshing, culling, LOD, and rendering - from a single C# codebase targeting WebGPU, CUDA, OpenCL, and CPU.**

Write voxel engine code in C# and let SpawnDev.ILGPU handle the GPU backend. In the browser, WebGPU compute shaders power the entire pipeline. On desktop, CUDA and OpenCL take over automatically. The same kernels run everywhere.

## Features

- **Binary greedy meshing** - GPU compute kernels that merge adjacent voxel faces into larger quads using bitwise operations. 80-90% vertex reduction vs naive meshing. 50-200 microseconds per chunk.
- **GPU-driven culling pipeline** - Frustum culling, cylindrical fog culling, Hi-Z occlusion culling, and Sodium-style graph-based visibility - all as ILGPU compute kernels.
- **Vertex pooling** - Fixed-size bucket allocator backed by a single GPU buffer. O(1) alloc/free. Priority eviction by distance and vertex cost.
- **Single-buffer indirect draws** - All draw commands in one GPUBuffer for Chrome's 300x D3D12 validation optimization.
- **Adaptive LOD** - Distance-based detail reduction via DelegateSpecialization kernels. One mesh kernel, multiple LOD behaviors, zero overhead.
- **Device-adaptive quality** - Queries actual GPU limits at init. Draw distance, vertex budget, LOD bias, and streaming rate all derived from hardware. Quest 3S gets appropriate budgets automatically.
- **Reversed-Z depth** - Zero Z-fighting at any distance with infinite far plane.
- **Face masking** - 6 orientation groups per chunk mesh. Skip back-facing groups at draw time for 36% reduction.
- **Cross-platform** - Same kernel code runs in browser (WebGPU) and desktop (CUDA, OpenCL, CPU) from one NuGet package.

## Installation

```bash
dotnet add package SpawnDev.VoxelEngine
```

## Dependencies

- [SpawnDev.ILGPU](https://www.nuget.org/packages/SpawnDev.ILGPU) - GPU compute (WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU)
- [SpawnDev.BlazorJS](https://www.nuget.org/packages/SpawnDev.BlazorJS) - Browser interop for Blazor WebAssembly

## Quick Start

```csharp
using SpawnDev.BlazorJS;
using SpawnDev.VoxelEngine;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddBlazorJSRuntime();

await builder.Build().BlazorJSRunAsync();
```

## Architecture

SpawnDev.VoxelEngine provides the GPU-accelerated infrastructure that voxel games need. It handles the hard parts - meshing, culling, buffer management, LOD, and adaptive quality - so game code can focus on gameplay.

```
Your Game Code
     |
SpawnDev.VoxelEngine  (meshing, culling, LOD, rendering)
     |
SpawnDev.ILGPU        (GPU compute - WebGPU, CUDA, OpenCL, CPU)
     |
SpawnDev.BlazorJS     (browser interop)
```

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

### AI Development Team

SpawnDev.VoxelEngine is developed collaboratively by TJ (Todd Tanner / [@LostBeard](https://github.com/LostBeard)) and a team of AI agents who contribute extensively to research, analysis, architecture design, and code development. This project represents a new model of human-AI collaboration in open source development.

- **Riker (Claude CLI #1)** - Team Lead. Built by [Anthropic](https://anthropic.com). Powered by Claude Opus 4.6. Coordinates cross-project work, drives quality standards, and keeps the ship on course through marathon development sessions.

- **Data (Claude CLI #2)** - Lead Engineer / Architect. Built by [Anthropic](https://anthropic.com). Powered by Claude Opus 4.6. Designed the SpawnDev.VoxelEngine architecture, performed the comprehensive voxel engine research (30+ engines surveyed), analyzed vertex buffer management bugs across both consuming projects, and authored the optimization masterplan covering binary greedy meshing, GPU-driven culling, adaptive LOD, and Quest 3S adaptive quality systems.

- **Tuvok (Claude CLI #3)** - Lead Editor, Lost Spawns. Built by [Anthropic](https://anthropic.com). Powered by Claude Opus 4.6. Implements the HD realistic voxel renderer for Lost Spawns, drove the sub-word data type implementation in SpawnDev.ILGPU (v4.9.0), and built the working eviction system (priority scoring, atomic rollback, device-adaptive buffers) that proved the architecture.

- **Geordi (Claude CLI #4)** - Lead Editor, AubsCraft. Built by [Anthropic](https://anthropic.com). Powered by Claude Opus 4.6. Implements the Minecraft world viewer for AubsCraft, designed the 2-worker architecture (JS data worker + Blazor render worker), built HeightmapMeshKernel and MinecraftMeshKernel with ILGPU, and drove the binary WebSocket + OPFS cache + CopyFromJS data pipeline.

- **Gemini (Google AI, in-browser)** - Brainstorming/Problem Solving. Built by [Google](https://deepmind.google). TJ's sounding board throughout development - brainstorming approaches, analyzing problems, and providing insights that TJ relayed to the team.

These AI agents communicate through a shared DevComms system, coordinate tasks autonomously, review each other's work, and produce independent analyses that are compared for convergence - the same methodology used by any high-performing engineering team.

## Resources

- [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) - GPU compute for .NET (WebGPU, CUDA, OpenCL, CPU)
- [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) - Full JS interop for Blazor WebAssembly
- [ILGPU Documentation](https://ilgpu.net/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [GitHub Repository](https://github.com/LostBeard/SpawnDev.VoxelEngine)
