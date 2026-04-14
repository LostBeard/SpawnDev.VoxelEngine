# SpawnDev.VoxelEngine

High-performance voxel engine library built on SpawnDev.ILGPU. Runs on WebGPU, WebGL, Wasm, CUDA, OpenCL, and CPU from a single codebase. Targets desktop, browser, and VR/AR (Quest 3S).

## Build Commands

```bash
dotnet build SpawnDev.VoxelEngine/SpawnDev.VoxelEngine.csproj   # Main library
dotnet build SpawnDev.VoxelEngine.slnx                           # Full solution
```

Target: **net10.0**

## Architecture

### Library Structure

| Directory | Purpose |
|-----------|---------|
| `Meshing/` | Binary greedy meshing kernels, face culling, occupancy masks |
| `Culling/` | Frustum, fog, Hi-Z occlusion, Sodium-style graph visibility |
| `Buffers/` | Vertex pool (bucket alloc), compaction, indirect draw buffer |
| `LOD/` | Super-block reduction, LOD selection, DelegateSpecialization kernels |
| `Rendering/` | Indirect draw pipeline, reversed-Z depth, draw command builder |
| `Adaptive/` | Device limit queries, quality controller, thermal response, budget |

### Design Principles

1. **WebGPU-first.** All compute kernels are ILGPU kernels targeting WebGPU compute shaders. Desktop backends (CUDA/OpenCL/CPU) work automatically via SpawnDev.ILGPU's cross-platform transpilation.

2. **GPU-resident data.** Voxel data enters the GPU and stays there through meshing, culling, and rendering. No CPU round-trips on the hot path. CopyFromJS for browser data ingestion.

3. **Parameterized, not hardcoded.** Chunk size, block format, vertex format, LOD levels, draw distance - all configurable. The library serves both blocky (AubsCraft) and HD realistic (Lost Spawns) voxel styles.

4. **Adaptive by default.** Query device limits at init, derive all budgets from actual hardware. Quest 3S gets smaller budgets automatically. No platform-specific code in consumers.

5. **ILGPU features everywhere.** DelegateSpecialization for LOD kernels. Lambda captures for per-frame scalars. RadixSort for draw ordering. Scan for compaction. Sub-word types for block storage. Shared memory for neighborhood loading.

### Consuming Projects

| Project | Style | Focus |
|---------|-------|-------|
| AubsCraft | Blocky (Minecraft) | Server-connected viewer, 2-worker architecture |
| Lost Spawns | HD Realistic | Standalone game, terrain gen, survival |

Both use project references during development. NuGet package when stable.

### Dependencies

- `SpawnDev.ILGPU` - GPU compute (all 6 backends)
- `SpawnDev.BlazorJS` - Browser interop, Web Workers

No other external dependencies.

## Key Patterns

### Kernel Conventions

- All kernels are static methods (or lambdas capturing scalars only)
- Use `ArrayView<T>` parameters for GPU buffers
- Stay within WebGPU's 10 storage buffer binding limit (pack structs, use scalars via lambda capture)
- Bounds-check ALL atomic counter writes: `if (offset + size > buffer.IntLength) { Atomic.Add(ref counter[0], -size); return; }`
- Zero counters before dispatch, guard every write path

### Buffer Management

- Vertex pool uses fixed-size buckets with FIFO free list (O(1) alloc/free)
- All indirect draw commands in ONE GPUBuffer (Chrome D3D12 300x validation win)
- Evict by priority score: `distSq * vertexCount` (far + heavy chunks first)
- Never silently discard - evict, compact, or re-queue

### Testing

- PlaywrightMultiTest runs desktop + browser tests from one `dotnet test`
- Test real production kernels with real data on all backends
- GPU-side verification via GpuTestVerify (no large CPU readbacks)
- No mock tests. No fake tests. Every test proves production code works.

## Research

Comprehensive research docs in `Research/`:
- `voxel-engine-optimization-masterplan.md` - Full technical reference for all optimization techniques
- `voxel-engines-comparison.md` - Survey of 30+ existing voxel engines with platform/feature comparison

## Global Rules

See `D:\users\tj\Projects\CLAUDE.md` for all global rules. Key ones for this project:
- Rule 1: No compromises. Every release is the final release.
- Rule 2: Fix libraries first. Bugs here are HIGHEST PRIORITY.
- Rule 4: Performance IS the mission. Squeeze every TFLOP.
- Rule 5: Test in unit tests, not demos.
