# SpawnDev.VoxelEngine Changelog

## 0.1.0-rc.2 (2026-04-28) - Fix unrestorable dep on rc.1

rc.1 was published to nuget.org with a hard dep on `SpawnDev.ILGPU 4.9.2-rc.7`, but rc.7 was a local-feed-only burner build that never made it to nuget.org. External consumers running `dotnet add package SpawnDev.VoxelEngine --version 0.1.0-rc.1` got `NU1102` and could not restore.

### Changes

- **Pin `SpawnDev.ILGPU` to `4.9.2-rc.24`** - the current latest published rc on nuget.org. The kernel + dispatch behaviour we depend on is unchanged from rc.7 (rc.7 to rc.24 is internal P2P / sweep-fix work).
- **Replace `SpawnDev.BlazorJS` `3.*` wildcard with a pinned `3.5.6`** - same class of bug; floating wildcards can pin to local-feed-only builds at package time. Pinning forces an explicit bump when the dep moves.

No source changes. Same bits, restorable deps. Thanks to Geordi for catching it via the `_audit-nugetorg-deps.ps1` audit script.

## 0.1.0-rc.1 (2026-04-25) - First Release Candidate

First public release candidate. Driven by Lost Spawns deployment - swap from `<ProjectReference>` to `<PackageReference>` so GitHub Actions / GitHub Pages can build the consumer.

### Shipped Subsystems

- **Meshing** - Binary greedy meshing kernels, face culling, occupancy masks (50-200 microseconds per chunk on WebGPU).
- **Culling** - Frustum culling, cylindrical fog culling, Sodium-style graph visibility. Hi-Z occlusion is reserved for v2 (graph + frustum handle 95%+ of voxel-world occlusion in practice).
- **Buffers** - Vertex pool with O(1) bucket alloc/free, single-buffer indirect draws (Chrome D3D12 300x validation optimization), priority eviction by distance and vertex cost.
- **LOD** - Distance-based super-block reduction via DelegateSpecialization kernels - one mesh kernel, multiple LOD behaviors, compile-time inlined.
- **Rendering** - Indirect draw pipeline, reversed-Z depth (zero Z-fighting at any distance with infinite far plane), 6 face-orientation groups per chunk for back-face skipping.
- **Adaptive** - Device limit queries at init, adaptive draw distance / vertex budget / LOD bias / streaming rate. Quest 3S gets appropriate budgets automatically.
- **Carving** - `TerrainCarveService` with sphere/box/cylinder add/remove modes via `ITerrainCarve`. NMS-style sphere terraform shipping in Lost Spawns.
- **Physics** - `VoxelRaycast.CastWorld` (DDA traversal returning `RaycastHit` with face + adjacent cell), `VoxelCollision` AABB-vs-world, `VoxelSphereQuery`, `StructuralIntegrity` for cave-in / floating-block detection.
- **Destruction** - `ExplosionKernels` for radial damage propagation on the GPU.
- **SDF** - `SdfMeshPipeline` with Dual Marching Cubes kernels and SDF noise generation for smooth-terrain styles alongside the cubic mesher.
- **Terrain** - `HeightmapTerrainProvider` and `BiomeAssigner` for procedural worlds.
- **VR / WebXR** - `StereoRenderer`, `FoveatedRendering`, `WebXRHelper` for Quest 3S and other WebXR runtimes. Includes 6 VR matrix helpers with 31 unit tests across all 6 backends (183/0 green).
- **Caching** - Chunk persistence layer for save/load round-trips.

### DI Registration

- `AddVoxelEngine(config => { ... })` - core registration with full configuration callback.
- `AddVoxelEngineAubsCraft()` - Minecraft-style blocky defaults (VoxelSize=1.0, SectionSize=16, DrawDistance=16, 4096 sections).
- `AddVoxelEngineLostSpawns()` - HD realistic defaults (VoxelSize=0.5, SectionSize=16, DrawDistance=24, 8192 sections).

### Cross-Platform

Same kernel code runs in browser (WebGPU/WebGL/Wasm) and desktop (CUDA, OpenCL, CPU) from one NuGet package via [SpawnDev.ILGPU](https://www.nuget.org/packages/SpawnDev.ILGPU).

### Dependencies

- SpawnDev.ILGPU 4.9.2-rc.7+
- SpawnDev.BlazorJS 3.*

### Known limitations

- API surface still RC-quality - expect iteration during the RC cycle.
- Hi-Z occlusion culling deferred to v2.
- Some XML-doc gaps on VR/WebXR enums (warnings only, no functional impact).
