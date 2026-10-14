# SpawnDev.VoxelEngine Changelog

## 0.2.0-local.1 (2026-10-14, local feed only, not released) - Textured block rendering

AubsCraft's 3D viewer is moving onto this library (its own mesh kernels read every mesh back to .NET and draw one flat colour
per block). `VertexPullPipeline` draws flat colours only, so the renderer a Minecraft-style game needs lives here now:

- **`BlockRenderPipeline`**: textured, tinted, lit, fogged blocks from `PackedQuad` buffers, in three passes: `Opaque` (cull back
  faces), `Cross` (plants: two crossed planes per block, double-sided) and `Water` (alpha blended, depth test, NO depth write).
  Alpha cutout (leaf gaps, plant outlines); per-face brightness steps; sun + ambient + a fixed cool fill light; start/end fog
  with a quadratic blend. The caller owns the lighting model (`BlockFrameUniforms`: camera, sun, ambient, fog) - a day/night
  cycle or a dungeon is the caller's choice. Reversed-Z depth.
- **Texture ARRAY with a repeating sampler, not an atlas**: a greedy-merged w x h quad tiles its texture once per block, which an
  atlas cannot do. `BlockTextureArray.FromAtlas` slices an RGBA atlas into layers ON THE GPU (one upload, then
  copyTextureToTexture per tile; layer = row * tilesX + column); the pixels stay in the JS typed array (a `byte[]` overload exists
  for tests and tools).
- **`BlockMaterial`** (32 bytes per block type, a storage buffer): a texture layer for each of the six faces, a tint, an alpha for
  the water pass, flags (`FlagTinted`, `FlagLowerTop` = a water surface 0.1 below the block top, `FlagFlatShading`). A face with
  `NoTexture` is drawn in the tint colour; a block type nobody registered is untextured magenta.
- **Section placement through the instance index**: `Draw(pass, firstQuad, quadCount, sectionIndex)` draws with
  `firstInstance = sectionIndex`; the vertex shader reads that section's origin from a storage buffer. One quad buffer holds many
  sections and drawing one writes no uniform. (A direct draw needs no extra WebGPU feature for firstInstance.)
- **Plants** are two extra quad "faces" in the same 8-byte `PackedQuad` (`BlockRenderShaders.FaceCrossA/B` = 6 and 7): two
  crossed planes cost 16 bytes instead of 48 vertices.
- Every cube face is wound counter-clockwise from outside and shows its texture upright and unmirrored from outside; a test with a
  four-coloured-quadrant texture checks all six.

Dependencies aligned with the consumers: `SpawnDev.ILGPU` 5.3.7 (was 5.2.16), `SpawnDev.SpawnJS.Blazor` 3.0.4 (was 2.1.18). The
27 existing WebGPU render tests pass on them.

### Tests (WebGPU lane, real device, pixel readback; 18 incl. the struct-size and atlas-validation ones)
Orientation of all six faces, tiling across a merged quad, alpha cutout, water blend and no-depth-write, the lowered water top (the
surface edge is PROJECTED and compared with the first lit screen row), double-sided plants from four sides, section origins, tint /
flat colour / untextured / unregistered-type magenta, quadratic fog, sun + ambient lighting, reversed-Z ordering, atlas layer
mapping. Red-checked with twelve mutations (mirrored +X UV, clamp instead of repeat, no discard, water writing depth, lowered top
ignored, plants culled, section origin ignored, linear fog, tint ignored, column-major atlas slicing, wrong water blend factor, sun
ignoring N.L): exactly the twelve aimed tests failed and the four unaffected ones passed.

## 0.1.0-rc.3 (2026-09-20) - Port BlazorJS to SpawnJS

- **Replace `SpawnDev.BlazorJS` 3.5.6 with `SpawnDev.SpawnJS.Blazor` 2.1.18** (brings `SpawnDev.SpawnJS` 2.1.17).
- **Bump `SpawnDev.ILGPU` to 5.2.12** - ILGPU 4.9.x still depended on BlazorJS; 5.2.x is the SpawnJS stack.
- Demo/tests: `AddSpawnJSRuntime` / `SpawnJSRunAsync`; `UnitTesting.Blazor` -> `UnitTesting.Browser` 2.7.0.
- Dispose `GPURenderPipeline.GetBindGroupLayout` wrappers (`using var layout = ...`).

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
