# SpawnDev.VoxelEngine Changelog

## Unreleased (0.2.0-rc.2) - quad buffers shared by many meshes

AubsCraft on an Intel UHD 620 laptop: 837 draws AND 837 buffer binds a frame (one quad buffer per column and pass), 17.6 ms of the 19.1 ms
frame spent recording them. After `EmbedSectionIndex` removed the per-section draws, the per-mesh buffer was the remaining multiplier.

### Added
- **`QuadArena`**: a few large GPU buffers (blocks of 2 Mi quads = 16 MiB by default) that hold the quad lists of many meshes.
  `ColumnMesher.Arena` / `HeightmapMesher.Arena`: the compact pass writes each list straight to its place in a block (no extra copy, no
  buffer created per mesh), and `Draw(pass, list.FirstQuad, list.Count, slot)` draws it. Hundreds of columns share one buffer, so a pass
  binds once instead of once per column. A block is added when nothing fits (a list longer than a block gets a block of its own);
  disposing a mesh gives its ranges back, a second Dispose is harmless. Null (the default): each list keeps its own buffer, nothing changes.
- **`ColumnMesh.QuadList`**: `FirstQuad`, `IsPooled`, `ArenaBlock`, and `CopyToHostAsync()` (exactly the list's quads, pooled or not).
- **`RangeAllocator`** (`Buffers/`): the best-fit range allocator with merging behind the arena; pure bookkeeping.
- Tests (CPU, Wasm and WebGPU lanes): the allocator against a brute-force model (20k random allocate/free steps, no overlap, totals),
  best-fit vs first-fit, double free, merge; pooled meshes hold exactly the quads of unpooled ones (as a set: the merge appends with
  atomics, so the order differs between meshings), match the CPU reference across several tiny blocks and after frees, oversize lists,
  the heightmap mesher; and a WebGPU pixel test: two different columns in one block, each drawn from its own range at its own origin.
  Eight mutations of the allocator and the arena were each caught by at least one test.

### Notes
- Dispose the arena only after the accelerator has finished the last mesh (`await accelerator.SynchronizeAsync()`): `MeshAsync` flushes its
  compaction without waiting for it, and the Wasm backend frees a disposed buffer at once (a stale dispatch then writes into whatever
  was allocated there next; the A/B that showed it changed only that one line of a test).

## 0.2.0-rc.1 (2026-10-15) - AubsCraft's viewer release: textured block rendering, GPU column and heightmap meshers, fewer calls per frame

The four local builds below (0.2.0-local.1 to local.4) are this release; AubsCraft's 3D viewer has run on them in production since 2026-10-14.

### Fixed in the release build
- **`ColumnMesher` / `HeightmapMesher` on the Wasm backend: a second mesh on the same instance could report every quad twice.** Each mesh
  cleared its counters with a synchronous `MemSetToZero`, which the browser backends do not order after the previous run's still-queued
  kernels (the Wasm backend applies it immediately while the previous compaction may still be reading the counters). Now the counters are
  reset with `CopyFromCPU`, which every backend orders after pending work. Found by the full multi-lane run
  (`WasmTests.ColumnMesher_SingleBlock_SixQuadsAndNothingElseTest`, deterministic); a synchronize between the two meshes made it pass, which
  pinned the cause. WebGPU (what AubsCraft runs) was never affected.

### Known issues
- 39 tests of the older kernels fail on the WebGL lane (FaceCull, Compaction, GreedyMergeGpu, Dmc*, FrustumCull: WebGL has no atomics or
  shared memory). The identical set fails on the commit before this release (ILGPU 5.2.16), so they are not caused by it. The new
  `BlockRenderPipeline`, `ColumnMesher` and `HeightmapMesher` tests are skipped on WebGL (they need atomics / scatter stores).
- Full run: 2055 tests, 1749 passed, 267 skipped, 39 failed (all WebGL, pre-existing).

### Built as 0.2.0-local.4 - fewer calls per frame

AubsCraft's viewer profile (CPU time to RECORD the frame was the whole frame: 9-18 ms for 2400-4100 draws and 700-1100 buffer binds)
showed the cost is calls into the browser, not the GPU. Two changes:

- **Section index inside the quad.** `PackedQuad` bits 47-51 hold the quad's 16-high section within its column
  (`PackedQuad.Pack(..., sectionIndex)`, `UnpackSectionIndex`, `WithSectionIndex`); `BlockRenderShaders` lifts the quad by
  `sectionIndex * 16`. `ColumnMesher.EmbedSectionIndex` / `HeightmapMesher.EmbedSectionIndex` (off by default: nothing changes for
  existing callers) write it during compaction, so ONE draw covers a whole column (or far chunk) instead of one per section, with a
  single origin per column. Tests: field round trip; every meshed quad carries its section and the unit faces are unchanged (column and
  heightmap meshers); a pixel test that a quad of section 1 draws 16 blocks up.
- **`BlockRenderPipeline.BindPass` skips repeats**: the pipeline + frame group are set only when the pass kind (or the pass) changes,
  the quad-buffer group only when the buffer does (3 browser calls became 0-1). Test: a sequence that repeats a bind, changes only the
  buffer, changes the kind, and returns to an earlier one - all draws land.
- 7 single mutations of the above, all killed (one needed a stronger test first: a buffer change within one pass kind).

### Built as 0.2.0-local.3 - HeightmapMesher (the far tier)

- **`HeightmapMesher`**: a chunk's heightmap (top block per column, plus the seabed under water) meshed on the GPU into
  `PackedQuad`s per 16-high section - the same `ColumnMesh` result, so a far chunk is drawn exactly like a meshed column, by section
  origin. One thread per column: a top face, plus a wall over the blocks above each neighbour's solid top (split at section borders);
  water gets a water-pass top face and its seabed becomes a solid column of type + `SeabedTypeOffset` (so a viewer can darken what
  lies under water). Plants, air and unknown types draw nothing. The heightmap goes in as a GPU buffer (`HeightmapLayout`, 3072 bytes,
  from a JS typed array with `CreateHeightmapBufferFromJS`: no .NET copy); the four neighbouring chunks' heightmaps are optional and
  decide the border walls (a missing neighbour is bottomless: the wall reaches the world floor). One counter readback per chunk.
- Tests (10, WebGPU + CPU lanes): the GPU quads expanded to unit faces must equal a plain-rules CPU reference exactly: flat chunk with
  and without neighbours, terrain with lakes/plants/air holes for every neighbour combination, shore walls reaching the seabed,
  a worst-case checkerboard through every section (the scratch bound), reuse after a busy chunk, wrong-size rejection, JS upload,
  a type emitting both passes is solid. 21 single mutations, all killed.

### Built as 0.2.0-local.2 - ReleaseQuadBuffer

- **`BlockRenderPipeline.ReleaseQuadBuffer(GPUBuffer)`**: drops one quad buffer's cached bind group. A viewer re-meshes and evicts
  columns all the time, and the cache otherwise keeps a bind group for every buffer ever drawn. `CachedQuadBindGroups` reports the
  count. Test: bind -> 1, release + destroy -> 0, a later buffer draws and counts 1, releasing an unbound or already released
  buffer is harmless; the no-op mutation is killed by it.

### Built as 0.2.0-local.1 - Textured block rendering

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

- **`ColumnMesher`: a whole block column meshed on the GPU, class-aware, all sections per dispatch.** `VoxelMeshPipeline` meshes one
  16^3 section per call (CPU padding, a counter readback each); a Minecraft column is 24 sections. `ColumnMeshKernels` are the
  batched form of the same bit-column face culling and greedy merge (occupancy -> face cull -> merge +X-X+Z-Z, then +Y-Y -> plants ->
  compaction), each a flat Index1D covering every section at once, so a column is a handful of dispatches and ONE tiny readback (the
  75 counters that size the output). The four neighbouring columns' edge cells sit in a ring around each section, so culling is exact
  across column borders; a missing neighbour counts as solid (no faces toward an unloaded column) or open (`MissingNeighborsAreSolid`).
- **`BlockClass`**: per-type flags (emit / occlude per pass, plant). A face is drawn where a block emits and its neighbour does not
  occlude: leaves and glass emit but do not occlude, water has a pass of its own, plants are two crossed-plane quads (faces 6, 7).
  Output: `ColumnMesh` = opaque / plants / water `QuadList`s, exact-size GPU buffers with per-section start offsets. Scratch is
  4096 cells x 6 faces = 24576 quads per section, a proven bound; nothing is silently dropped (`Overflow` is reported, must be 0).
- **`ColumnMesher.CreateColumnBufferFromJS`**: a column from a JS `Uint8Array` straight to the GPU buffer (`CopyFromJS`), never through
  the .NET heap.
- Tests (12): the GPU mesh is expanded back to unit faces and must equal a brute-force CPU reference exactly (each face covered once,
  same type): single block, a slab that merges to the minimum, class rules one by one, neighbour columns, noise columns x neighbour
  variants, 24-section terrain columns, the 3D-checkerboard worst case (exactly the scratch bound), reuse after a busy column, wrong-size
  rejection, JS typed-array upload. WebGPU lane 12/12, CPU lane 11/11. Mutation-checked one at a time (16 mutations: 15 killed, 1
  equivalent - the merge consumes every mask bit, so stale masks cannot exist).

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
