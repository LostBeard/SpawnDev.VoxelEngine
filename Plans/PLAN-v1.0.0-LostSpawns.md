# SpawnDev.VoxelEngine v1.0.0 - Complete Plan for Lost Spawns

**Target:** Production-ready voxel engine powering a DayZ-style survival game in the browser.
**Date:** 2026-04-16
**Author:** Data (Claude CLI #2, Editor)
**First consumer:** Lost Spawns (HD voxel DayZ on Blazor WASM + WebGPU)
**Second consumer:** AubsCraft (Minecraft viewer)
**Test standard:** Every feature has unit tests. Every visual feature has pixel readback tests.

---

## Current State (2026-04-16)

- **843/0/0 tests** across 6 backends (WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU)
- **47 source files**, 19 test files, 20 rendering pixel-readback tests
- **Bugs fixed today:** double-transpose, buffer lifetime crash, 4/6 face culling, standard bind group cache
- **Working:** meshing, face culling, greedy merge, frustum culling, visibility graph, LOD, collision, raycasting, structural integrity, terrain gen, biomes, dynamic uniform batching, reversed-Z, per-vertex AO, quality controller, thermal manager, buffer compaction, draw ordering

## Current State (2026-04-17 delta)

Since yesterday's EOD:

- **Task #7 DMC output wired:** COMPLETE (SdfMeshPipeline pipeline stages all GPU-resident)
- **Task #8 XZ chunk-boundary padding:** COMPLETE (inter-chunk via WorldService.GenerateNeighborBlocksAsync, intra-section via padded slabs in kernel)
- **Task #11 intra-chunk Y section-boundary:** COMPLETE (yPadMinusSlab/yPadPlusSlab optional params on MeshSectionAsync, 233/233 desktop)
- **Task #15 SDF + block hybrid:** library contract SHIPPED (VoxelEngineTestBase.HybridRender.cs, uniform layouts binary-compatible, two-pass composite pattern). WebGPU end-to-end BLOCKED on SpawnDev.ILGPU 4.9.2-rc.7 (sub-word LEA codegen fix queued for publish by Geordi today).
- **SdfRendering_FlatQuad:** FIXED (test geometry mismatch, not a shader bug; quad grown 2x2 -> 8x8, threshold 50 -> 200). Build clean.
- **Blocked items (WebGPU):** 16 SDF/DMC tests + HybridRender end-to-end, all awaiting rc.7 + test authority.

---

## Phase 1: Fix Known Bugs

- [x] **Greedy merge +Y/-Y race condition** - FIXED 2026-04-16. Split dispatch (faces 0-3 then 4-5) + Atomic.And for bit clearing. Full XZ merge preserved, race eliminated.
- [x] **i32 face mask split for WebGPU** - NOT NEEDED. SpawnDev.ILGPU's WGSL transpiler handles i64 Atomic.And as two independent i32 atomicAnd ops on vec2<u32> halves. Verified working in checkerboard test (2026-04-16).
- [x] **Unit test for greedy merge race** - DONE. GreedyMergeGpu_Checkerboard_CoverageMatchesCpu catches the race (showed 224 vs 192 faces before fix).
- [x] **Unit test for i32 face mask split** - NOT NEEDED. ILGPU transpiler handles automatically, verified by all GPU greedy merge tests passing on WebGPU backend.

---

## Phase 2: Test Coverage for Existing Untested Features

- [x] **TransparencyMesher tests** - DONE. 4 tests: transparent/translucent detection, opaque-only negative, face mask generation.
- [x] **CrossQuadMesher tests** - DONE. 3 tests: plant counting, 12 vertices per plant, no-plants-no-vertices.
- [x] **TextureArrayManager tests** - DONE. 4 tests: registration, idempotency, per-face layer indices, variant hash determinism.
- [x] **DynamicLighting tests** - DONE. 4 tests: point/directional creation, facing-brighter, max lights cap.
- [x] **TimeOfDay tests** - DONE. 6 CPU tests + 1 visual pixel test (noon vs midnight brightness).
- [ ] **PBRShaders tests** - deferred to Phase 3 (requires texture pipeline wired in).
- [x] **DamageOverlay tests** - DONE. 3 CPU tests + 1 visual pixel test (damage crack stages).
- [x] **StereoRenderer tests** - DONE. 3 tests: default state, activation, center eye midpoint.
- [x] **WebXRHelper tests** - DONE. 4 tests: default state, session lifecycle, matrix updates, controller ray extraction.
- [x] **FoveatedRendering tests** - DONE. 4 tests: default, thermal critical, unsupported, WebXR foveation values.
- [x] **VoxelMeshPipeline integration test** - DONE (Rendering_FullIntegration_MeshToPixelTest, Phase 1).
- [x] **Visual AO test** - DONE. Pixel readback: max-AO block darker than no-AO block.
- [x] **Visual transparency test** - DONE. Pixel readback: glass lets background blue through, stone doesn't.
- [x] **Orthographic camera** - DONE. ReversedZHelper.CreateReversedOrthographic + OffCenter. 4 tests: reversed-Z depth, no size distortion, visual render, off-center bounds. Used for editor, minimap, shadow maps.

---

## Phase 3: SDF + Dual Marching Cubes (Smooth Terrain)

Captain's order: hybrid smooth terrain + blocky building THIS VERSION. Natural terrain uses SDF evaluated on GPU, meshed via Dual Marching Cubes. Player structures use existing greedy mesh. Both coexist in the same world - carve into smooth hillsides, build walls inside with blocks.

### SDF Data
- [x] **SDF chunk data structure** - `SpawnDev.VoxelEngine/SDF/SdfChunk.cs` uses 16-bit fixed-point at scale 256 (~0.004 precision, -128..+127 range). 32x32x32 = 64 KB per chunk.
- [ ] **SDF storage in ChunkManager** - dual storage: PackedBlock for blocks, SDF grid for terrain (Lost Spawns WorldService not yet SDF-aware)
- [ ] **SDF serialization** - compact format for OPFS region files

### SDF Evaluation (GPU Kernel)
- [x] **Noise primitives** - `SdfNoiseKernels.cs` has Hash3D, ValueNoise3D, FBM, DomainWarpedFBM (GPU-compatible, all backends)
- [x] **Uber Noise composition** - 6 layers in `EvaluateSdfKernel` (base elevation, hills, mountain peaks, fine detail, caves, overhangs)
- [x] **Cave generation** - Layer 5 via `SdfSmoothSubtract` on 3D density noise
- [x] **Smooth blending** - `SdfSmoothUnion` and `SdfSmoothSubtract` implemented (positive-inside convention, k = blend radius)
- [ ] **Biome-driven noise params** - different noise weights per biome (mountains vs plains vs coast)
- [x] **ILGPU kernel** - `EvaluateSdfKernel` dispatches Index3D, one thread per voxel, embarrassingly parallel

### Dual Marching Cubes (GPU Kernel Pipeline)
- [x] **Cell classification kernel** - `DualMarchingCubesKernels.ClassifyActiveCellsKernel`
- [x] **Prefix sum for compaction** - wired in `SdfMeshPipeline.MeshSdfAsync`
- [x] **DMC vertex generation kernel** - `GenerateDualVerticesKernel` + `GenerateQuadsKernel`. Quad output, crack-free at LOD.
- [ ] **Normal computation** - gradient of SDF for smooth per-vertex normals
- [ ] **Material assignment** - SDF value + depth + slope determines material (rock, dirt, grass, sand)

### Integration
- [ ] **DMC vertex buffer integration** - separate `SdfRenderPipeline` exists; unification with `VertexPullPipeline` not yet decided
- [~] **Hybrid render pass** - library contract shipped (`VoxelEngineTestBase.HybridRender.cs`, uniform layouts binary-compatible, two-pass composite validated on CPU). WebGPU end-to-end blocked on ILGPU rc.7.
- [ ] **Triplanar texture projection** - for smooth terrain surfaces (no UV stretching on cliffs)
- [x] **SDF terrain modification kernel** - `ModifySdfSphereKernel` (mode 0 = dig via SmoothSubtract, mode 1 = fill via SmoothUnion)
- [ ] **Chunk re-meshing on modification** - re-run DMC on affected chunks after terrain edit (consumer-side, not yet wired)
- [ ] **LOD for SDF terrain** - coarser SDF grid at distance, DMC produces fewer vertices naturally

### Tests
- [x] **Unit test: SDF evaluation deterministic** - `SdfEvaluate_SameSeedSameInput_ProducesIdenticalOutput` (+ DifferentSeeds, DeepUnderground, HighAltitude)
- [ ] **Unit test: DMC produces manifold mesh** - no T-junctions, correct topology (horizontal plane vertex-count and Y-coord tests exist but no full manifold check)
- [~] **Unit test: cave carving CSG** - `SdfModifySphere_Dig_CarvesHoleInSolid` covers sphere subtraction. Noise-driven cave layer lacks dedicated test.
- [~] **Pixel readback: smooth sphere renders** - `SdfRendering_*` tests exist; fixed FlatQuad geometry 2026-04-17; rest blocked on ILGPU rc.7
- [~] **Pixel readback: terrain + block hybrid** - `HybridRender_BlockAndSdfInSameFrame_BothVisibleTest` written, blocked on ILGPU rc.7
- [ ] **Pixel readback: triplanar texturing** - cliff face doesn't stretch

---

## Phase 4: Texture Pipeline

The solid-color renderer works for prototyping but Lost Spawns needs PBR textures for the DayZ aesthetic.

- [ ] **Wire TextureArrayManager into VertexPullPipeline** - replace solid color lookup with texture array sampling
- [ ] **Texture atlas loader** - load block textures from image files into GPU texture array
- [ ] **Per-face texture mapping** - different textures for top/side/bottom of blocks (grass top, dirt sides)
- [ ] **Texture variant selection** - deterministic hash(blockPos) selects 1 of 4 variants per block type
- [ ] **Normal map support** - second texture layer for bump mapping, tangent-space normals
- [ ] **Roughness map support** - third texture layer for PBR roughness
- [ ] **Mipmap generation** - GPU-side mipmap chain for texture arrays
- [ ] **UV coordinate generation** - vertex shader outputs correct UVs based on quad dimensions and face direction
- [ ] **Pixel readback tests** - render textured block, verify UV correctness (corner pixels match texture corners)
- [ ] **Visual test: textured vs untextured** - verify textured render produces different colors than solid color at same block type

---

## Phase 4: Shadow Mapping

Shadows are the single biggest visual upgrade. Without them the world looks flat - critical for DayZ atmosphere.

- [ ] **Cascaded shadow map (2-3 cascades)** - directional sun shadow, cascade split based on view distance
- [ ] **Shadow map render pass** - depth-only pass from sun's perspective using DepthOnlyShader
- [ ] **Shadow map texture** - depth texture array (one per cascade)
- [ ] **Shadow sampling in fragment shader** - PCF filtering for soft shadow edges
- [ ] **Cascade selection** - fragment shader picks correct cascade based on view-space depth
- [ ] **Shadow bias** - slope-scaled bias to prevent shadow acne
- [ ] **Shadow distance fade** - soft transition at cascade boundaries and max shadow distance
- [ ] **Integration with TimeOfDay** - sun direction drives shadow map VP matrices
- [ ] **Pixel readback test: shadow occlusion** - render block above ground plane, verify ground beneath block is darker than sunlit ground
- [ ] **Pixel readback test: shadow direction** - verify shadow moves with sun position (TimeOfDay)
- [ ] **Performance budget** - shadow pass must fit within QualityController's frame budget

---

## Phase 5: Post-Processing Pipeline

Lost Spawns' art direction requires: SSAO, bloom, NVG, color grading, film grain, vignette.

- [ ] **Post-process render pass infrastructure** - fullscreen quad pass reading from color/depth textures
- [ ] **SSAO (Screen-Space Ambient Occlusion)** - sample depth buffer in hemisphere around each fragment. Upgrades per-vertex AO significantly.
- [ ] **Bloom** - threshold bright pixels, downsample, blur, composite. For fire, explosions, light sources.
- [ ] **NVG post-process** - green/white phosphor tint, noise grain, vignette. Reads from TimeOfDay.IsNVGActive.
- [ ] **Color grading** - desaturated cold blue outdoors, warm near fire. LUT-based or parametric.
- [ ] **Film grain** - subtle noise overlay, DayZ-style grit
- [ ] **Vignette** - darken screen edges for cinematic look
- [ ] **Motion blur** - per-object or camera-based, helps VR comfort
- [ ] **Depth of field** - optional, blurs far objects slightly
- [ ] **Post-process toggle** - each effect individually enable/disable via QualityController
- [ ] **Pixel readback test: SSAO darkens corners** - render corner of room, verify corner pixels are darker than flat surface
- [ ] **Pixel readback test: bloom brightens** - render bright pixel, verify surrounding pixels gain brightness
- [ ] **Pixel readback test: NVG tints green** - enable NVG, verify output is green-tinted
- [ ] **Pixel readback test: color grading shifts hue** - apply cold grade, verify blue shift

---

## Phase 6: Water Rendering

Swimming, rivers, underwater exploration - core DayZ survival gameplay.

- [ ] **Water surface mesh** - separate render pass for water blocks with animated vertex displacement
- [ ] **Water UV animation** - scrolling UVs for surface movement, flow direction support
- [ ] **Fresnel reflectance** - angle-dependent reflection/refraction blend
- [ ] **Screen-space reflections (SSR)** - reflect terrain/sky on water surface
- [ ] **Refraction** - distort underwater view through water surface
- [ ] **Depth-based absorption** - deeper water appears darker/bluer (Beer-Lambert)
- [ ] **Underwater fog** - increase fog density and shift color when camera is below water
- [ ] **Water caustics** - projected light patterns on submerged surfaces
- [ ] **Foam at edges** - white foam where water meets terrain
- [ ] **Pixel readback test: water surface is visible** - render water block, verify non-sky pixels
- [ ] **Pixel readback test: underwater is blue-tinted** - camera below water, verify blue color shift
- [ ] **Pixel readback test: water transparency** - verify objects behind water are visible but tinted

---

## Phase 7: Weather and Sky System

DayZ's weather directly affects gameplay (temperature, visibility, wet clothing).

### Dynamic Sky
- [ ] **Sky dome/quad renderer** - atmospheric scattering or gradient-based sky
- [ ] **Cloud layer** - animated cloud textures, density affects sunlight
- [ ] **Sun/moon disc** - visible sun and moon positions matching TimeOfDay
- [ ] **Moon phases** - affects nighttime brightness
- [ ] **Stars** - visible at night, rotate with time
- [ ] **Dawn/dusk colors** - red/orange/purple sky transitions

### Weather Effects
- [ ] **Rain particle system** - GPU-instanced raindrop particles, splash on surfaces
- [ ] **Snow particle system** - slower falling, accumulation on terrain surfaces
- [ ] **Fog density modulation** - thick fog weather event, reduces visibility to 50m
- [ ] **Wind system** - affects particles, sound propagation, ballistics
- [ ] **Lightning** - flash screen + branching light + delayed thunder sound
- [ ] **Weather state machine** - clear/cloudy/rain/storm/snow transitions with interpolation
- [ ] **Weather affects gameplay** - wet clothing debuff, cold damage, reduced visibility

### Tests
- [ ] **Pixel readback test: rain visible** - enable rain, verify particle pixels exist
- [ ] **Pixel readback test: fog reduces visibility** - verify distant block is less visible in heavy fog vs clear
- [ ] **Pixel readback test: sky color changes with time** - dawn vs noon vs night sky colors differ

---

## Phase 8: World Persistence

Browser memory is limited. Chunk save/load with compression is essential for DayZ-scale maps.

**Storage decision: OPFS region files.** Benchmarked in AubsCraft (2026-04-12):
- OPFS region-batched write: **118 MB/s** (107x faster than IndexedDB)
- OPFS region-batched read: **310 MB/s** (69x faster than IndexedDB, 15x faster than IDB getAll)
- Region file pattern: one OPFS file per 32x32 chunk area (`r.{rx}.{rz}.bin`), fixed-size slots with header offsets. Single file I/O loads up to 1024 chunks at once. SpawnDev.BlazorJS has OPFS wrappers ready.

- [ ] **Chunk serialization format** - compact binary format for section block data
- [ ] **LZ4 compression** - compress chunk data before storage (80-90% reduction)
- [ ] **OPFS region file provider** - IChunkCacheProvider using OPFS region files (32x32 per file, fixed slots, header with offsets)
- [ ] **IndexedDB fallback** - for browsers without OPFS support (detect and fall back)
- [ ] **World metadata** - seed, player position, time of day, weather state
- [ ] **Incremental save** - only save modified chunks, dirty tracking per region
- [ ] **Background save** - save regions on web worker without blocking render (FileSystemSyncAccessHandle)
- [ ] **Memory pressure management** - evict distant chunks from memory, reload on approach
- [ ] **World import/export** - download/upload world saves
- [ ] **Unit test: serialize/deserialize roundtrip** - verify block data survives compression
- [ ] **Unit test: dirty tracking** - verify only modified chunks are re-saved
- [ ] **Unit test: region file packing** - verify multiple chunks pack/unpack correctly from one file

---

## Phase 9: Fluid Simulation

Water flow, lava - core gameplay for survival (rivers as obstacles, flooding bases).

- [ ] **Cellular automaton fluid** - Minecraft-style flow rules (source blocks, flow levels 1-7)
- [ ] **Water source blocks** - placed water that spreads
- [ ] **Flow direction** - water flows downhill, fills spaces
- [ ] **Flow level rendering** - partial-height water blocks based on flow level
- [ ] **Lava** - slower flow, damages entities, ignites flammable blocks
- [ ] **Fluid tick system** - process fluid updates per chunk per tick
- [ ] **Boundary propagation** - fluid spreads across chunk boundaries
- [ ] **Unit test: water flows downhill** - place source block on slope, verify water reaches bottom
- [ ] **Unit test: water fills enclosed space** - verify water stops spreading when contained
- [ ] **Unit test: lava ignites wood** - verify lava adjacent to flammable block triggers destruction

---

## Phase 10: Procedural Generation

Lost Spawns needs a full DayZ-scale procedural world with roads, buildings, and resources.

### Terrain
- [ ] **Multi-octave noise terrain** - Perlin/Simplex noise for natural-looking heightmaps
- [ ] **Biome-aware terrain** - mountain biome has peaks, swamp is flat, coast has cliffs
- [ ] **River generation** - hydraulic erosion or path-finding to carve rivers
- [ ] **Lake generation** - fill depressions with water source blocks
- [ ] **Cave generation** - 3D noise carving underground passages

### Structures
- [ ] **Structure template system** - save/load pre-built structures as templates
- [ ] **Road network** - connect settlements with roads, dirt paths
- [ ] **Building placement** - procedural building selection and placement in towns
- [ ] **Building interiors** - furnished rooms, lootable containers
- [ ] **Military zone generation** - bases, checkpoints, helicopter crash sites
- [ ] **Industrial zone generation** - factories, warehouses
- [ ] **Residential generation** - houses, apartments
- [ ] **Infrastructure** - bridges, power lines, water towers, radio towers

### Resources
- [ ] **Ore distribution** - underground mineable resources
- [ ] **Surface resources** - rocks, sticks, berries, mushrooms
- [ ] **Loot table system** - configurable loot spawns per building type
- [ ] **Unit test: terrain is deterministic** - same seed produces same heightmap
- [ ] **Unit test: structures don't overlap** - verify placement doesn't create intersecting buildings
- [ ] **Unit test: roads connect settlements** - verify pathfinding creates connected road network

---

## Phase 11: VR Integration

Complete the VR stubs with real implementations. Quest 3S is a primary target.

- [ ] **Wire StereoRenderer to VertexPullPipeline** - render left/right eye views
- [ ] **WebXR session lifecycle** - request session, handle frame loop, clean shutdown
- [ ] **Controller input mapping** - Quest controllers mapped to game actions
- [ ] **Hand tracking** - WebXR hand tracking for natural interaction
- [ ] **Passthrough AR mode** - Quest passthrough for tabletop editor view
- [ ] **Foveated rendering integration** - reduce fragment work in peripheral vision
- [ ] **VR comfort settings** - snap turn, smooth turn, vignette during movement
- [ ] **VR UI panels** - world-space UI panels readable in VR
- [ ] **Unit test: stereo VP matrices** - verify left/right eye matrices produce correct parallax
- [ ] **Unit test: foveated resolution** - verify center resolution > edge resolution

---

## Phase 12: Entity Rendering

Players, zombies, animals, dropped items all need a non-voxel rendering path.

- [ ] **Entity mesh renderer** - separate pipeline for non-voxel meshes (glTF/custom format)
- [ ] **Skeletal animation** - bone transforms for character animation
- [ ] **Entity LOD** - billboard sprites at distance, full mesh up close
- [ ] **Entity shadow casting** - entities cast shadows onto voxel terrain
- [ ] **Dropped item rendering** - 3D items on ground with physics
- [ ] **Entity frustum culling** - separate from voxel culling, same frustum
- [ ] **Pixel readback test: entity visible** - render entity mesh, verify pixels

---

## Phase 13: Advanced Optimizations

Performance ceiling - squeeze every TFLOP.

- [ ] **Hi-Z occlusion culling** - GPU depth pyramid for above-ground occlusion rejection (2-4x speedup for forests/buildings)
- [ ] **Multi-draw indirect (MDI)** - GPU fills draw buffer from culling pass, zero CPU draw loop
- [ ] **Geomorphing** - smooth LOD transitions in vertex shader, no popping
- [ ] **Compute-driven culling** - move frustum + occlusion culling to compute shader
- [ ] **Chunk streaming priority** - load/mesh nearest chunks first, background for distant
- [ ] **Memory pool** - pre-allocated GPU buffer pool to avoid alloc/free churn
- [ ] **T-junction fix** - sub-pixel quad expansion to prevent seam artifacts between LOD levels
- [ ] **Indirect draw consolidation** - merge all sections into one indirect draw call (Chrome D3D12 validation win)
- [ ] **Texture streaming** - load textures on demand, placeholder for unloaded
- [ ] **Unit test: Hi-Z rejects occluded** - verify occluded section is culled
- [ ] **Unit test: MDI produces same output as CPU draw loop** - pixel parity test

---

## Phase 14: Flood Fill Lighting

Every engine with good atmosphere has this. Without it, no interior darkness, no torch illumination. Critical for Lost Spawns DayZ feel.

- [ ] **Light propagation BFS** - flood fill from light sources through transparent/air blocks
- [ ] **Sky light** - top-down propagation, attenuates through solid blocks
- [ ] **Block light** - point sources (torches, campfires) with configurable range
- [ ] **Colored lights** - 32 range levels, color mixing (Vintage Story approach: 16,384 combos)
- [ ] **Light data storage** - per-block light level in chunk data (4 bits sky + 4 bits block minimum)
- [ ] **Fragment shader integration** - multiply vertex color by light level
- [ ] **Light update on block change** - re-propagate affected region when blocks placed/destroyed
- [ ] **Smooth lighting interpolation** - average light across block face vertices (no hard edges)
- [ ] **Unit test: torch illuminates room** - place torch in dark room, verify surrounding blocks have light
- [ ] **Unit test: sky light through window** - transparent block transmits sky light
- [ ] **Pixel readback: lit block brighter than unlit** - verify light data reaches shader output

## Phase 15: Palette Compression

4x chunk storage reduction. Most sections use fewer than 16 block types.

- [ ] **Palette encoder** - scan section, build local palette of used block types
- [ ] **Variable bit packing** - 1-4 bits per block when palette is small (1 type = 0 bits, 2 = 1 bit, 4 = 2 bits, 16 = 4 bits)
- [ ] **Section metadata** - palette + bit width stored in section header
- [ ] **Decoder** - unpack to full block array for meshing
- [ ] **OPFS integration** - store compressed sections in region files
- [ ] **Unit test: roundtrip** - compress then decompress, verify identical data
- [ ] **Unit test: size reduction** - uniform section compresses to near zero

## Phase 16: Audio Integration Points

VoxelEngine doesn't own audio but needs to provide data for it.

- [ ] **Material at position query** - what block type is at (x,y,z)? Drives footstep sounds.
- [ ] **Surface under entity query** - what material is the entity standing on?
- [ ] **Occlusion query for sound** - line-of-sight between two points for audio attenuation
- [ ] **Explosion radius query** - what blocks are affected? Drives destruction sounds.
- [ ] **Weather state exposure** - rain/snow/wind state for ambient audio

---

## Visual Test Checklist (Pixel Readback)

Every visual feature must have a pixel readback test that draws to an offscreen target and verifies the output.

### Already Done (20 tests)
- [x] Uniform struct size (128 bytes)
- [x] Matrix byte layout correctness
- [x] Reversed-Z projection math
- [x] PackedQuad shader unpack matching
- [x] Pipeline init state machine
- [x] Single block renders (standard mode)
- [x] Block color match (gray block)
- [x] All 6 face directions render individually
- [x] Fog density affects brightness
- [x] Fog color tints toward specified color
- [x] Directional lighting (sun-facing brighter)
- [x] Voxel size scaling
- [x] Dynamic multi-section batch render
- [x] Section offset moves geometry
- [x] Per-section fog isolation (dynamic uniforms)
- [x] Greedy merge visual parity (merged = individual)
- [x] Full mesh-to-pixel integration
- [x] Buffer lifetime (destroy/recreate between frames)
- [x] Block type colors distinct
- [x] Reversed-Z depth ordering

### Still Needed
- [ ] Textured block rendering (texture array)
- [ ] Normal map effect on lighting
- [ ] Shadow occlusion (block casts shadow on ground)
- [ ] Shadow direction follows sun
- [ ] SSAO darkens room corners
- [ ] Bloom brightens surrounding pixels
- [ ] NVG tints output green
- [ ] Color grading shifts hue
- [ ] Water surface visible
- [ ] Underwater blue tint
- [ ] Rain particles visible
- [ ] Sky color changes with time of day
- [ ] Transparency sorting (translucent behind opaque)
- [ ] Cross-quad plant rendering
- [ ] Damage overlay crack visibility
- [ ] LOD geomorphing smoothness
- [ ] Stereo rendering left/right parallax
- [ ] Entity mesh visible
- [ ] Hi-Z correctly rejects occluded section
- [ ] MDI output matches CPU draw loop

---

## Version Milestones

| Version | Scope | Gate |
|---------|-------|------|
| **v0.5.0** | Phases 1-3 done. Bug fixes, test coverage, SDF + DMC smooth terrain. | Smooth terrain renders, hybrid block+smooth works. |
| **v0.6.0** | Phase 4 done. Texture pipeline (PBR, normal maps, triplanar). | Textured terrain + blocks. |
| **v0.7.0** | Phases 5-6 done. Shadows + post-processing. DayZ atmosphere. | Shadow + SSAO pixel tests pass. |
| **v0.8.0** | Phases 7-9 done. Water, weather, persistence. Playable world. | Water renders, world saves/loads. |
| **v0.9.0** | Phases 10-13 done. Fluids, procgen, VR, entities. Full feature set. | All gameplay features implemented. |
| **v1.0.0** | Phases 14-16 done. Lighting, compression, optimizations, audio. Ship-ready. | 100% test pass, performance targets met. |

---

*Make it so.* 🖖
