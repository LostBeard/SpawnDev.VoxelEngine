# SpawnDev.VoxelEngine - Complete Implementation Plan

**Target:** v1.0.0 (the only version - Rule 1: every release is the final release)
**Date:** 2026-04-14
**Author:** Data (Claude CLI #2)
**Reviewed by:** Tuvok (APPROVED with adjustments), Geordi (critical issues identified)
**For:** Captain (TJ), full team
**Repo:** https://github.com/LostBeard/SpawnDev.VoxelEngine

---

## Context

Both AubsCraft (Minecraft world viewer) and Lost Spawns (HD voxel survival) independently fight the same rendering problems - vertex buffer overflow, missing geometry, no LOD, hardcoded limits. SpawnDev.VoxelEngine extracts the shared GPU infrastructure into one library. Fix once, both projects benefit. Both target Quest 3S (8GB shared RAM, mobile VR).

**Current state:** 105/105 tests passing on all 6 backends (face cull GPU kernels, greedy merge CPU reference, vertex pool). Frustum cull coded but tests have matrix math bug. Two known bugs: greedy merge GPU race condition, frustum test matrix.

---

## Review Feedback Incorporated (Tuvok + Geordi)

Key changes from initial draft:
1. **Section-based architecture** (Geordi): All meshing operates on 16x16x16 sections, not full columns. PackedQuad Y = 0-15 (4 bits). AubsCraft already shipped this pattern (commit fb9d0fd).
2. **+Y/-Y race fix redesigned** (Geordi): Dispatch per-XZ-column (iterates Y internally), not per-Y-layer. Eliminates race without atomics.
3. **Device capabilities moved earlier** (Tuvok): Now Sprint 2 alongside buffer management.
4. **CullResult contract defined early** (Geordi): Section-based format agreed before rendering/culling diverge.
5. **Hi-Z made optional** (Geordi): Graph visibility handles 95% of cases. Hi-Z deferred, not v1.0 blocker.
6. **Pick geomorphing over skirts** (Tuvok): Skirts as fallback only.
7. **Block data = ArrayView<int>** (Tuvok): Consumers promote at boundary. Simple, proven.
8. **Streaming upload** (both): Verify WebGPU needs double-buffering before building. queue.writeBuffer may suffice.

---

## Foundational Decisions

| Decision | Rationale |
|----------|-----------|
| **Section-based (16x16x16)** | Matches visibility graph, PackedQuad Y fits in 4 bits, AubsCraft already uses this |
| **ArrayView<short> for blocks (PackedBlock)** | 12-bit block type + 4-bit damage. Sub-word types (v4.9.0). Lost Spawns needs damage states (DayZ walls take 50 hits). AubsCraft uses type-only (damage=0). |
| **Variable voxel size (config)** | Lost Spawns = 0.5m (LOCKED), AubsCraft = 1.0m. Affects vertex positions, UV tiling, LOD super-block size. |
| **Block registry on GPU** | Storage buffer of BlockProperties (hardness, blast resistance, transparency, flags). Kernel reads for face emission, game reads for interaction. |
| **Chunk height = kernel param** | Lambda capture makes it compile-time inlined. Zero overhead. |
| **i32 atomics only on WebGPU** | i64 emulated as vec2<u32>, atomics fall back to non-atomic stores |
| **Per-XZ-column dispatch for +Y/-Y merge** | One thread owns entire column, iterates Y internally. No race. |
| **Transparency = reverse draw order** | Same RadixSort keys, iterate backward. One sort, two passes. |
| **PBR textures (diffuse + normal + roughness)** | Lost Spawns = HD realistic (DayZ style). Texture array with 3 layers per block type per face. |
| **4 texture variants per block type** | Deterministic hash(blockPos) breaks visual monotony. Stone/grass don't look identical everywhere. |

---

## Lost Spawns HD Requirements (Tuvok, 2026-04-13)

These are NOT optional. Lost Spawns targets DayZ in voxel form - realistic, dark, destructible.

### Block Data Format: PackedBlock (ushort)
```
bits [0:11]  = block type (12 bits, 0-4095)
bits [12:15] = damage level (4 bits, 0-15)
```
- `ArrayView<short>` in kernels (sub-word types, v4.9.0)
- Extract: `blockType = packed & 0xFFF`, `damage = (packed >> 12) & 0xF`
- AubsCraft: damage always 0, type uses lower 12 bits
- Promote from int[]: `packedBlock = (short)(blockType & 0xFFF)` (no damage for AubsCraft)

### Block Registry (GPU storage buffer)
```csharp
struct BlockProperties {
    float Hardness;        // seconds to break with bare hands
    float BlastResistance; // explosion resistance
    byte SoundType;        // 0=stone, 1=wood, 2=metal, 3=dirt, 4=sand, 5=glass
    byte Flammability;     // 0=none, 1=slow, 2=fast
    byte Transparency;     // 0=opaque, 1=transparent, 2=translucent
    byte Flags;            // gravity, liquid, plant, structural
}
```

### Variable Voxel Size
```csharp
config.VoxelSize = 0.5f;  // Lost Spawns (half-meter cubes)
config.VoxelSize = 1.0f;  // AubsCraft (standard blocks)
```
Vertex positions: `worldPos = sectionOrigin + blockPos * voxelSize`
UV tiling: `fract(worldPos / voxelSize)`
LOD: 2x2x2 of 0.5m = 1.0m super-block

### PBR Material Layers (per block type per face)
- Layer N: diffuse (muted, desaturated, weathered)
- Layer N+1: normal map (per-face depth without extra geometry)
- Layer N+2: roughness/metallic (optional)
- Fragment shader: `finalColor = diffuse * lighting(normal_from_map, roughness)`

### Texture Variants (4 per block type)
```wgsl
let variant = hash(blockPos) % 4u;
let baseLayer = blockType * textureLayersPerType + variant * 3u; // *3 for diffuse+normal+roughness
```

### Block Damage Rendering
- 4-5 crack overlay textures (increasing severity)
- Blended on top of block texture based on damage bits
- Persists until repaired (not Minecraft break-and-vanish)

### Dynamic Lighting (design for, build in Sprint 10)
- Point lights (torch ~10m, campfire ~15m, flashlight cone ~20m)
- Max 8-16 active lights as uniform buffer
- Attenuation: `1.0 / (1.0 + dist * dist * falloff)`
- Pipeline must support adding lights later

### Night Rendering
- Ambient drops to near-zero at night
- Only lit areas visible (pitch black otherwise)
- NVG post-process: green phosphor, noise, limited range
- Time-of-day uniform (Captain has this in AubsCraft)

### Destruction
- `DestroyBlocksInSphere(center, radius)` - modifies block grid
- Triggers re-mesh of affected sections
- Structural integrity check post-destruction (flood fill to ground)
- Unsupported blocks become falling entities

### Underground Bases (Lost Spawns differentiator)
- Cave culling (Sprint 5) essential
- Interior darkness (enclosed spaces dark without light sources)
- X-ray debug/admin toggle
- Ventilation: connected-to-surface check (reuses cave cull flood fill)

---

## 9 Sprints, 33 Phases, Dependency-Ordered

### Sprint 1: Fix Bugs + Complete Meshing (Phase 1)

**1.1 Section-Based Architecture + PackedBlock Format**
- Change PackedQuad Y from 6 bits (0-63) to 4 bits (0-15) for section-local coordinates
- Reclaim 2 bits -> expand block type to 12 bits + 4 bits damage in PackedBlock
- PackedBlock (ushort): `blockType = packed & 0xFFF, damage = (packed >> 12) & 0xF`
- All kernels accept `ArrayView<short>` blocks (sub-word types, v4.9.0)
- Kernel extracts block type via `blockType = blocks[idx] & 0xFFF` for face comparison
- Update VoxelMeshConstants: add DefaultSectionHeight = 16, MaxBlockTypes = 4096
- Add VoxelSize as config parameter (0.5f Lost Spawns, 1.0f AubsCraft)
- All kernels operate on 16x16x16 sections, dispatched per-section
- Modify: `Meshing/PackedQuad.cs`, `Meshing/VoxelMeshConstants.cs`, all kernel files
- New file: `PackedBlock.cs` (Pack/Unpack helper, matching PackedQuad pattern)
- Update all existing tests for section-based dimensions + short block data

**1.2 Fix Greedy Merge GPU Kernel (two changes)**
- **Change A - Fix +Y/-Y dispatch:** Change from [height, 2] to [chunkXZ*chunkXZ, 2]. Each thread = one XZ column, iterates all Y layers internally. Eliminates the race without needing atomics for mask clearing within a column.
- **Change B - i32 face masks for WebGPU:** Split long[] into int[] pairs (lo/hi 32 bits). MergePerpendicularPlane's extend-width loop reads neighbor columns via i32 reads. The inter-column reads are safe because each column is owned by its own thread (from Change A), so no concurrent modification.
- New file: `Meshing/FaceMaskSplitter.cs`
- Modify: `Meshing/GreedyMergeKernels.cs`
- New tests: GPU greedy merge vs CPU reference on ALL 6 backends
- File: `Demo.Shared/UnitTests/VoxelEngineTestBase.GreedyMergeGpu.cs`

**1.3 Fix Frustum Cull Test Matrix Math**
- Trace known point through VP matrix, verify plane normals point inward
- Read System.Numerics.Matrix4x4 docs for CreateLookAt/CreatePerspective conventions
- Modify: `Culling/FrustumCullKernels.cs`
- Fix all 4 existing frustum cull tests

**1.4 Face Masking**
- 6 separate atomic counters in greedy merge (one per face direction)
- ChunkMeshInfo struct: FaceGroupOffset[6], FaceGroupCount[6]
- FaceMaskService: dot(faceNormal, cameraToChunk) -> skip back-facing groups
- Realistic savings: 15-25% surface viewing, higher underground (Geordi's correction)
- New file: `Meshing/FaceMaskService.cs`
- Tests: Camera-direction-based culling, total visible < total

### Sprint 2: Buffer Management + Device Capabilities (Phase 2)

**2.1 Define CullResult + DrawCommand contracts early**
- SectionEntry: {Cx, Sy, Cz, QuadOffset, QuadCount, Connectivity}
- CullResult: {VisibleIndices, VisibleCount}
- DrawCommand: {VertexCount, InstanceCount, FirstVertex, FirstInstance} (matches WebGPU indirect)
- New file: `Contracts.cs` or in respective directories
- These types are consumed by rendering AND culling - must be stable

**2.2 Device Capabilities (moved from Sprint 4)**
- Query MaxBufferSize, MaxStorageBufferBindings at init
- Derive: vertex budget, chunk budget, pool bucket count, draw command limit
- Classify: Desktop / MobileHigh / MobileLow
- New file: `Adaptive/DeviceCapabilities.cs`
- Tests: Budget math, tier classification

**2.3 Indirect Draw Buffer**
- ALL commands in ONE GPUBuffer (300x Chrome validation win)
- O(1) add/remove via end-swap
- Device-adaptive max draw count from DeviceCapabilities
- New file: `Buffers/IndirectDrawBuffer.cs`
- Tests: Add/remove/update, capacity, O(1) removal integrity

**2.4 Buffer Compaction**
- GPU-copy live data to front via compute kernel when fragmentation > 30%
- Uses ILGPU Scan (prefix sum) for compacted offsets
- New file: `Buffers/BufferCompactionKernels.cs`
- Tests: Fragmented pool -> compact -> data integrity

**2.5 Streaming Upload (verify need first)**
- Benchmark WebGPU queue.writeBuffer vs manual double-buffer
- If WebGPU handles staging internally, skip manual ring buffer
- Only build if benchmarks show a need
- Conditional file: `Buffers/StreamingUploadBuffer.cs`

### Sprint 3: Rendering Foundation (Phase 3)

**3.1 Reversed-Z Depth**
- depth32float, depthCompare: greater, clearValue: 0.0
- Infinite far plane projection (only needs zNear)
- New file: `Rendering/ReversedZHelper.cs`
- Tests: Depth ordering correctness

**3.2 Vertex Pulling Pipeline**
- WGSL vertex shader unpacks PackedQuad from storage buffer
- quadIdx = vertexID/6, cornerIdx = vertexID%6
- Fragment shader: texture array lookup OR solid color (configurable)
- Support BOTH texture arrays (preferred) AND atlas (AubsCraft compat, per Geordi)
- New file: `Rendering/VertexPullPipeline.cs`
- Tests: Compute kernel mimicking vertex shader + screenshot comparison via Playwright

**3.3 Draw Ordering via RadixSort**
- ILGPU RadixSort on distance keys, produces sorted draw order
- Front-to-back for opaque, reverse for transparent (same sort, two passes)
- New file: `Rendering/DrawOrderKernels.cs`
- Tests: Nearest-first ordering, reorder correctness

### Sprint 4: Quality Controller (Phase 4)

**4.1 Quality Controller**
- Frame time EMA, asymmetric rates (fast decrease 4x, slow increase 1x)
- 5 levels: Ultra -> High -> Medium -> Low -> Minimal
- Draw distance feedback loop
- Thermal detection (frame time spike without scene change)
- New file: `Adaptive/QualityController.cs`
- Tests: Over-budget decrease, under-budget increase, thermal detection, emergency

**4.2 Streaming Budget**
- Max vertices/uploads/bytes per frame (from DeviceCapabilities)
- Priority queue: distance + view direction + LOD urgency
- New file: `Adaptive/StreamingBudget.cs`
- Tests: Budget enforcement, priority ordering

### Sprint 5: Culling Pipeline (Phase 5)

**5.1 Graph-Based Visibility (Sodium-style)**
- 48-bit per-section visibility encoding via flood fill
- BFS from camera, outward-only, bitwise getConnections
- CPU-side (runs before GPU culling)
- New file: `Culling/VisibilityGraph.cs`
- Tests: Open/solid/tunnel/cave scenarios

**5.2 Culling Pipeline Orchestrator**
- Integrates: graph BFS (CPU) -> fog (GPU) -> frustum (GPU) -> face masking
- Hi-Z slot reserved but NOT implemented in v1.0 (Geordi recommendation)
- New file: `Culling/CullingPipeline.cs`
- Tests: Full pipeline produces correct visible set

**5.3 Hi-Z Occlusion (OPTIONAL, deferred)**
- Mip chain from depth, AABB testing in compute
- Only build if graph visibility + frustum is insufficient
- New files: `Culling/HiZOcclusionKernels.cs`, `Culling/VisibilityBuffer.cs`

### Sprint 6: LOD System (Phase 6)

**6.1 LOD Reduction Kernel**
- DelegateSpecialization: one mesh kernel, LOD level parameter
- Super-block = most common non-air in NxNxN group
- Section-based: 16x16x16 -> 8x8x8 -> 4x4x4 -> 2x2x2
- New file: `LOD/LODReducer.cs`
- Tests: Type preservation, reduced dimensions, mesh coverage

**6.2 LOD Selection**
- SelectLOD(distance, pressure) with hysteresis
- New file: `LOD/LODSelector.cs`
- Tests: Distance tiers, pressure shift, no oscillation

**6.3 Geomorphing (primary LOD transition method)**
- morphFactor = smooth blend in transition zone
- Vertex shader interpolates between LOD N and LOD N+1 positions
- New file: `LOD/Geomorphing.cs`
- Tests: morphFactor=0 matches LOD N, morphFactor=1 matches LOD N+1

**6.4 LOD Skirts (fallback only)**
- Only implement if geomorphing causes visual issues
- New file: `LOD/LODSkirts.cs`

### Sprint 7: Visual Completeness (Phase 7)

**7.1 Transparency** - Dual occupancy masks, separate mesh pass, back-to-front via reversed sort
- New file: `Meshing/TransparencyMesher.cs`

**7.2 Plants/Cross-Quads** - X-shaped quads, non-greedy
- New file: `Meshing/CrossQuadMesher.cs`

**7.3 Per-Vertex AO** - 3 corner neighbors -> AO 0-3, packed in PackedQuad reserved bits
- New file: `Meshing/AmbientOcclusion.cs`

**7.4 T-Junction Fix** - Sub-pixel quad expansion (1/512 block = 0.001953125 units, derived from minimum visible gap at max resolution). Must account for neighbor block types (Geordi's note).
- Integrated into VertexPullPipeline.cs

**7.5 Texture Array Manager** + UV Generation
- texture_2d_array, per-(blockType,face) layer mapping
- Support atlas fallback for AubsCraft compatibility
- fract(worldPos) tiling in fragment shader
- New file: `Rendering/TextureArrayManager.cs`

### Sprint 8: Chunk Management + Integration API (Phase 8)

**8.1 Chunk State Machine** - UNLOADED -> LOADING -> MESHING -> GPU_READY -> VISIBLE -> CACHED
- Section-based: (cx, sy, cz) keys
- New file: `ChunkManager.cs`

**8.2 Cache Integration** - IChunkCacheProvider interface
- New file: `Caching/ChunkCacheProvider.cs`

**8.3 Service Registration** - AddVoxelEngine() DI extension
- New file: `VoxelEngineServiceExtensions.cs`

**8.4 Configuration** - VoxelEngineConfig (chunk size, height, LOD, quality, classifiers)
- Chunk height as lambda-captured kernel parameter (Tuvok's solution)
- New file: `VoxelEngineConfig.cs`

**8.5 Coordinate Conventions Doc** (Geordi's request)
- Section-local (0-15), chunk-local (0-chunkXZ), world-space mapping
- Library only speaks section-local and chunk-local
- New file: `Research/CoordinateConventions.md`

### Sprint 9: HD Rendering Quality (Phase 9) - Lost Spawns Requirements

**9.1 PBR Material Pipeline**
- Texture array with 3 layers per block type per face (diffuse + normal + roughness)
- Fragment shader: sample diffuse, read normal from normal map, compute lighting with roughness
- Normal map transforms per-face tangent space to world space
- Roughness controls specular highlight size/intensity
- New file: `Rendering/PBRMaterialPipeline.cs`
- Tests: Known material produces expected lighting output, normal map flips normals correctly

**9.2 Block Damage Rendering**
- 4-5 crack overlay textures at increasing severity
- Fragment shader reads damage bits from PackedBlock, blends crack overlay
- `crackAlpha = textureSample(crackAtlas, uv, damageLevel).a`
- `finalColor = mix(blockColor, crackColor, crackAlpha)`
- Damage persists (DayZ style, not Minecraft break-and-vanish)
- New file: `Rendering/DamageOverlay.cs`
- Tests: damage=0 no overlay, damage=15 full crack, intermediate levels proportional

**9.3 Texture Variant System**
- 4 texture variants per block type (deterministic hash selection)
- `hash(blockPos) % 4` selects variant, embedded in fragment shader
- Breaks visual monotony on large surfaces (grass fields, stone walls)
- Hash must be deterministic (same block position = same variant across sessions)
- Integrated into VertexPullPipeline.cs fragment shader
- Tests: Same position always produces same variant, distribution roughly uniform

**9.4 Dynamic Point Lights**
- Uniform buffer of up to 16 active lights: {position, color, radius, falloff}
- Per-fragment lighting: iterate lights, compute attenuation, accumulate
- Attenuation: `1.0 / (1.0 + dist * dist * falloff)`
- Light types: point (torch, campfire), spot (flashlight), directional (sun/moon)
- New file: `Rendering/DynamicLighting.cs`
- Tests: Single light illuminates correct radius, falloff curve matches formula, no light = ambient only

**9.5 Night Rendering + Time of Day**
- Time-of-day uniform controls ambient light level and sun direction
- Night: ambient drops to near-zero, only point lights visible
- Dawn/dusk: warm directional light at low angle
- Fog color shifts with time of day
- NVG post-process mode: green phosphor filter, noise grain, limited range
- New file: `Rendering/TimeOfDay.cs`
- Tests: t=0 (midnight) ambient near zero, t=0.5 (noon) full ambient, interpolation correct

**9.6 Block Registry (GPU storage buffer)**
- Upload BlockProperties[] as storage buffer accessible to kernels
- Kernel reads for face emission decisions (transparency, plant flags)
- Game reads for interaction (hardness, blast resistance, sound type)
- New file: `BlockRegistry.cs`
- Tests: Registry lookup returns correct properties, transparency classification correct

### Sprint 10: Terrain Generation + World Systems (Phase 10)

**10.1 Heightmap Terrain Provider**
- Read binary heightmap (Int16 per cell, meters above sea level)
- Bilinear interpolation between cells
- Slope computation from neighbor heights
- Promote from Lost Spawns' HeightmapLoader.cs to engine
- New file: `Terrain/HeightmapTerrainProvider.cs`
- Tests: Known heightmap produces correct block grid, interpolation smooth

**10.2 Biome System**
- Elevation + slope + moisture -> biome -> block types + vegetation rules
- Beach (<3m above sea), Cliff (slope>=4), Rocky summit (>55m), Sparse veg (>45m, slope>=2), Forest (else)
- Configurable biome rules (not hardcoded to Deer Isle)
- New file: `Terrain/BiomeAssigner.cs`
- Tests: Flat at sea level = beach, steep = cliff, high = summit

**10.3 Destruction System**
- `DestroyBlocksInSphere(center, radius)` modifies block grid
- Sphere test on GPU: one thread per block, check distance to center
- Marks destroyed blocks as air, triggers re-mesh of affected sections
- Returns list of affected section keys for re-mesh
- New file: `Destruction/ExplosionKernels.cs`
- Tests: Sphere centered on block destroys correct count, edge blocks preserved

**10.4 Structural Integrity**
- Post-destruction flood fill from ground upward
- Blocks not connected to ground -> mark for collapse
- GPU flood fill kernel (similar to cave visibility flood fill)
- Collapsed blocks become game-level falling entities (not engine responsibility)
- New file: `Destruction/StructuralIntegrity.cs`
- Tests: Pillar with base removed -> entire pillar collapses, bridge with supports -> stays

### Sprint 11: VR/AR (Phase 11)

**11.1 Stereo Rendering** - Combined frustum, shared mesh, two VP matrices
- New file: `VR/StereoRenderer.cs`

**11.2 WebXR Helpers** - Session setup via SpawnDev.BlazorJS
- New file: `VR/WebXRHelper.cs`

**11.3 Quest 3S Thermal Management**
- New file: `Adaptive/ThermalManager.cs`

**11.4 Fixed Foveated Rendering Integration**
- WebXR FFR session configuration
- Voxel-specific: FFR helps fragment-bound scenes (PBR lighting), less for vertex-bound
- Quality controller adjusts FFR level with thermal state
- New file: `VR/FoveatedRendering.cs`

**1B. Fix Frustum Cull Test Matrix Math**
- Problem: CPU reference returns 0 visible for chunks in front of camera
- Root cause: Gribb-Hartmann plane extraction + System.Numerics.Matrix4x4 coordinate system alignment
- Fix: Read the actual Matrix4x4 docs, trace a known point through VP matrix, verify plane normals point inward
- Modify: `Culling/FrustumCullKernels.cs` - ExtractFrustumPlanes and/or BuildTestViewProj
- Tests: Fix all 4 existing frustum cull tests

**1C. Face Masking**
- 6 separate atomic counters (one per face direction) in greedy merge
- Store FaceGroupOffset[6] and FaceGroupCount[6] per chunk
- At draw time: dot(faceNormal, cameraToChunk) < 0 -> skip group. 36% reduction.
- New file: `Meshing/FaceMaskService.cs`
- Tests: Camera facing +Z skips -Z group, total visible quads < total quads
- File: `Demo.Shared/UnitTests/VoxelEngineTestBase.FaceMasking.cs`

### Sprint 2: Buffer Management

**2A. Indirect Draw Buffer**
- ALL draw commands in ONE GPUBuffer (300x Chrome D3D12 validation win)
- Layout: [vertexCount:u32, instanceCount:u32, firstVertex:u32, firstInstance:u32] per draw
- O(1) add/remove via end-swap
- New file: `Buffers/IndirectDrawBuffer.cs`
- Tests: Add/remove/update commands, capacity limits, O(1) removal integrity

**2B. Buffer Compaction**
- When fragmentation > 30%, GPU-copy live data to front via compute kernel
- Uses ILGPU Scan (prefix sum) for compacted offset computation
- New file: `Buffers/BufferCompactionKernels.cs`
- Tests: Create fragmented pool, compact, verify all live data preserved

**2C. Streaming Upload Buffer**
- Ring buffer for per-frame mesh uploads without GPU stalls
- Double-buffered (write B while GPU reads A)
- New file: `Buffers/StreamingUploadBuffer.cs`
- Tests: Wraparound, double-buffer alternation, capacity exhaustion

### Sprint 3: Rendering Foundation

**3A. Reversed-Z Depth**
- depth32float, depthCompare: greater, clearValue: 0.0, infinite far plane
- Projection matrix builder: only needs zNear (far = infinity)
- New file: `Rendering/ReversedZHelper.cs`
- Tests: Near objects depth ~1.0, far objects ~0.0, no Z-fighting at 10000 units

**3B. Vertex Pulling Pipeline**
- WGSL vertex shader reads PackedQuad from storage buffer
- quadIdx = vertexID/6, cornerIdx = vertexID%6, unpack position+dims+face+type
- Fragment shader: texture array lookup via blockType*6+face -> layer index
- New file: `Rendering/VertexPullPipeline.cs`
- Tests: Compute kernel mimicking vertex shader, verify unpacked positions

**3C. Draw Ordering via RadixSort**
- ILGPU RadixSort sorts chunk distance keys, produces sorted draw order
- Maximizes early-Z rejection (front-to-back)
- New file: `Rendering/DrawOrderKernels.cs`
- Tests: Verify nearest-first ordering, draw command reorder correctness

### Sprint 4: Adaptive Quality (Quest 3S viability)

**6A. Device Capabilities**
- Query MaxBufferSize, MaxStorageBufferBindings from accelerator
- Derive: vertex budget, chunk budget, draw command budget, pool bucket count
- Classify: Desktop / MobileHigh / MobileLow tiers
- New file: `Adaptive/DeviceCapabilities.cs`
- Tests: Budget math, tier classification for known profiles

**6B. Quality Controller**
- Frame time EMA with asymmetric rates (fast decrease 4x, slow increase 1x)
- 5 quality levels: Ultra -> High -> Medium -> Low -> Minimal
- Thermal throttling detection (frame time spike without scene change)
- Draw distance feedback loop (shrink under pressure, grow when stable)
- New file: `Adaptive/QualityController.cs`
- Tests: Over-budget -> decrease, under-budget -> slow increase, thermal detection, emergency drop

**6C. Streaming Budget**
- Max vertices/uploads/bytes per frame (device-adaptive)
- Priority queue: distance + view direction + movement prediction + LOD urgency
- New file: `Adaptive/StreamingBudget.cs`
- Tests: Budget enforcement, priority ordering, frame reset

### Sprint 5: Advanced Culling

**4A. Graph-Based Visibility (Sodium-style)**
- 48-bit per-section visibility encoding via flood fill
- BFS from camera section, outward-only, incoming direction accumulation
- Bitwise getConnections (no branches)
- New file: `Culling/VisibilityGraph.cs`
- Tests: Open air (all connected), solid (none), tunnel (axis only), cave BFS

**4B. Hi-Z Occlusion**
- Mip chain from depth buffer (max of 2x2 per level)
- AABB projection + mip level selection in compute shader
- Two-phase rendering with visibility buffer for frame coherence
- New files: `Culling/HiZOcclusionKernels.cs`, `Culling/VisibilityBuffer.cs`
- Tests: Occluder blocks large chunk, no occluder passes all, mip chain values

**4C. Culling Pipeline Orchestrator**
- Integrates: graph BFS (CPU) -> fog cull (GPU) -> frustum cull (GPU) -> Hi-Z (GPU) -> face masking
- Returns CullResult with visible chunk indices and face masks
- New file: `Culling/CullingPipeline.cs`

### Sprint 6: LOD System

**5A. LOD Reduction Kernel**
- DelegateSpecialization: one mesh kernel, LOD level as parameter (compile-time inlined)
- Super-block type = most common non-air in NxNxN group
- LOD 0: full, LOD 1: 2x (4x fewer verts), LOD 2: 4x (16x fewer), LOD 3: 8x (64x fewer)
- New file: `LOD/LODReducer.cs`
- Tests: Uniform chunk preserves type, mixed chunk picks most common, dimensions correct

**5B. LOD Selection**
- SelectLOD(distance, pressure) with configurable thresholds
- Pressure = totalVertices / vertexBudget shifts transitions inward
- Hysteresis to prevent oscillation
- New file: `LOD/LODSelector.cs`
- Tests: Close=LOD0, far=LOD3, pressure shifts, no oscillation

**5C. LOD Boundary Skirts**
- Thin polygons extruded downward at LOD boundaries to hide seams
- New file: `LOD/LODSkirts.cs`
- Tests: Skirt quads exist at boundaries, height correct

**5D. Geomorphing**
- Blend between LOD levels via per-chunk morph factor in vertex shader
- morphFactor = smooth transition based on distance within LOD transition zone
- New file: `LOD/Geomorphing.cs`

### Sprint 7: Visual Completeness

**8A. Transparency** - Dual occupancy masks (opaque + transparent), separate mesh pass
- New file: `Meshing/TransparencyMesher.cs`

**8B. Plants/Cross-Quads** - X-shaped quads, cannot greedy merge
- New file: `Meshing/CrossQuadMesher.cs`

**8C. Per-Vertex AO** - 3 corner neighbors -> AO level 0-3, packed in PackedQuad reserved bits
- New file: `Meshing/AmbientOcclusion.cs`

**8D. T-Junction Fix** - Sub-pixel quad expansion in vertex shader (0.001953125 world units)
- Integrated into VertexPullPipeline.cs

**9A. Texture Array Manager** - texture_2d_array, per-(blockType,face) layer mapping
- New file: `Rendering/TextureArrayManager.cs`

**9B. UV Generation** - fract(worldPos) tiling, textureIndexLUT uniform buffer
- Integrated into vertex pulling shader

### Sprint 8: Chunk Management + Integration API

**7A. Chunk State Machine** - UNLOADED -> LOADING -> MESHING -> GPU_READY -> VISIBLE -> CACHED -> EVICTING
- New file: `ChunkManager.cs`

**7B. Cache Integration** - IChunkCacheProvider interface for OPFS/other storage
- New file: `Caching/ChunkCacheProvider.cs`

**11A. Service Registration** - AddVoxelEngine() DI extension matching SpawnDev.BlazorJS pattern
- New file: `VoxelEngineServiceExtensions.cs`

**11B. Configuration** - VoxelEngineConfig with chunk size, height, LOD, quality, transparency classifiers
- New file: `VoxelEngineConfig.cs`

**11C. Events** - OnChunkLoaded/Evicted/Visible, OnQualityChanged, OnThermalThrottle

### Sprint 9: VR/AR

**10A. Stereo Rendering** - Combined frustum (union of both eyes), shared mesh, two VP matrices
- New file: `VR/StereoRenderer.cs`

**10B. WebXR Helpers** - Session setup via SpawnDev.BlazorJS
- New file: `VR/WebXRHelper.cs`

**10C. Quest 3S Thermal Management** - Frame time spike detection, quality reduction, recovery
- New file: `Adaptive/ThermalManager.cs`

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| i32 atomics for face mask clearing | i64 atomics not supported on WebGPU (emulated, non-atomic fallback) |
| Greedy merge on GPU, not CPU | ILGPU guaranteed faster than WASM CPU on Quest 3S |
| Single indirect draw buffer | 300x Chrome D3D12 validation improvement (measured by Toji) |
| Reversed-Z with depth32float | Zero Z-fighting at any distance, infinite far plane |
| DelegateSpecialization for LOD | One kernel, multiple LOD levels, zero overhead (compile-time inlined) |
| Cylindrical fog (not spherical) | Prevents terrain disappearing below when flying high (Sodium's approach) |
| Texture arrays (not atlas) | No mipmap bleeding, no per-tile padding, hardware tiling via fract() |

---

## Testing Requirements Per Phase

Every phase MUST have:
1. GPU kernel tests that run on ALL 6 backends via PlaywrightMultiTest
2. CPU reference implementation producing identical results
3. For meshing: coverage verification (expanded quads == original face masks)
4. For culling: CPU and GPU must agree on exact visible set (sorted comparison)
5. For buffers: data integrity after every operation
6. GpuTestVerify for large buffers (no CPU readback of 4M+ elements)
7. `dotnet test PlaywrightMultiTest` must pass with 0 failures before any commit

---

## Files Summary

**New files to create:** ~30 source files + ~15 test files
**Existing files to modify:** GreedyMergeKernels.cs, FrustumCullKernels.cs, PackedQuad.cs
**Empty directories to populate:** LOD/, Rendering/, Adaptive/

---

## NuGet Versioning

```
0.1.0: Meshing + VertexPool + FrustumCull (Sprints 1-2)
0.2.0: + Rendering Foundation + Adaptive (Sprints 3-4)
0.3.0: + Culling Pipeline + LOD (Sprints 5-6)
0.4.0: + Visual Completeness - transparency, AO, textures (Sprint 7)
0.5.0: + Chunk Management + Integration API (Sprint 8)
0.6.0: + HD Rendering - PBR, lighting, damage, night (Sprint 9)
0.7.0: + Terrain Gen, Destruction, Structural Integrity (Sprint 10)
1.0.0: + VR/AR + Thermal Management (Sprint 11) = FULL FEATURE SET
```

Project references during all 0.x development. NuGet publish at 1.0.0.

---

## Consumer Migration Order (Tuvok's proposal, approved)

```
Step 1: VertexPool replaces both projects' free-list buffers (smallest change, biggest stability win)
Step 2: FrustumCull replaces CPU culling (drop-in, same planes)
Step 3: GreedyMesh replaces per-face mesh kernels (biggest perf win, most testing)
Step 4: IndirectDraw + DrawOrder replaces per-chunk draw calls (architecture change)
Step 5+: LOD, QualityController, VisibilityGraph (incremental, no breaking changes)
```

Each step: both consumers update ProjectReference, replace one component, tests pass before next.

---

## Verification

After full implementation:
```bash
# All tests pass on all 6 backends
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj
# Expected: 200+ tests, 0 failures

# Browser demo loads and renders test scene
dotnet run --project SpawnDev.VoxelEngine.Demo
# Navigate to /tests, run all, verify 100% green

# Desktop demo runs on CUDA/OpenCL/CPU
dotnet run --project SpawnDev.VoxelEngine.DemoConsole
# List all tests, run each, verify all pass
```

---

## Cross-Project Integration Testing (Tuvok's request)

Test fixtures that load sample data from BOTH consuming projects:
- AubsCraft: sample 16x16x384 chunk from Minecraft server (24 sections)
- Lost Spawns: sample 16x16x256 Deer Isle terrain chunk (16 sections)
- Both meshed through same pipeline, verified against CPU reference
- Proves the library works with real production data, not just test generators
