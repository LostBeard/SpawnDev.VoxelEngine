# Voxel Engine Master Feature Comparison

**Date:** 2026-04-16
**Purpose:** Comprehensive feature comparison between SpawnDev.VoxelEngine and 25+ industry engines
**Goal:** Identify gaps, opportunities, performance ideas, and features to prioritize

---

## Engines Surveyed

| Engine | Type | Rendering | Notable For |
|--------|------|-----------|-------------|
| SpawnDev.VoxelEngine | Our engine | WebGPU rasterization | 6 backends, browser-native, ILGPU compute |
| Minecraft (Java/Bedrock) | Commercial | OpenGL / RenderDragon | Palette compression, Caves&Cliffs worldgen |
| Teardown | Commercial | Voxel ray tracing (no HW RT) | Full ray tracing in fragment shaders, 262MB shadow volume |
| No Man's Sky | Commercial | Forward rendering | Planet-scale, Dual Marching Cubes, 18 quintillion planets |
| Veloren | Open source (Rust) | wgpu (Vulkan/Metal/DX12) | Greedy meshing, LISPSM shadows, SIMD |
| Vintage Story | Commercial (C#) | OpenGL (OpenTK) | 32-range colored lights, seasons, climate sim |
| 7 Days to Die | Commercial | Unity | Structural integrity physics |
| Deep Rock Galactic | Commercial | UE4 | Real-time terrain destruction via marching cubes |
| Astroneer | Commercial | UE4 | Terrain deformation as core mechanic |
| Luanti (Minetest) | Open source | Irrlicht/OpenGL | Extreme moddability, server-side occlusion culling |
| Terasology | Open source | LWJGL/OpenGL | Cubic chunks, research-oriented, DAG render pipeline |
| ClassiCube | Open source (C) | OpenGL 1.1 / D3D9 | Runs on anything, minimal overhead |
| Craft (Fogleman) | Open source (C) | Modern OpenGL | Clean educational codebase |
| VoxelPlugin | Commercial (UE) | UE rendering | Smooth + cubic modes, Transvoxel LOD seams |
| Cube World | Commercial | Vulkan (Omega rewrite) | Procedural character generation |
| Voxel Farm | Commercial | Any (outputs polygons) | Clipmap LOD, Dual Contouring, distributed render farm |
| Atomontage | Commercial | Volumetric microvoxels | No meshing, millimeter-scale, 20+ years R&D |
| MagicaVoxel | Tool | Path tracing | .vox format standard, GPU-accelerated editing |
| Blockbench | Tool | WebGL | Voxel modeling, glTF export, plugin ecosystem |
| John Lin | Research | Vulkan ray tracing | Multi-format voxel architecture, format-agnostic pipeline |
| Gabe Rundlett (Gvox) | Research | Vulkan HW RT | No meshing, intersection shaders, GPU voxel compression |
| Douglas Dwyer (Octo) | Research | WebGPU | Path-traced lighting IN BROWSER, WASM mods |
| Voxlap (Silverman) | Historic | CPU ray casting | Pioneer (2000), RLE surface-only storage |
| Aokana | Academic | GPU-driven hybrid | Shallow SVDAGs, 10B voxels at 6ms, Hi-Z culling |

---

## CATEGORY: MESHING

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Binary greedy meshing (GPU) | **DONE** | Us, cgerikj reference | 50-200us/chunk on CPU, we do it on GPU via ILGPU |
| Face culling (bitwise, 64 simultaneous) | **DONE** | Us | FaceCullKernels.cs |
| PackedQuad (8 bytes/quad) | **DONE** | Us, cgerikj | Compact vertex format |
| Transparency meshing (separate pass) | **DONE** | Us | TransparencyMesher.cs |
| Cross-quad meshing (X-shaped plants) | **DONE** | Us, Minecraft | CrossQuadMesher.cs |
| Per-vertex ambient occlusion | **DONE** | Us, Exile, Craft | AmbientOcclusion.cs |
| Marching Cubes | NOT PLANNED | NMS, DRG, Astroneer, VoxelPlugin | For smooth terrain - not our aesthetic |
| Dual Contouring | NOT PLANNED | Voxel Farm | Sharp feature preservation for smooth surfaces |
| Transvoxel (LOD seam stitching) | NOT PLANNED | VoxelPlugin, Lengyel | Relevant if we add smooth LOD transitions |
| Surface Nets | NOT PLANNED | Academic | Simpler than DC, no sharp features |
| No meshing (direct ray trace) | NOT PLANNED | Teardown, Gvox, Octo, Atomontage | Major architecture change, but interesting long-term |

**Gaps/Opportunities:**
- We're at the cutting edge for blocky meshing. Binary greedy on GPU is state of the art.
- Dual Contouring could be interesting for Lost Spawns' "HD realistic" style if we ever want smooth terrain alongside blocks
- Teardown's no-meshing approach (ray march volume textures) is the future trend but requires fundamentally different pipeline

---

## CATEGORY: LOD

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| LOD selector (distance + pressure) | **DONE** | Us | LODSelector.cs |
| LOD reducer (super-block via DelegateSpecialization) | **DONE** | Us | LODReducer.cs |
| Geomorphing (smooth LOD transitions) | **PLANNED** | VoxelPlugin, Voxel Farm | Prevents popping |
| LOD skirts (seam hiding) | **PLANNED** | Us | Fallback for geomorphing |
| Clipmap LOD (Voxel Farm) | NOT IN PLAN | Voxel Farm | Concentric rings, error-metric refinement |
| Octree LOD (NMS) | NOT IN PLAN | NMS | Planet-scale, single 64x64x64 sample per LODn |
| Shallow multi-DAG (Aokana) | NOT IN PLAN | Aokana (academic) | 6ms at 10B voxels, 9x memory reduction |
| MIP-level ray skipping (Teardown) | NOT IN PLAN | Teardown | Implicit LOD during ray marching |
| Distant Horizons style (4096 chunks) | NOT IN PLAN | MC mod | Extreme render distance via simplified distant chunks |

**Gaps/Opportunities:**
- Clipmap LOD (Voxel Farm style) could dramatically increase draw distance for Lost Spawns
- Aokana's shallow multi-DAG approach achieves 10B voxels at 6ms - worth studying for WebGPU
- "Distant Horizons" approach (simplified mesh for far chunks) is low-hanging fruit for visible draw distance

---

## CATEGORY: CULLING

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Frustum culling (GPU compute) | **DONE** | Us, everyone | FrustumCullKernels.cs |
| Graph-based visibility (Sodium-style BFS) | **DONE** | Us, Sodium MC mod | VisibilityGraph.cs |
| Culling pipeline orchestrator | **DONE** | Us | CullingPipeline.cs |
| Hi-Z occlusion culling | **PLANNED** | Aokana, Voxel Farm | GPU depth pyramid, 2-4x speedup for forests/buildings |
| Server-side occlusion culling | NOT IN PLAN | Luanti | Reduces network 50-80%, relevant for multiplayer |
| Voxel-based occlusion (Voxel Farm) | NOT IN PLAN | Voxel Farm | 64x64 software rasterizer, inscribed occluder quads |
| Custom early-Z checkpoints (Teardown) | NOT IN PLAN | Teardown | 8+ checkpoint system for ray marching |
| 64-bit visibility buffer (Aokana) | NOT IN PLAN | Aokana | 24-bit depth + chunk ID + voxel coords per pixel |

**Gaps/Opportunities:**
- Hi-Z is already planned and is the biggest single culling upgrade (2-4x for forests)
- Server-side occlusion culling from Luanti is relevant when multiplayer lands
- Visibility buffer rendering (Aokana) is a modern GPU-driven approach worth researching

---

## CATEGORY: LIGHTING

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Dynamic lighting (point/directional) | **DONE** | Us | DynamicLighting.cs |
| Time of day (sun direction, ambient) | **DONE** | Us | TimeOfDay.cs |
| Per-vertex AO | **DONE** | Us, Exile, Craft | Corner-based, 4 values per vertex |
| Flood fill light propagation | NOT IN PLAN | Minecraft, Luanti, Vintage Story | BFS 16 levels, dual bank (sky/block) |
| 32-range colored light mixing | NOT IN PLAN | Vintage Story | 16,384 color/level combos, correct mixing |
| Stochastic ray-traced AO (Teardown) | NOT IN PLAN | Teardown | 2 rays/pixel, spatiotemporal denoise |
| Path-traced indirect lighting | NOT IN PLAN | Octo (WebGPU!), MagicaVoxel | Octo does this IN BROWSER |
| Global illumination | NOT IN PLAN | Teardown, MC RTX | Per-voxel light bounce |
| Voxel cone tracing | NOT IN PLAN | Teardown | Environment shadow approximation |

**Gaps/Opportunities:**
- **Flood fill lighting is a MAJOR gap.** Minecraft, Luanti, and Vintage Story all do it. Without it, buildings have no interior darkness, torches don't illuminate, and the world feels flat. This should be high priority for DayZ atmosphere.
- Vintage Story's colored light system (32-range, 16K combos) is the gold standard for colored block lights
- Octo proves path-traced lighting works in WebGPU/WASM - long-term this is the direction
- SSAO is planned (post-process) - good intermediate step before full flood fill

---

## CATEGORY: SHADOWS

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Cascaded shadow maps (2-3 cascades) | **PLANNED** | Standard technique | Phase 4 of Data's plan |
| PCF soft shadow filtering | **PLANNED** | Standard technique | Phase 4 |
| Volumetric shadow map (Teardown) | NOT IN PLAN | Teardown | 262MB for entire play area, CPU-side updates |
| LISPSM (Veloren) | NOT IN PLAN | Veloren | Light-Space Perspective SM, better shadow distribution |
| Pixel-aligned shadows (MC Bedrock) | NOT IN PLAN | MC RenderDragon | Deferred pipeline, crisp voxel shadows |
| Ray-traced shadows | NOT IN PLAN | Teardown, MC RTX, Gvox | No shadow maps needed |

**Gaps/Opportunities:**
- Cascaded shadow maps are the right first step (planned)
- LISPSM from Veloren is a cheap upgrade over standard CSM - better shadow distribution
- Pixel-aligned shadows (Bedrock RenderDragon) would look amazing with blocky voxels

---

## CATEGORY: WATER

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Water surface mesh (animated displacement) | **PLANNED** | Phase 6 | |
| Fresnel reflectance | **PLANNED** | Phase 6 | |
| SSR (screen-space reflections) | **PLANNED** | Phase 6 | |
| Refraction/distortion | **PLANNED** | Phase 6 | |
| Depth-based absorption (Beer-Lambert) | **PLANNED** | Phase 6 | |
| Underwater fog/caustics | **PLANNED** | Phase 6 | |
| Transparent two-pass rendering | DONE (AubsCraft) | AubsCraft | Opaque pass + alpha blend pass |
| Flowing water simulation | **PLANNED** (Phase 9) | Minecraft, Vintage Story | Cellular automaton, source blocks |
| Reactive splash masks (Teardown) | NOT IN PLAN | Teardown | Up to 64 animated splashes |
| Frozen water/ice (Cube World) | NOT IN PLAN | Cube World | Temperature-dependent water state |

**Gaps/Opportunities:**
- Water plan is comprehensive
- Frozen water/ice should be added - fits DayZ survival (temperature system)
- Reactive splash masks from Teardown are a nice visual touch

---

## CATEGORY: WEATHER & SKY

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Sky dome/atmospheric scattering | **PLANNED** | Phase 7 | |
| Cloud layer | **PLANNED** | Phase 7 | |
| Rain/snow particles (GPU-instanced) | **PLANNED** | Phase 7 | |
| Day/night cycle | **PLANNED** | Phase 7 (TimeOfDay exists) | |
| Weather state machine | **PLANNED** | Phase 7 | |
| Full season simulation | NOT IN PLAN | Vintage Story | 27-day cycles, snow accumulation, snowmelt by latitude |
| Localized weather patterns | NOT IN PLAN | Vintage Story | Different weather in different areas |
| Physically-based atmospheric scattering | NOT IN PLAN | WebGPU research | Compute shader LUT precomputation |
| Puddles via Perlin noise (Teardown) | NOT IN PLAN | Teardown | Roughness modification, shadow map prevents indoor puddles |

**Gaps/Opportunities:**
- **Vintage Story's season simulation** is exactly what Lost Spawns needs for DayZ survival
- Localized weather (rain here, clear there) adds huge immersion
- Puddles from Teardown are a cheap visual upgrade (modify roughness based on noise + exposure to sky)

---

## CATEGORY: PHYSICS & DESTRUCTION

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Voxel raycaster (DDA) | **DONE** | Us | VoxelRaycast.cs |
| AABB-voxel collision (sweep test) | **DONE** | Us | VoxelCollision.cs |
| Sphere-voxel intersection | **DONE** | Us | VoxelSphereQuery.cs |
| Structural integrity (flood fill) | **DONE** | Us, 7DtD | StructuralIntegrity.cs |
| Explosion kernels | **DONE** | Us | ExplosionKernels.cs |
| Terrain deformation tool | NOT IN PLAN | NMS, Astroneer | Core mechanic in both games |
| CPU voxel-vs-voxel collision (Teardown) | NOT IN PLAN | Teardown | SIMD + multithreaded, per-object volumes |
| Material-based structural properties | NOT IN PLAN | 7DtD | Wood < concrete < steel integrity |
| Cascading structural collapse | NOT IN PLAN | 7DtD, Teardown | Physics-driven destruction chains |
| Falling blocks (gravity) | NOT IN PLAN | Minecraft, Vintage Story | Sand, gravel fall when unsupported |

**Gaps/Opportunities:**
- **Terrain deformation tool** (NMS/Astroneer style) - TJ and Aubs both love this. The explosion kernels + structural integrity already provide the foundation. Adding a "terrain manipulator" tool that adds/removes blocks in a sphere is straightforward.
- **Material-based structural properties** from 7DtD - fits DayZ base building perfectly
- **Falling blocks** - simple gravity for unsupported sand/gravel, adds realism
- **Cascading collapse** - when structural integrity fails, blocks fall with physics. Dramatic and functional.

---

## CATEGORY: WORLD GENERATION

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Heightmap terrain provider | **DONE** | Us | HeightmapTerrainProvider.cs |
| Biome assigner (elevation+slope+moisture) | **DONE** | Us | BiomeAssigner.cs |
| Multi-octave noise | **PLANNED** | Phase 10 | |
| Cave generation (3D noise) | **PLANNED** | Phase 10 | |
| Structure templates | **PLANNED** | Phase 10 | |
| Road network generation | **PLANNED** | Phase 10 | |
| Real-world DEM import | **WIP** | Us (Lost Spawns) | Copernicus/USGS data |
| Minecraft Noise Router | NOT IN PLAN | Minecraft | Density functions for terrain + biomes + aquifers |
| 3D Aquifer system | NOT IN PLAN | Minecraft | Underground water systems with barriers |
| Multi-noise biome selection | NOT IN PLAN | Minecraft | 5-parameter biome placement |
| Climate-constrained spawning | NOT IN PLAN | Vintage Story | Animals spawn by temp/rain/forest/elevation |
| Rock strata simulation | NOT IN PLAN | Vintage Story | Geological layers underground |
| Soil fertility tracking | NOT IN PLAN | Vintage Story | Farming quality varies by location |
| Deterministic from seed | ASSUMED | Everyone | Same seed = same world |
| Distributed generation (farm) | NOT IN PLAN | Voxel Farm | TCP/IP worker nodes |

**Gaps/Opportunities:**
- **Minecraft's Noise Router** is the state of the art for procedural terrain - density functions are more flexible than simple heightmaps
- **3D Aquifers** (Minecraft) create underground water systems - great for cave exploration
- **Rock strata** (Vintage Story) adds mining depth - different ores at different geological layers
- **Climate-constrained spawning** ties directly to the survival mechanics already planned

---

## CATEGORY: DATA STRUCTURES & COMPRESSION

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| PackedBlock (12-bit type + 4-bit damage) | **DONE** | Us | ushort per block |
| Flat array per section (16x16x16) | **DONE** | Us | Standard approach |
| OPFS region files (118 MB/s write) | **DONE** (AubsCraft) | Us | 107x faster than IndexedDB |
| Palette compression | NOT IN PLAN | Minecraft | 4096 indices in 64-bit arrays, min 4 bits/index |
| Run-Length Encoding (surface only) | NOT IN PLAN | Voxlap | Only surface voxels stored |
| Sparse Voxel Octrees | NOT IN PLAN | Laine & Karras, Voxlap | Compact, GPU ray castable |
| Sparse Voxel DAGs | NOT IN PLAN | Academic | Orders of magnitude better than SVO |
| Symmetry-aware SVDAGs | NOT IN PLAN | Academic | Exploits mirror symmetries |
| Shallow multi-DAG (Aokana) | NOT IN PLAN | Aokana | Better cache locality than deep tree |
| LZ4 compression | **PLANNED** | Phase 8 | 80-90% reduction |
| Volume textures (Teardown) | NOT IN PLAN | Teardown | 8 voxels per texel, 2x2x2 packing |

**Gaps/Opportunities:**
- **Palette compression** (Minecraft) is the single biggest compression win for chunk storage. Most sections use <16 block types, so 4 bits per block instead of 16 = 4x compression for free.
- Our OPFS region file approach is already best-in-class for browser storage (benchmarked)
- SVDAGs are fascinating for the future but require different data access patterns

---

## CATEGORY: NETWORKING / MULTIPLAYER

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| P2P via WebRTC | **PLANNED** | Us (SpawnDev.RTC ready) | |
| Dedicated servers (ASP.NET) | **PLANNED** | Us | |
| Ed25519 identity | **PLANNED** | Us (Crypto lib ready) | |
| Chunk serialization + compression | **PLANNED** | Standard | |
| Server-side occlusion culling | NOT IN PLAN | Luanti | Reduces bandwidth 50-80% |
| ECS-based entity sync | NOT IN PLAN | Veloren | specs library |
| Voxel destruction synchronization | NOT IN PLAN | Teardown | Real-time multi-player destruction |
| MC protocol bridge | **PLANNED** (AubsCraft Phase Z) | Us | Direct Minecraft protocol |

**Gaps/Opportunities:**
- SpawnDev.RTC + SpawnDev.WebTorrent provide a unique P2P infrastructure nobody else has in browser
- Server-side occlusion culling (Luanti) should be added when multiplayer ships - 50-80% bandwidth reduction

---

## CATEGORY: VR / AR

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| Stereo renderer | **DONE** | Us | StereoRenderer.cs |
| WebXR helper | **DONE** | Us | WebXRHelper.cs |
| Foveated rendering | **DONE** | Us | FoveatedRendering.cs |
| Hand tracking (WebXR) | **DONE** (GameUI) | Us | XRHandProvider |
| Controller input | **DONE** (GameUI) | Us | XRControllerProvider |
| VR session lifecycle | **PLANNED** | Phase 11 | |
| Passthrough AR | **PLANNED** | Phase 11 | Quest 3S |
| VR comfort (snap turn, vignette) | **PLANNED** | Phase 11 | |
| Scale model / tabletop view | **PLANNED** | AubsCraft Phase Q | |

**Gaps/Opportunities:**
- We're ahead of most voxel engines on VR - most don't have WebXR support at all
- Quest 3S passthrough AR tabletop view is a unique differentiator

---

## CATEGORY: UI

| Technique | Our Status | Who Does It | Notes |
|-----------|-----------|-------------|-------|
| GPU-rendered UI (no HTML) | **DONE** | Us (GameUI) | 36+ elements, SDF fonts |
| DayZ-style radial menu | **DONE** | Us | UIRadialMenu |
| Inventory grid with drag-drop | **DONE** | Us | UIGrid |
| Equipment slots (paper doll) | **DONE** | Us | UIEquipmentPanel |
| Crafting panel | **DONE** | Us | UICraftingPanel |
| Status effects (buff/debuff) | **DONE** | Us | UIStatusEffects |
| Minimap with compass | **DONE** | Us | UIMapPanel + UICompass |
| Debug overlay (F3) | **DONE** | Us | UIDebugOverlay |
| SDF text (resolution-independent) | **DONE** | Us | SDFFontAtlas |
| Nine-slice rendering | **DONE** | Us | NineSliceBorder |
| Accessibility themes | **DONE** | Us | HighContrast, Colorblind |
| VR hand/controller input | **DONE** | Us | Full input abstraction |
| Tween animation system | **DONE** | Us | 10 easing functions |

**Gaps/Opportunities:**
- GameUI is comprehensive and unique - no other browser voxel engine has a full GPU-rendered UI system
- The VR input abstraction (mouse/controller/hand/gaze all unified) is ahead of the industry

---

## CATEGORY: UNIQUE TO US (NO OTHER ENGINE HAS)

| Feature | Status | Why It's Unique |
|---------|--------|----------------|
| 6 GPU backends from one codebase | **DONE** | WebGPU + WebGL + Wasm + CUDA + OpenCL + CPU |
| Runs in browser (Blazor WASM) | **DONE** | Full voxel engine in a browser tab |
| ILGPU compute kernel transpilation | **DONE** | C# kernels compiled to WGSL/GLSL/Wasm/PTX |
| P2P distributed GPU (AcceleratorType.P2P) | **DONE** | GPU compute across WebRTC peers |
| Zero-copy JS interop (HeapView) | **DONE** | 5 GB/s .NET-to-JS data sharing |
| AI crew development model | **DONE** | 4 coordinated AI agents, file-based protocol |
| PWA installable game | **PLANNED** | Install from browser, works offline |
| WebTorrent asset delivery | **PLANNED** | P2P model/texture distribution |
| Ed25519 cryptographic identity | **PLANNED** | Hardware key (YubiKey/Trezor) swarm ownership |
| Quest 3S AR tabletop editor | **PLANNED** | Edit world from above via passthrough |

---

## TOP PRIORITY GAPS (Recommended for Lost Spawns)

### Must Have (DayZ atmosphere depends on these)
1. **Flood fill lighting** - Without it, no interior darkness, no torch illumination. Every engine with good atmosphere has this.
2. **Texture pipeline** (Phase 3, already planned) - Solid colors won't cut it for DayZ feel.
3. **Cascaded shadow maps** (Phase 4, already planned) - Flat world without shadows.
4. **Palette compression** - 4x chunk storage reduction, enables larger worlds in browser memory.

### Should Have (Major visual/gameplay upgrade)
5. **Season simulation** (Vintage Story inspired) - Temperature, snow accumulation, seasonal biomes.
6. **Terrain deformation tool** - NMS/Astroneer style, TJ and Aubs love this. Foundation exists (ExplosionKernels + StructuralIntegrity).
7. **Material-based structural integrity** - 7DtD style, wood < concrete < steel for base building.
8. **Falling blocks** - Gravity for unsupported sand/gravel. Simple, adds realism.
9. **Colored block lights** (Vintage Story inspired) - Torches warm, redstone red, etc.

### Nice to Have (Differentiation)
10. **Clipmap LOD** (Voxel Farm style) - Extreme draw distance.
11. **Localized weather** - Rain here, clear there.
12. **Puddles** (Teardown trick) - Perlin noise on roughness + sky exposure check.
13. **3D Aquifers** (Minecraft) - Underground water systems.
14. **Rock strata** (Vintage Story) - Geological mining depth.

### Long-Term Research
15. **Voxel ray tracing** (Teardown/Octo direction) - Fragment shader ray marching of volume textures.
16. **Path-traced indirect lighting** (Octo proves this works in WebGPU).
17. **SVDAGs** - Orders of magnitude compression improvement.
18. **Dual Contouring** - If we ever want smooth terrain alongside blocks.

---

## PERFORMANCE TECHNIQUES WORTH STEALING

| Technique | From | Benefit | Effort |
|-----------|------|---------|--------|
| Palette compression for chunks | Minecraft | 4x storage reduction | Medium |
| Multi-draw indirect (single draw call) | cgerikj, Exile | Eliminates per-chunk draw overhead | Medium (planned) |
| Front-to-back draw sorting via DAIC | Nick's Blog | 2x rasterization speedup | Low (RadixSort exists) |
| MIP-level acceleration for ray ops | Teardown | 2x2x2 and 4x4x4 block skipping | Medium |
| Shallow multi-DAG (Aokana) | Academic | 6ms at 10B voxels, 9x memory reduction | High |
| Server-side occlusion culling | Luanti | 50-80% network bandwidth reduction | Medium |
| Instanced quad rendering | Exile | 1 instance per face, gl_VertexID selects corner | Low |
| Spatiotemporal denoising | Teardown | Enables stochastic lighting (fewer rays) | High |
| OPFS region files | Us (already done) | 107x faster than IndexedDB | Done |
| Compute shader atmospheric LUT | WebGPU research | Precompute sky, sample cheaply | Medium |

---

*Research compiled by Tuvok (Claude CLI #3) for Captain TJ. Sources: 25+ engines, 4 academic papers, 4 local SpawnDev projects.*
