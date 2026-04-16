# No Man's Sky Technical Deep Dive

**Date:** 2026-04-16
**Purpose:** Deep technical research for implementing NMS-style smooth terrain in SpawnDev.VoxelEngine
**Researcher:** Tuvok (Claude CLI #3)

---

## 1. Terrain Generation Pipeline

### Seed to Surface

NMS uses a single 64-bit seed to generate the entire universe deterministically. Generation cascades hierarchically:

1. **Seed** drives PRNG that plots star positions and classifications
2. **Star coordinates** become seeds for planetary systems
3. **Planet position** seeds terrain, climate, atmosphere, biome
4. **Biome seed** drives flora, fauna, resource distribution

Nothing stored on servers - planets regenerated identically from positional seed. Total world-building data: ~300 MB (Sean Murray, GDC 2017).

### Cube-Sphere Projection

- Each planet built as a cube with 6 faces
- Mathematical normalization projects cube surface to sphere: `normalize(P)`
- Each cube face is an independent quadtree for LOD
- Voxel data stored as flat arrays on cube faces
- Distortion from cube mapping is opposite of perspective projection distortion - GPU-friendly

### SDF Representation

Terrain stored as Signed Distance Field - scalar field where each point stores distance to nearest surface, with sign indicating solid (positive) or air (negative).

SDF enables:
- **Smooth surfaces** - surface at zero-crossing, not block boundaries
- **Caves and overhangs** - impossible with 2D heightmaps, natural with 3D SDF
- **Boolean composition** - multiple SDF layers combine via min/max
- **Terrain deformation** - modify SDF values locally, re-mesh affected region

### The "Uber Noise" System

Custom noise system built on **domain warping**: regular fBm noise where each sample point is offset by another fBm noise map before evaluation. Creates terrain that looks warped and twisted.

**7 noise layers** (from decompiled biome MBINs):
1. **Base** - fundamental ground plane + height variation
2. **Hill** - hill-scale features
3. **Mountain** - sharp peaked features
4. **Rock** - rock-scale detail
5. **UnderWater** - underwater terrain shape
6. **Texture** - fine surface variation
7. **Elevation** - large-scale elevation changes

Plus 9 grid layers with turbulence noise sub-layers.

**Parameters per layer:** Active, Octaves, Amplitude, Persistence (~0.5), Frequency/Lacunarity (~2x), HeightOffsets, Noise Type

### Cave and Cliff Generation (CSG Boolean Operations)

```
Terrain = Union of positive features MINUS negative features

Union:        min(a, b)            // merge two solids
Subtraction:  max(-a, b)           // carve a from b
Intersection: max(a, b)            // keep only overlap
Smooth Union: min(a,b) - h*h/(4k)  // smooth blending (Inigo Quilez)
```

Caves = 3D volumes of negative space carved from positive terrain via noise-driven SDF subtraction.

### LOD: Space to Surface

Quadtree hierarchy on cube faces:
1. From space: low-poly sphere with procedural texture
2. Approaching: quadtree subdivides visible faces
3. In atmosphere: higher LOD loads
4. Near surface: maximum LOD, full voxel detail
5. On surface: only nearby chunks at full resolution

Transition disguised through atmosphere entry effects, clouds, progressive loading. NOT truly seamless - cleverly disguised loading.

---

## 2. Meshing: Marching Cubes to Dual Marching Cubes

### Original Marching Cubes

- 256 cases (8 corners, inside/outside)
- Lookup table maps each case to triangulation (up to 5 triangles/cell)
- Edge interpolation: `t = (isovalue - v0) / (v1 - v0)`
- Problems: high vertex count, poor triangle quality, LOD seams

### Dual Marching Cubes (2024 Update)

DMC inverts vertex/face relationship:

**Standard MC:** Vertices on edges, faces are triangles
**Dual MC:** Vertices inside cells (feature points), faces are quads

Algorithm:
1. Cell classification (same 8-bit case as MC)
2. Dual point generation - one vertex per cell at feature point
3. Quad generation - connect dual vertices of adjacent cells
4. Lookup tables: `dualPointsList[256][4]`

**Vertex count comparison:**
| Method | Triangles |
|--------|-----------|
| Standard MC | 67,000 |
| Dual Contouring | 17,000 |
| **Dual MC** | **440** |

**150x reduction** in favorable cases.

**LOD seam advantage:** DMC is inherently crack-free at LOD transitions. Vertices inside cells (not on edges) means no geometric disagreement at resolution boundaries. No Transvoxel patching needed.

### Reference Implementation

`dualmc` C++ library (dominikwodniok/dualmc on GitHub):
- `DualPointKey`: linearized cell ID + point code for vertex deduplication via hash map
- Output: indexed quad mesh (4-index Quad structs)
- Optional manifold mode (Rephael Wenger's algorithm)

---

## 3. Rendering Techniques

### Engine

- Vulkan (since 2019, was OpenGL)
- Deferred rendering
- 60fps+ target (PS5 120Hz VRR)

### Worlds Part I/II Overhaul (2024-2025)

**Lighting:**
- Complete overhaul with increased precision
- GTAO (Ground Truth AO) rewritten from scratch
- Microshadow detail via AO enhancements
- Screenspace shadowing techniques
- Interior lighting precisely localized

**Water:**
- Mesh-based wave system (replaced plane-based)
- Subsurface scattering (sun through wave crests)
- Correct reflections (clouds, planets in water)
- Underwater crepuscular rays
- Dynamic waves from objects and rain

**Clouds:**
- Volumetric, completely rewritten
- Multiple types (cirrus, nimbus)
- Coverage varies with time and weather

**Atmosphere:**
- Unified wind simulation (trees, leaves, smoke, rain, fog, snow)
- Thick snowfall, ash, heavy rain, dust whorls
- Height-based fog calculation

### Procedural Textures

- **Triplanar mapping**: 3 textures projected by surface normal, blended. No UV stretching on cliffs.
- **Procedural color palettes**: per planet, influenced by biome/atmosphere/star type
- **Tag-based material selection**: assigned by slope, altitude, biome tags

---

## 4. Terrain Deformation / Mining

### Terrain Manipulator

Four modes: Mine, Create, Restore, Flatten

- **Mine**: Remove terrain in sphere (3 sizes). Resource yield time-based, not volume-based.
- **Create**: Add terrain in sphere/cube. Select shape, type, size.
- **Restore**: Return to original procedural state.
- **Flatten**: Level terrain to a plane.

### Storage

Edits stored in player's local save file:
- 256 buffer maximum (spatial groups)
- 15,000 total edit limit
- 10,000 base-related edit sub-limit
- Mining creates many edits per operation
- Oldest unprotected buffers overwritten when limits exceeded

### Multiplayer Limitations

- Planet generated from seed first, then edits applied as overlay
- **No real-time sync** of terrain edits between players
- Each player sees different terrain (different save files)
- Fundamentally client-side system

### Re-meshing Performance

Only affected chunk's SDF values change. With DMC: fewer vertices, less memory, faster than original MC.

---

## 5. Performance

### Threading

Dual-priority system:
- **High priority**: character models, ships, physics (Havok)
- **Low priority**: terrain generation, textures, AI, loading
- 4 active compute threads max

### Memory

- Uniform chunks (all solid or all air) = near-zero memory
- SDF: 8-bit or 16-bit fixed-point (not 32-bit float)
- Distant values clamped
- Total game: ~6 GB on disc, 300 MB for world gen data

### Streaming

- Dithering during planet approach = LOD streaming
- Worlds Part II: 4x faster load times, 10%+ smaller game size
- Terrain gen on low-priority threads

---

## 6. Implementation Plan for SpawnDev.ILGPU

### ILGPU Kernel Architecture

```csharp
// Phase 1: SDF Evaluation (embarrassingly parallel)
// One thread per voxel in chunk (32x32x32 = 32,768 threads)
Kernel_EvaluateSDF(Index3D index, ArrayView<float> sdfOutput, 
    int seed, Vector3 chunkWorldPos, NoiseParams noiseParams)

// Phase 2: Mark Active Cells
// One thread per cell, writes 1/0 to mask buffer
Kernel_MarkActiveCells(Index3D index, ArrayView<float> sdfInput, 
    ArrayView<int> maskOutput, ArrayView<int> vertexCounts)

// Phase 3: Prefix Sum for output compaction
// Standard parallel scan (ILGPU Algorithms already has this)
Kernel_PrefixSum(ArrayView<int> offsets)

// Phase 4: Generate Mesh (one thread per active cell)
Kernel_GenerateVertices(ArrayView<int> activeCellIds, 
    ArrayView<float> sdfInput, ArrayView<float> vertexOutput, 
    ArrayView<int> offsetBuffer, ArrayView<int> caseTable)
```

**All four kernels chain without CPU readback.** Data stays on GPU from SDF evaluation through to final vertex buffer.

### Memory Budget for Browser

| Resource | Console (PS5) | Browser (WebGPU) |
|----------|--------------|-------------------|
| VRAM | 16 GB shared | 256-512 MB typical |
| RAM | 16 GB | 2-4 GB WASM heap |
| Storage | SSD streaming | OPFS (118 MB/s) |
| Threads | 8+ cores | SharedArrayBuffer workers |

**Browser terrain budget: ~256 MB total**
- 50 MB geometry buffers
- 50 MB SDF volumes
- 100 MB textures
- 50 MB vegetation/structures

### SDF Storage Efficiency

- 32x32x32 chunk at 16-bit = 64 KB
- At 8-bit = 32 KB
- Uniform chunks (all solid/air) = ~0 bytes
- Average compressed: 2-8 KB per chunk
- Complex volumetric: 20-100 KB

### Hybrid Architecture: Blocks + Smooth Terrain

1. **Keep existing block infrastructure** for game logic, inventory, base building
2. **Add SDF layer** alongside block type per voxel
3. **Dual meshing paths**: greedy mesh for placed blocks/structures, DMC for natural terrain
4. **Noise-based terrain gen**: replace heightmap with 3D SDF via layered domain-warped noise
5. **Triplanar texturing** for terrain, standard UV for blocks
6. **GPU compute pipeline**: all meshing via ILGPU dispatches

### Performance Target

WebGPU reference (Cinevva browser terrain):
- 3.5-6.5ms total terrain system per frame
- Budget: heightmap 0.5-1ms, volumetric mesh 0.5-2ms, materials 1-1.5ms
- At 60fps (16.6ms), terrain uses ~21% of frame time

WebGPU Marching Cubes reference (Will Usher):
- 256^3 grid on RTX 3080 = ~32ms
- Within 6% of native Vulkan performance

---

## Key References

### GDC Talks
- **GDC 2017** - Innes McKendrick: "Continuous World Generation in No Man's Sky" (definitive pipeline talk)
- **GDC 2017** - Sean Murray: "Building Worlds Using Math(s)" (mathematical foundations)
- **GDC 2018** - Innes McKendrick: "Beyond Procedural Horizons"

### Papers
- Gregory M. Nielson (2004): "Dual Marching Cubes" (Rice University)
- Laine & Karras (2010): "Efficient Sparse Voxel Octrees" (NVIDIA)
- Eric Lengyel (2010): "Transvoxel Algorithm"
- Inigo Quilez: SDF distance functions (iquilezles.org)

### Implementations
- `dualmc` C++ library: github.com/dominikwodniok/dualmc
- Will Usher WebGPU Marching Cubes: willusher.io/graphics/2024/04/22/webgpu-marching-cubes
- WebGPU SDF Editor: reindernijhoff.net/2026/01/webgpu-sdf-editor
- Cinevva browser terrain: app.cinevva.com/guides/landscape-generation-browser

### Updates
- Worlds Part I (July 2024): DMC, volumetric clouds, water overhaul
- Worlds Part II (2025): GTAO rewrite, 120Hz VRR, 4x faster loading

---

*"I don't believe in the no-win scenario." - James T. Kirk*
*Neither does SpawnDev.ILGPU.*
