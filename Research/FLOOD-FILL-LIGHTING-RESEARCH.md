# Flood Fill Lighting Research — Phase 14

**Author:** Tuvok (Claude CLI #3, Research/Planning)
**Date:** 2026-04-16
**Audience:** Data (VoxelEngine editor), Captain (TJ)
**Status:** Research complete. Architecture proposed. Awaiting review before Phase 14 kickoff.
**References in this repo:**
- `Plans/PLAN-v1.0.0-LostSpawns.md` — Phase 14 spec
- `Research/MASTER-FEATURE-COMPARISON.md` — flagged flood fill lighting as MAJOR gap
- `Research/voxel-engine-optimization-masterplan.md` — uses flood fill for section visibility (different feature, same primitive)
- `SpawnDev.VoxelEngine/Rendering/DynamicLighting.cs` — current 16-light forward renderer
- `SpawnDev.VoxelEngine/Meshing/AmbientOcclusion.cs` — current per-corner AO pattern (to reuse)
- `SpawnDev.VoxelEngine/Meshing/PackedQuad.cs` — 64-bit quad format with 17 reserved bits

---

## 1. Problem Statement

Current lighting consists of:
- **DynamicLighting** — up to 16 forward-rendered point/spot/directional lights, sampled per-fragment. Lights shine *through* walls because they don't know about voxel geometry.
- **AmbientOcclusion** — baked per-vertex, 2 bits per corner, purely geometric.
- **TimeOfDay** — global ambient + sun direction, no spatial variation.

Result: night is uniformly dark, torches glow in open air but do not illuminate caves or building interiors, sky light does not stop at roofs. This breaks the DayZ atmosphere Lost Spawns needs.

Flood fill lighting solves this by storing a per-voxel light level that propagates through transparent blocks and stops at solid ones. Every voxel engine with serious atmosphere has it (Minecraft, Luanti, Vintage Story). Teardown replaces it with full ray tracing, which is out of scope for v1.0.

---

## 2. Prior Art (Verified)

### 2.1 Minecraft Java Edition
- **Algorithm:** Serial CPU BFS. Separate increase/decrease queues.
- **Data:** 4 bits sky + 4 bits block, separate nibble arrays per 16³ section.
- **Attenuation:** light drops by `max(1, opacity)` per step; sky light preserves value 15 straight down.
- **Updates:** affected chunk + 8 neighbors invalidated; vanilla reads 24k–181k blocks per glowstone removal.
- **Meshing:** light baked into vertex attributes (smooth lighting, 4-voxel average at corners).
- **Source:** `minecraft.wiki/w/Light`, `greyminecraftcoder.blogspot.com` lighting breakdown.

### 2.2 Starlight (PaperMC)
- **Same algorithm, engineered harder.** Unified BFS queue, shape-check skip flags, direction-mask to avoid backtracking, opacity bitsets (1 bit per block) to skip transparent-section reads.
- **Measured 12–37× faster than vanilla; Mojang absorbed the designs into 1.20+.**
- **Source:** `github.com/PaperMC/Starlight/blob/fabric/TECHNICAL_DETAILS.md`.

### 2.3 Luanti (Minetest)
- **Priority-queue BFS.** `LightQueue` has 16 separate vectors (one per light level) for O(1) highest-first extraction.
- **Separate increase/unlight/relight queues.**
- **Monochrome** (day/night only, no colored lights).
- **Source:** `minetest/voxelalgorithms.cpp`.

### 2.4 Vintage Story
- **Colored lights: RGB channels, not palette lookup.** Tyron explicitly rejected HSV (mixing artifacts were unfixable) and switched to RGB.
- **Source:** Mighty Blocks R40 devlog (`vintagestory.at/blog.html/news/mighty-blocks-r40/`).
- **UNVERIFIED:** the "32 range levels / 16,384 combos" figure cited in MASTER-FEATURE-COMPARISON.md is not in any public source I could find. Closed source; only the API module is public. Worth confirming before citing.

### 2.5 Distant Horizons
- **Simplified per-LOD lighting**, not full BFS.
- Exposes `dhMaterialId` with `DH_BLOCK_ILLUMINATED` flag so shaders can distinguish emissive surfaces.
- Relevant for our LOD pipeline — do not try to run full propagation on distant LOD sections.

### 2.6 Teardown
- **Not flood fill.** Full ray-tracing through a 3-mip volume texture packing 2³ voxels per byte. Spatiotemporal denoiser. Small CPU texture uploads on destruction.
- **Takeaway:** path-traced voxel lighting on commodity GPUs is viable, but out of scope for v1.0. Future direction.

### 2.7 Hytale, Avorion, Veloren
- Hytale bakes painted shadows into textures + runtime shadowmap. Acknowledges "visible light propagation issues" and plans colorimetry later.
- Avorion "light block" is a sprite, not a propagating source.
- Veloren: traditional mesh-based per-voxel lighting, CPU-side. Physically-based attenuation (water uses real material properties). No BFS specifics published.

---

## 3. GPU-Parallel Flood Fill Primitives

**We are GPU-first. Serial BFS queues are the wrong tool.**

### 3.1 Iterative Stencil (recommended)
- Every voxel reads its 6 neighbors each pass, sets itself to `max(neighbor - opacity, selfEmission)`.
- Ping-pong buffers avoid read/write hazards.
- ~15 passes cover the full 0–15 range (N passes = max propagation distance N).
- Embarrassingly parallel. Maps directly to all ILGPU backends (CUDA, OpenCL, CPU, WebGPU compute, WebGL transform-feedback, Wasm SIMD).
- Matches the existing ExplosionKernels stencil pattern.

### 3.2 Jump Flooding Algorithm (JFA) — REJECT for lighting
- Rong & Tan 2006 (`comp.nus.edu.sg/~tants/jfa/`). Logarithmic jumps, O(log N) passes.
- **Approximates taxicab distance.** For signed distance fields this is fine; for discrete light levels, the approximation creates visible boundary artifacts unless post-cleaned with bucket-BFS.
- Net cost exceeds just iterating 15 times. Keep JFA in mind for SDF phase, not lighting.

### 3.3 SWAR Nibble Parallelism
- Lysenko (`0fps.net/2018/02/21/voxel-lighting/`): pack R/G/B/sky into one 16-bit word, compute `max` / `saturating-subtract` across all four nibbles in one ALU op.
- Our ILGPU kernels can do this on any backend that has 32-bit integer ops (all of them).
- **This is how we propagate 4 channels for ~free** when colored lights ship.

### 3.4 Light Propagation Volumes — REJECT
- Kaplanyan/Dachsbacher I3D 2010. Spherical harmonics probe grid.
- Designed for global illumination (indirect lighting), not direct block lighting. Wrong problem. Too heavy.

---

## 4. Current VoxelEngine State (Survey)

| Component | Status | Relevance |
|-----------|--------|-----------|
| `DynamicLighting.cs` | 16-light forward renderer, 64 bytes per light, inverse-square falloff | Keep as-is for player flashlights / dynamic sources. Flood fill will be a SEPARATE channel. |
| `AmbientOcclusion.cs` | Per-corner, 2 bits each, CPU during meshing | **Reuse the exact corner-sampling pattern for light baking.** Same 4-corner smooth interpolation. |
| `TimeOfDay.cs` | Global ambient + sun direction | Drives the 0–15 sky light level every pass (Time → skyLightValue). |
| `PackedQuad.cs` | 64-bit format, **17 reserved bits [47:63]** | Room for 16 bits of per-corner light. Exact fit. |
| `SectionCoord` | 16×16×16 sections, Y range configurable | Propagation boundary. Needs 1-voxel border reads from neighbor sections. |
| `BlockRegistry.cs` | Block type definitions | Add `Opacity` and `Emission` properties per block. |

---

## 5. Proposed Architecture

### 5.1 Storage

**Per-voxel light data:** 16 bits per voxel (one `ushort`), stored in a grid parallel to the existing `PackedBlock` grid.
- MVP (Phase 14.1): **4 bits sky + 4 bits block** = 8 bits used, 8 bits reserved.
- Full (Phase 14.2): **4 bits R + 4 bits G + 4 bits B + 4 bits sky** = 16 bits. Vintage Story-style colored lights.

Memory cost: 16³ voxels × 2 bytes = **8 KB per section**. For 4096 loaded sections, 32 MB total. Acceptable.

Uniform sections (all air or all solid, common case) get the same compression treatment as `PackedBlock` — one entry for the whole section.

### 5.2 Block Properties (add to BlockRegistry)

```csharp
public struct BlockLightProps
{
    public byte Opacity;       // 0 = transparent, 15 = full block
    public byte EmissionSky;   // 0–15 (usually 0; glowstone = 15)
    public byte EmissionR;     // 0–15 (torches: 14 warm red-orange)
    public byte EmissionG;     // 0–15
    public byte EmissionB;     // 0–15
}
```

Air: `Opacity=0, Emission=0`.
Glass: `Opacity=0, Emission=0` (sky light passes through).
Leaves: `Opacity=2, Emission=0` (MC Bedrock rule; let them dim light slightly).
Torch: `Opacity=0, EmissionR=14, EmissionG=9, EmissionB=3` (warm fire color).
Stone: `Opacity=15, Emission=0`.

### 5.3 Propagation Kernel (ILGPU)

```csharp
// Stencil pass — one kernel dispatch per pass, ~15 passes per dirty section batch.
static void PropagateLightKernel(
    Index3D idx,
    ArrayView3D<ushort> src,           // current light
    ArrayView3D<ushort> dst,           // next light
    ArrayView3D<byte> opacity,         // per-voxel opacity
    ArrayView3D<ushort> emission)      // per-voxel emission (static)
{
    int x = idx.X, y = idx.Y, z = idx.Z;

    // Read self + 6 neighbors (with section-boundary apron handled by kernel dispatcher)
    ushort self = src[idx];
    ushort best = emission[idx];            // floor = static emission

    best = NibbleMax(best, AttenuateAll(src[x - 1, y, z], opacity[x - 1, y, z]));
    best = NibbleMax(best, AttenuateAll(src[x + 1, y, z], opacity[x + 1, y, z]));
    best = NibbleMax(best, AttenuateAll(src[x, y - 1, z], opacity[x, y - 1, z]));
    best = NibbleMax(best, AttenuateAll(src[x, y + 1, z], opacity[x, y + 1, z]));
    best = NibbleMax(best, AttenuateAll(src[x, y, z - 1], opacity[x, y, z - 1]));
    best = NibbleMax(best, AttenuateAll(src[x, y, z + 1], opacity[x, y, z + 1]));

    dst[idx] = best;
}
```

`NibbleMax` and `AttenuateAll` are SWAR helpers that operate on all 4 nibbles at once using plain integer math — work on every ILGPU backend. Examples:

```csharp
// max of two u16 values, computed nibble-wise in a branchless fashion
static ushort NibbleMax(ushort a, ushort b) { /* SWAR */ }

// subtract opacity from each nibble, saturating at 0
static ushort AttenuateAll(ushort a, byte opacity) { /* SWAR */ }
```

**Dispatch pattern:**
- Group dirty sections into batches. Allocate `src`/`dst` buffers sized `(N + 2)³` per section to include a 1-voxel apron from neighbors (needed so edge voxels see the correct neighbor light).
- Run ~15 ping-pong passes per batch.
- Sky light has a special seed pass before iteration: every voxel with clear column above gets `skyNibble = 15`.

### 5.4 Update Strategy

**On block change at position P:**

1. Mark `SectionMask` bits for every section within Manhattan distance ≤ 15 of P (maxLight = 15). Usually just the containing section and its immediate neighbors.
2. At end of frame (or next tick), run the propagation kernel on the dirty sections as one batch.
3. Sections whose light changed trigger mesh re-bake (only re-bake light nibbles in PackedQuads, not full re-mesh if geometry unchanged — see 5.6).

**Cost:** one batch ≈ 15 stencil passes over dirty sections. On desktop GPUs a single passover 27 dirty sections (3×3×3 local cluster) = 27 × 4096 voxels × 15 passes ≈ 1.6M voxel updates ≈ sub-millisecond. On Quest 3S, budget 2–4 ms per batch and throttle dirty flushing to once every 2–3 frames.

### 5.5 Section Boundary Handling

Each dirty section's propagation kernel reads a 1-voxel apron from its 6 face neighbors. Two options:

- **(A) Per-dispatch copy:** kernel-dispatcher builds an `(18)³` padded src buffer by copying faces from neighbors before each batch. Simpler, wastes some memory bandwidth.
- **(B) Unified flat buffer:** all loaded sections live in one giant 3D `ArrayView<ushort>` indexed by `(globalX, globalY, globalZ)`. Kernel samples directly. Zero copy. More complex addressing but fastest.

**Recommendation: (B).** We already unify sections into shared buffers for mesh pool / indirect draws. Same pattern.

### 5.6 Meshing Integration

Two changes to greedy mesh:

1. **Compute per-corner light during quad emission.** Same pattern as `AmbientOcclusion.ComputeQuadAO` — sample 4 voxels meeting at each corner, average their light values. Reuses corner-sampling math exactly.

2. **Extend PackedQuad with 16 bits of light data.** Fills the reserved [47:62] range. Bit 63 stays reserved for a future flag.
   ```
   [47:50] corner0 light (4 bits: MAX of sky/block for MVP; full RGBA later)
   [51:54] corner1 light
   [55:58] corner2 light
   [59:62] corner3 light
   [63]    reserved
   ```

   For the MVP, the 4 bits stored per corner is `max(skyLight * skyFactor, blockLight)` where `skyFactor` comes from TimeOfDay (1.0 at noon, 0.0 at night). This gives correct "dark at night, bright at day" without needing separate sky/block channels in the quad itself.

   **Phase 14.2 upgrade:** when RGB lights ship, widen PackedQuad from 64 bits to 96 bits (12 bytes) to store 4 corners × (R+G+B+sky) × 4 bits = 64 bits of light. Breaking change — bundle with the colored-lights feature release.

3. **Light update without geometry change:** if only light changed (block placed at distance, torch placed), skip the full greedy-merge step and just re-compute the light nibbles in existing quads. Needs a fast per-quad "refresh light" kernel reading the current light grid and patching PackedQuad bits. This is the key to torch placement not causing a full re-mesh.

### 5.7 Fragment Shader Integration

Vertex shader unpacks per-corner 4-bit light → float in [0,1]. Rasterizer interpolates across the quad. Fragment shader:

```wgsl
let lightLevel = interpolatedLight;    // 0..1 from vertex
let timeAmbient = timeOfDay.ambient;   // existing TimeOfDay uniform
let aoMul = aoToMultiplier(interpolatedAO);

let baseColor = texSample * blockColor;
let dynamicLights = accumulateDynamicLights(pos, normal);  // existing DynamicLighting pass

// Flood fill contribution
let staticLight = lightLevel * (timeAmbient + sunContribution);

// Combine
let finalColor = baseColor * aoMul * (staticLight + dynamicLights);
```

- **Flood fill light is MULTIPLICATIVE with base color, ADDITIVE with dynamic lights.**
- Torch in a cave: flood fill lights a 10-voxel sphere → player's face stays warm-lit as they walk in, while the rest of the cave is dark.
- Player's flashlight: `DynamicLighting` point light adds a cone on top. Not propagated (can't flood-fill-update at 60 fps for a moving source without stuttering).

---

## 6. Testing Strategy

Data's plan calls for 3 tests; I propose expanding to cover the boundary cases:

1. **Unit: torch illuminates room** — place torch block in a 7×7×7 all-stone room (hollow center), verify every center voxel has blockLight > 0 after propagation.
2. **Unit: sky light through window** — stone column with 1-voxel opening, verify light penetrates vertically through the opening.
3. **Unit: opacity attenuates** — stack of leaves (opacity=2), verify light decrements by 2 per layer.
4. **Unit: section boundary propagation** — place torch 1 voxel from section edge, verify neighbor section's voxels receive light correctly (tests 1-voxel apron handling).
5. **Unit: block removal triggers decrease** — place then remove torch, verify all affected voxels return to 0.
6. **Unit: deterministic ordering** — run propagation on same input from different starting states, verify identical output (no race condition).
7. **Pixel readback: lit vs unlit quad** — render 2 blocks, one surrounded by stone (light=0), one in open air with sky light 15. First pixel darker than second.
8. **Pixel readback: gradient across quad** — large merged quad (6×6 face) with one corner in dark, one in light; verify smooth gradient across quad face.

All CPU reference paths via existing `GreedyMergeCpuReference` pattern. All GPU verification via `GpuTestVerify` (no big CPU readbacks).

---

## 7. Phased Rollout

### Phase 14.1 — Monochrome MVP (ship first)
- Storage: 8 bits per voxel (4 sky + 4 block).
- Propagation: stencil kernel, unified flat buffer, SWAR on 2 nibbles.
- PackedQuad: 4 bits per corner (max of sky/block, scaled by time).
- Block props: Opacity + EmissionSky + EmissionBlock.
- Tests 1–7 above.
- Target: v1.0.0 core acceptance criteria met.

### Phase 14.2 — Colored Lights (follow-up)
- Storage upgrade to 16 bits (R/G/B/sky).
- Propagation: same kernel, SWAR extends to 4 nibbles (identical op count — this is the beauty of the layout).
- PackedQuad: widen to 96 bits (12 bytes), 64 bits of light + existing 32 bits of block/AO/face data.
- Block props extend to EmissionR/G/B.
- New tests for color mixing correctness.

### Phase 14.3 — Dynamic Source Integration (optional)
- Moving torches (player-held) can be fed into flood fill at lower update rate.
- Consider a "secondary" low-res light grid for dynamic sources if static flood fill at 60 fps isn't achievable for moving sources.

---

## 8. Open Questions for Captain

1. **Light range = 15 or 31?** Minecraft standard is 15 (4 bits). A 5-bit range (0–31) gives longer light falloff (Vintage Story claim, unverified) but doubles kernel iteration count and requires unaligned bit-packing. **My recommendation: 15.** Proven, aligned, matches player expectations.

2. **Do we commit to colored lights for v1.0?** The layout is designed to support it cleanly, but 14.2 is an extra scope. **My recommendation: ship monochrome in v1.0, colored in v1.1.** Monochrome already closes the "interior darkness" gap that Lost Spawns needs most.

3. **Directional opacity (stairs, slabs, pistons)?** Minecraft lets light pass through the open side of a stair. This is non-trivial — each block gets 6-directional opacity, not scalar. **My recommendation: v1.0 skips directional opacity, every block is fully cubic for lighting purposes.** Revisit when we ship non-cubic block shapes.

4. **Storage strategy confirmation.** Parallel 3D array alongside PackedBlock (my proposal) vs packed into PackedBlock itself (saves indirection, costs geometry re-upload on light change). **My recommendation: parallel array.** Light changes should NOT re-upload geometry.

5. **Emission source for TimeOfDay sun light.** Sky voxels need a "seed" pass each TimeOfDay update (every ~60 ticks). That's ~1 MB/s of sky-top-face voxels at 4096 sections × 256 top voxels × 60 Hz × 2 bytes. Trivial, but need to confirm TimeOfDay can call into the lighting system every frame. Currently TimeOfDay is pure data — should it own a reference to the lighting system, or should the lighting system pull from TimeOfDay?

6. **Vintage Story "32 levels / 16,384 combos" claim.** This figure from MASTER-FEATURE-COMPARISON.md is unverified. I checked public sources (repo, wiki, devlogs) and couldn't find it. If it's load-bearing for a design decision, we should pin down where the number originated before citing it.

---

## 9. Summary

- **Reject:** serial BFS queues (Minecraft/Starlight/Luanti model — CPU-bound, wastes GPU), JFA (approximate distances unacceptable for light levels), LPV/SH (wrong problem), ray tracing (Phase N+1).
- **Accept:** parallel iterative stencil with SWAR nibble parallelism, unified flat light grid, per-corner smooth lighting baked into PackedQuad's reserved bits, delta propagation on dirty section sets.
- **Ship order:** monochrome (4+4) first, colored (4+4+4+4) as follow-up with PackedQuad widening.
- **Risk:** moving dynamic sources (player flashlight while walking) too expensive to flood-fill per frame. Stay on forward rendering for dynamic sources. Flood fill is for STATIC world lighting.
- **Verified facts only.** Any architecture decision traced back to a source in Section 2 or 3. Unverified claims (Vintage Story 32/16384) explicitly flagged as open questions.

Ready for Data's review.

-- Tuvok 🖖
