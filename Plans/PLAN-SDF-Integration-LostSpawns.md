# PLAN: SDF Integration into Lost Spawns

**Status:** Draft. Implementation recipe, not a brainstorm.
**Owner:** Data (VoxelEngine editor).
**Consulted:** Tuvok (PLAN-Terrain-Carving.md gameplay), Captain (approvals).
**Last updated:** 2026-04-17

---

## Scope

This plan covers the **consumer-side integration** of the VoxelEngine SDF/DMC subsystem (Phase 3 library work) into Lost Spawns' `WorldService`, `VoxelEngineService`, and `RenderService`. It answers: **how do we actually drop the SDF pipeline into the existing block-based world, without breaking the shipped Minecraft-style renderer?**

**Out of scope (covered elsewhere):**

- Gameplay design of carving tools, moldable brushes, traps, arms race - see `D:/users/tj/Projects/Lost/Lost/Plans/PLAN-Terrain-Carving.md` (Tuvok, extensive).
- Library-side phase roadmap (Phase 3 kernels exist, Phase 8 OPFS persistence, Phase 13 StreamingBudget) - see `Plans/PLAN-v1.0.0-LostSpawns.md` (Data).
- WebGPU BindGroup LEA codegen fix - shipped by Geordi in ILGPU 4.9.2-rc.7.

**What this plan owns:**

1. The concrete contract between block storage (`ChunkData.Blocks`) and SDF storage (`SdfChunk` at scale 256 fixed-point).
2. The hybrid render pipeline (both greedy-quad and DMC meshes coexist per section).
3. The `ITerrainCarve` API surface - signatures Lost Spawns actually calls.
4. The `CarveService` consumer-side wiring (input -> library -> re-mesh -> persistence).
5. Phase ordering - what lands first, what gates what.
6. Desktop-only unit test coverage for CSG edge cases.

---

## Current state (what exists, what does not)

### Library-side (SpawnDev.VoxelEngine) - verified 2026-04-17

- [x] `SDF/SdfChunk.cs` - 16-bit fixed-point signed distance, scale 256, DefaultSize=32, 32^3 chunks.
- [x] `SDF/SdfNoiseKernels.cs` - 6-layer noise terrain (base/hills/mountains/detail/caves/overhangs), Hash3D, ValueNoise3D, FBM, DomainWarpedFBM.
- [x] `SDF/SdfNoiseKernels.cs` - CSG ops: `SdfUnion` (max), `SdfSubtract`, `SdfIntersect` (min), `SdfSmoothUnion`, `SdfSmoothSubtract` (blend radius k).
- [x] `SDF/SdfNoiseKernels.cs` - `ModifySdfSphereKernel` (mode 0 = smooth dig, mode 1 = smooth fill).
- [x] `SDF/SdfMeshPipeline.cs` - 5-stage GPU pipeline (Evaluate -> Classify -> PrefixSum -> Vertex -> Quad), 5 compiled kernels, shared reusable buffers.
- [x] Hybrid render library contract (sectioned mesh results shipped 2026-04-16 late).
- [~] 16 SDF/DMC WebGPU tests authored - currently blocked on ILGPU rc.7 (WGSL sub-word LEA codegen fix).
- [ ] `ITerrainCarve` unified API surface - **not yet shipped**.
- [ ] `ModifySdfBrushKernel` (flatten, smooth, ramp profiles - for moldable terrain) - **deferred to Phase 4**.

### Consumer-side (Lost Spawns) - verified 2026-04-17

- [x] Block world fully working: `WorldService` + `VoxelEngineService.GenerateChunkMeshesAsync` (greedy mesh, neighbor-aware XZ + intra-chunk Y padding).
- [x] Heightmap input path: Perlin GPU kernel OR `HeightmapLoader` (Deer Isle data).
- [x] 16 vertical sections per chunk, GPU-resident `PackedQuad` buffers, no CPU readback after gen.
- [ ] **Zero SDF usage today.** `TerrainGenerator.cs` only produces blocks. `WorldService._chunks` only stores block meshes.
- [ ] **No carving today.** No input -> modification pipeline, no dirty-chunk tracking, no re-mesh after mod, no persistence.

### The gap this plan closes

1. Library needs `ITerrainCarve` API (single entry point; internally routes block vs SDF).
2. Consumer needs dual storage (blocks + SDF per section) and dual render path.
3. Consumer needs `CarveService` (input -> `ITerrainCarve.ApplySphere` -> dirty flag -> re-mesh queue).
4. Library and consumer both need persistence hooks (OPFS region files for dirty chunks - Phase 8).

---

## Architecture

### Storage model: dual representation per section

A "section" in Lost Spawns is a 16x16x16 sub-chunk. There are 16 sections per chunk column (16x16x256 total).

```
Section (16x16x16):
  +-- BlockPacked: int[18*18*16] padded   // greedy mesh source (existing, block-based)
  +-- SdfField: short[32*32*32] fixed     // SDF mesh source (NEW; higher resolution)
  +-- Flags:
      - HasBlockMesh: bool                  // some cells are blocky (buildings, trees, ores)
      - HasSdfMesh: bool                    // some cells are smooth (terrain under blocks)
      - DirtyBlock: bool                    // needs block re-mesh
      - DirtySdf: bool                      // needs SDF re-mesh
```

**Rationale for different resolutions:**

- Blocks are 1-voxel-per-world-unit. 16^3 = 4096 cells per section = 4KB at 1-byte-per-block.
- SDF is 2-voxels-per-world-unit (oversampled for quality). 32^3 = 32K cells per section at 2 bytes = 64KB.
- SDF oversampling gives smooth DMC output at block-scale features without coarser blobs.

**Memory budget check:** 25MB current block mesh + 64KB * 16 sections * 625 loaded columns = 640MB SDF fields. Too much. **Mitigation:** SDF field stored compressed in OPFS, only decompressed on-demand when section is near player. Dormant sections carry only the DMC mesh (GPU-resident) + a dirty flag.

### Hybrid render pipeline

```
  input (heightmap + noise seeds + optional carves)
    |
    v
  [VoxelEngineService]
    |
    +-- GenerateChunkMeshesAsync(blocks, xz-neighbors, y-pads)
    |       -> List<(sectionY, greedy quad buffer)>      // existing, block path
    |
    +-- GenerateSdfMeshesAsync(sdfField, xz-neighbors, y-pads)  // NEW, SDF path
    |       -> List<(sectionY, dmc triangle buffer)>
    |
    v
  [WorldService]
    _chunks[(cx,sy,cz)] = SectionMesh {
      BlockQuadBuffer: GPUBuffer,
      BlockQuadCount: int,
      SdfTriangleBuffer: GPUBuffer,
      SdfTriangleCount: int,
    }
    |
    v
  [RenderService]
    foreach section:
      if HasBlockMesh: VertexPullPipeline.DrawPackedQuads(blockBuffer)
      if HasSdfMesh: DmcPipeline.DrawTriangles(sdfBuffer)
```

**Key invariant:** a section can have BOTH meshes. A natural cliff has SDF mesh; a log cabin built against it has block mesh. Both render in the same frame, same depth buffer. Front-to-back order matters only for transparency (water, leaves) - reversed-Z already handles opaque correctly.

### Unified carving API (library surface to add)

```csharp
namespace SpawnDev.VoxelEngine.Carving;

public interface ITerrainCarve
{
    // Sphere is the one primitive for v1.0. Flatten/ramp brushes defer to Phase 4.
    // material:  "source" material for fill ops (dirt, stone); ignored for dig ops.
    // toolPower: 0=bare hands, 1=shovel, 2=pickaxe, 3=drill, 4=explosive.
    // Returns modified section keys for re-mesh scheduling.
    Task<IReadOnlyList<(int cx, int sy, int cz)>> ApplySphereAsync(
        Vector3 worldCenter, float radius,
        CarveMode mode, byte material, byte toolPower);
}

public enum CarveMode
{
    Dig,       // smooth subtract on SDF; block destruction on block cells
    Fill,      // smooth union on SDF; block placement on block cells
    Explode,   // noise-perturbed dig, larger radius, ignores toolPower
}
```

**Implementation notes:**

- `ApplySphereAsync` delegates to `ModifySdfSphereKernel` (smooth) for SDF cells and to `ExplosionKernels.DestroyInSphere` for block cells within the radius.
- Material lookup on dig returns the material to add to player inventory (conservation).
- Returns affected section keys so `WorldService` can schedule DMC + greedy re-mesh.
- Thread safety: internally serialize dispatches per accelerator (existing `_meshLock` pattern).

---

## Library surface additions (new files in SpawnDev.VoxelEngine)

| File | Role |
|------|------|
| `Carving/ITerrainCarve.cs` | Public contract (above). |
| `Carving/TerrainCarveService.cs` | Default impl, composes SDF + block kernels. |
| `Carving/CarveMode.cs` | Enum. |
| `SDF/SdfDirtyRegionTracker.cs` | Per-section dirty flag + affected-section bounds reporting. |
| `Meshing/HybridChunkMesher.cs` | Given blocks + SDF field, produces both mesh outputs in one pipeline. |
| `Tests/CarvingTests.cs` | **Desktop-only** CSG edge cases (no browser auth needed). |

**No library refactor of existing SDF kernels.** They already ship the math. We add composition around them.

---

## Lost Spawns consumer changes

### `Services/VoxelEngineService.cs`

Add:

```csharp
// NEW: GPU-resident SDF field storage.
private MemoryBuffer1D<short, Stride1D.Dense>? _sdfFieldBuffer;
private SdfMeshPipeline? _sdfPipeline;

public async Task<List<(int sectionY, DmcMeshResult mesh)>> GenerateSdfMeshesAsync(
    short[] sdfField, /* neighbor fields */);

public ITerrainCarve CarveApi => _carveApi!;  // injected from library
```

Initialization in `InitAsync`:

```csharp
_sdfPipeline = new SdfMeshPipeline(_accelerator);
_carveApi = new TerrainCarveService(_accelerator, _sdfPipeline, _meshPipeline);
```

### `Services/WorldService.cs`

Extend `ChunkMesh` (call it `SectionMesh`) to hold dual buffers:

```csharp
public class SectionMesh : IDisposable
{
    public GPUBuffer? BlockQuadBuffer { get; init; }
    public int BlockQuadCount { get; init; }
    public GPUBuffer? SdfTriangleBuffer { get; init; }
    public int SdfTriangleCount { get; init; }
    public bool HasBlockMesh => BlockQuadCount > 0;
    public bool HasSdfMesh => SdfTriangleCount > 0;
    public bool HasAnyMesh => HasBlockMesh || HasSdfMesh;
    // ... dispose both buffers
}
```

Extend `GenerateChunkGpuAsync` to generate SDF field alongside blocks:

```csharp
// Today:
chunk = _generator!.GenerateChunkFromHeightmap(cx, cz, heightmap);
var sectionMeshes = await _engine.GenerateChunkMeshesAsync(chunk.Blocks, ...);

// Proposed:
chunk = _generator!.GenerateChunkFromHeightmap(cx, cz, heightmap);
var sdfField = await _engine.GenerateSdfFieldAsync(cx, cz);   // NEW: GPU-side noise
var sectionMeshes = await _engine.GenerateHybridChunkMeshesAsync(chunk.Blocks, sdfField, ...);
```

Dirty-chunk tracking for re-mesh (new):

```csharp
private readonly Queue<(int cx, int sy, int cz)> _remeshQueue = new();

public void QueueReMesh((int cx, int sy, int cz) key) { _remeshQueue.Enqueue(key); }
public async Task ProcessReMeshQueueAsync(int maxPerFrame) { /* ... */ }
```

### `Services/CarveService.cs` (new)

```csharp
public class CarveService
{
    private readonly WorldService _world;
    private readonly VoxelEngineService _engine;
    private readonly InputService _input;

    public async Task OnPrimaryHitAsync(Vector3 worldPoint, byte toolPower)
    {
        var affected = await _engine.CarveApi.ApplySphereAsync(
            worldPoint, radius: 1.5f, CarveMode.Dig,
            material: 0, toolPower);

        foreach (var key in affected)
            _world.QueueReMesh(key);
    }
}
```

Register in `Program.cs`:

```csharp
builder.Services.AddSingleton<CarveService>();
```

### `Services/RenderService.cs`

No structural change to the renderer. Add a second draw pass:

```csharp
foreach (var (key, section) in _world.Sections)
{
    if (section.HasBlockMesh)
        _vertexPullPipeline.Draw(section.BlockQuadBuffer!, section.BlockQuadCount);

    if (section.HasSdfMesh)
        _dmcPipeline.Draw(section.SdfTriangleBuffer!, section.SdfTriangleCount);
}
```

The `DmcPipeline` is a thin wrapper around the library's DMC vertex format + a triplanar terrain material shader. Already shipped library-side per my 2026-04-16 EOD.

---

## Phase order (what lands in what order)

### Phase A - **blocked on** ILGPU 4.9.2-rc.7 publish

1. Geordi publishes rc.7.
2. I bump `SpawnDev.VoxelEngine.csproj` PackageReference 4.9.2-rc.6 -> 4.9.2-rc.7-local.1.
3. I re-run WebGPU sweep on existing 16 SDF/DMC tests. Target: 215/18/0/233 -> 231/2/0/233.
4. Captain gates official rc.7 publish.

### Phase B - library carving API (~1 day, desktop verifiable)

5. Add `Carving/ITerrainCarve.cs` + `TerrainCarveService.cs` + `CarveMode.cs` (library).
6. Add `Carving/CarvingTests.cs` - desktop-only CSG edge cases:
   - SmoothUnion with identity input (k=0) == Union(max).
   - SmoothSubtract with disjoint spheres == no-op.
   - `ModifySdfSphereKernel` mode 0 then mode 1 at same center == approximate round-trip within SDF quantization error.
   - Sphere radius 0 == no cells modified.
   - Sphere outside chunk bounds == no cells modified.
7. Run tests on CPU backend (no Captain browser-auth needed). Ship if green.

### Phase C - consumer SDF storage + hybrid render (~1 day, browser verifiable)

8. Lost Spawns: extend `VoxelEngineService` with `GenerateSdfFieldAsync` (reuses Phase 3 `EvaluateSdfKernel`).
9. Lost Spawns: extend `WorldService.SectionMesh` for dual buffers.
10. Lost Spawns: wire `RenderService` second draw pass.
11. Browser verify: natural terrain renders as smooth DMC under blocky buildings/trees. Captain confirms in demo.

### Phase D - consumer carving wiring (~1 day, browser verifiable)

12. Lost Spawns: add `CarveService` + input binding (left-click = dig, right-click = fill while holding shovel/pickaxe).
13. Lost Spawns: `WorldService.ProcessReMeshQueueAsync` hooked into frame loop.
14. Browser verify: click terrain, crater appears, re-meshes within one frame (per StreamingBudget cap).

### Phase E - persistence (Phase 8 library work, deferred)

15. `SdfDirtyRegionTracker` flags dirty sections.
16. OPFS region files persist dirty SDF fields + dirty block arrays on quit/periodic flush.
17. Re-load restores carved state across sessions.

---

## Test strategy

### Desktop-only CSG tests (no session-scope auth needed)

I can run these without asking Captain for browser-auth re-grant, since they target CPU accelerator only. Covers Phase B ship gate.

| Test | Input | Expected |
|------|-------|----------|
| `SmoothUnion_Identity_EqualsUnion` | k=0, two spheres | output == `Max(a,b)` per cell |
| `SmoothUnion_Disjoint_EqualsUnion` | k=0.5, spheres 10u apart | output == `Max(a,b)` per cell (far apart, blend has no effect) |
| `SmoothUnion_Overlap_LessThanUnion` | k=0.5, spheres touching | output < `Max(a,b)` at overlap region (smoothing lowers min) |
| `SmoothSubtract_Identity_EqualsSubtract` | k=0, two spheres | output == `Max(a, -b)` per cell |
| `ModifySphere_Dig_Fill_RoundTrip` | dig sphere, fill same sphere | output within quantization error of original |
| `ModifySphere_RadiusZero_NoOp` | radius 0 | field unchanged |
| `ModifySphere_OutOfBounds_NoOp` | center 1000u away | field unchanged |
| `SdfEvaluate_Deterministic_SameSeed` | same seed, same coords | identical output |

### Browser tests (Phase C/D, needs Captain's re-grant)

| Test | Verifies |
|------|---------|
| `HybridRender_BlockAndSdfCoexist` | section with blocks + SDF renders both |
| `CarveSphere_BrowserEndToEnd` | click -> dig -> crater visible within N frames |
| `CarveSphere_DirtyFlag_Persists` | carve -> refresh -> carve state restored (needs OPFS) |

**Note:** Phase B is shippable as a standalone library release even if I don't have browser-test authority yet. That's the value of desktop-only coverage.

---

## Risks + open questions

### Risks

1. **Memory budget** - 640MB SDF fields if all loaded eagerly. Mitigation: OPFS compression + dormant-section eviction. Need measurement on Quest 3S before committing.
2. **Re-mesh cost** - carving a large blast triggers N-section DMC re-mesh. Must fit StreamingBudget (Phase 13). Needs benchmark.
3. **SDF/block boundary** - when player carves at an SDF cliff face with a built block wall on top, the carve hits both. The contract for "block becomes debris" + "SDF smooth dent" in one operation must be atomic from the player's POV. Design needed.
4. **Material conservation** - `ApplySphereAsync` returns modified sections, but does NOT currently return material counts for inventory addition. Add `CarveResult { List<(int cx, int sy, int cz)> sections; Dictionary<byte, int> materialCounts; }`.

### Open questions (deferred unless Captain needs them now)

1. Do we need SDF at LOD1/LOD2 for distant chunks? DMC is already vertex-light; may be unnecessary.
2. Sub-chunk re-meshing (Phase 3 `[UNDECIDED]` in Tuvok's plan) - defer until performance measurement says we need it.
3. Material-ID storage for SDF cells (beyond signed distance). Smallest viable format: 16-bit SDF + 8-bit material = 3 bytes/voxel, rounds up to 32KB -> 96KB per section. Would push memory budget. Possibly a separate sparse material array rather than per-voxel storage.
4. Does the `ITerrainCarve` API belong in `SpawnDev.VoxelEngine.Carving` or `SpawnDev.VoxelEngine` root namespace? Leaning Carving - it's a bounded concern.

---

## Dependencies on other plans

| Need | Provided by |
|------|-------------|
| SDF kernels (Evaluate, Classify, Vertex, Quad) | Phase 3 (shipped) |
| DMC vertex format + triangle buffer | Phase 3 (shipped) |
| WGSL sub-word LEA codegen fix | ILGPU rc.7 (in flight today) |
| Hybrid render library contract | shipped 2026-04-16 late |
| OPFS region files | Phase 8 (pending) |
| StreamingBudget (re-mesh cap) | Phase 13 (pending) |
| Gameplay design (tools, moldable brush, arms race) | `PLAN-Terrain-Carving.md` (Tuvok) |
| World biome noise tuning (feeds `EvaluateSdfKernel`) | `PLAN-World-Biomes-Regions.md` (Tuvok) |

---

## Immediate next step when unblocked

When ILGPU rc.7 publishes:

1. Bump VoxelEngine package ref to rc.7-local.1.
2. Run existing WebGPU SDF/DMC sweep. Expected 16 tests flip green.
3. If green, start Phase B library carving API (desktop-only, no Captain auth needed).
4. Post DevComms Phase B start + ETA.

---

*Make it so.*

-- Data
