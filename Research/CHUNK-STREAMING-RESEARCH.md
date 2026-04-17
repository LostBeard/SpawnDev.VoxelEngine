# Chunk Streaming and Memory Residency - Research

**Author:** Tuvok (Claude CLI #3, Research/Planning)
**Date:** 2026-04-16
**For:** Data (VoxelEngine editor), Captain (TJ)
**Context:** SpawnDev.VoxelEngine v1.0.0 plan, spans Phase 8 (persistence) + Phase 13 (optimization). This is the load-bearing infrastructure both phases sit on top of.

---

## Problem statement

Lost Spawns is a DayZ-scale browser voxel game. Target world size on disk approaches 2 million 16x16x16 sections (~32 GB raw, ~2 GB palette-compressed per Phase 15). Of those, only ~2,000 sections are ever simultaneously "near the player" and must be renderable at 60 fps on Quest 3S.

The engine today has zero wiring between cold storage and the GPU. Every primitive needed (VertexPool eviction, StreamingBudget, SectionPriority, LODSelector, ManagedSection state machine, IChunkCacheProvider interface) **already exists in isolation**. None of them call each other. The engine is stateless with respect to the viewer - every culling/meshing API takes `cameraPos` as an argument from the host app.

**The gap:** a loader service that owns the viewer, derives the active section set each frame, drives the async pump, and enforces residency budgets. Without this, Phase 8 (persistence) has nothing to save/load and Phase 13 (optimization) has nothing to prioritize.

---

## Prior art (verified)

### Minecraft Java edition - ticket-based level propagation

**Source:** `net.minecraft.server.level.ChunkMap`, `DistanceManager`, `ChunkLevel` (Mojang mappings). Public wiki: https://minecraft.wiki/w/Chunk.

Every chunk has a numeric "level" (lower = more active). Tickets are placed on chunks by game systems; each ticket defines a level and radius. Level propagates outward from each ticket source at +1 per chunk. The lowest level from all tickets wins. Verified thresholds:

| Level | FullChunkStatus | Behavior |
|-------|-----------------|----------|
| ≤31 | Entity Ticking | full simulation |
| 32 | Block Ticking | blocks tick, entities don't |
| 33 | Border | loaded, accessible, no ticking |
| ≥34 | Inaccessible | world-gen only |

Verified ticket types (level / radius / timeout):
- `PLAYER` - level 31, centered on player
- `FORCED` - level 31, no timeout (persisted via `/forceload`)
- `PORTAL` - level 30, radius 3, 15-second timeout
- `START` - level 22, 23x23 chunks at spawn
- `UNKNOWN` - level >32, 1-tick timeout (emergency load)

**Why it matters here:** ticket propagation cleanly separates *who wants a chunk loaded* from *how far it propagates*. A structure being built, a portal linking two worlds, an explosion event, and a player all participate in the same system. Multiple players naturally union via lowest-level-wins. Timeouts allow temporary loading without permanent state.

**Licensing:** Minecraft is proprietary. Mojang mapping names (`ChunkMap`, `DistanceManager`) are public. DO NOT copy decompiled source. Mechanics (level propagation, ticket types) are independently reimplementable.

### 0fps.net - Mikola Lysenko "An Analysis of Minecraft-like Engines" (2012)

**Source:** https://0fps.net/2012/01/14/an-analysis-of-minecraft-like-engines/

Direct quote on chunk residency:

> "using a hash table to sparsely store only a subset of the pages which are currently required by the game. Chunks can be mapped back to disk when they are not needed (when there are no players nearby), and using a hash map to store the chunks allows one to maintain constant time random access, while simultaneously taking advantage of sparsity as in an octree."

Lysenko also recommends **Morton (Z-order) hashing** for the chunk key to improve spatial cache locality - nearby chunks cluster in memory.

Benchmark numbers from the same article (interval tree vs virtual array for intra-chunk storage):
- Sequential iteration: 0.003-0.006 µs (interval tree) vs 0.210 µs (virtual array)
- Random access: 0.571 µs (interval tree) vs 0.278 µs (virtual array)

**Why it matters here:** validates the "hash map of sections" pattern we already have (`Dictionary<SectionCoord, ManagedSection>` at `ChunkManager.cs:37`). Suggests Morton-ordered keys for better iteration performance when scanning the resident set.

### Luanti (formerly Minetest) - client-reactive cache

**Source:** https://github.com/luanti-org/luanti, `src/mapblock.cpp`, issue #14014 "Client uses a poor block caching strategy", PR #5595 mesh-thread cache.

Client maintains `client_mapblock_limit` blocks in a dictionary. MapBlocks are 16x16x16 nodes (same as our sections). Client does NOT predictively load - it receives blocks from the server and caches with timeout-based eviction. MeshUpdateThread has its own block cache to smooth out bursty neighbor arrivals.

**Why it matters here:** a reminder that client-reactive (receive-and-cache) is viable when storage is remote. AubsCraft-style networked play uses this pattern; Lost Spawns (local procedural generation) does not.

**License flag:** Luanti is **LGPL-2.1-or-later**. DO NOT copy source. Patterns and algorithms are fine.

### Vintage Story - 32x32x32 SQLite-backed chunks

**Source:** https://wiki.vintagestory.at/index.php/Modding:Chunk_Data_Storage

Chunks are 32x32x32 (8x our volume per section). Stored in a SQLite database file (`.vcdbs`) with per-stream compression (blocks, light, liquids in separate protobuf BLOBs). Idle-chunk compression is claimed to save 80-90% memory but **unverified** on primary source.

Recommended view distance: 192-256 blocks normal, 512+ "very demanding". RAM guidance: 8 GB minimum, 16 GB recommended.

Per-frame chunk budgets, eviction cadence, and prefetch strategy are **unverified** - not documented in public wiki.

**Why it matters here:** supports the "region file" pattern already decided for Phase 8. Validates that 32x32 region files backed by a single DB-like handle is production-viable. Confirms that CPU-side memory pressure is the real constraint (GB-range working sets), not I/O bandwidth.

### Nanite (UE5) - GPU-driven cluster streaming

**Source:** Karis et al., SIGGRAPH 2021 Advances presentation (advances.realtimerendering.com/s2021/). Epic docs: https://dev.epicgames.com/documentation/unreal-engine/nanite-virtualized-geometry.

Meshes are broken into clusters of ~128 triangles, grouped into compressed "pages." Visible cluster selection runs on GPU every frame. Only visible clusters are resident. Streaming pool is configurable (`r.Nanite.Streaming.StreamingPoolSize`, commonly 512 MB - exact default unverified in the sources found).

**Why it matters here:** Nanite's "only the visible part is resident" maps directly to our goal. The relevant pattern is NOT triangle clustering (our quads are already small) but the **GPU-driven residency decision** - the GPU tells the CPU what it needs rather than the CPU pushing everything. That's a Phase 13 follow-on; for v1.0 we ship CPU-driven residency and keep the GPU-driven path as an upgrade path.

### Teardown - NOT a streaming engine (intentional)

**Source:** Voxagon Blog, Software Engineering Daily interview with Dennis Gustafsson (2025-01-02).

Teardown uses "thousands of smaller volumes" (per-object voxel grids, translatable/rotatable) rather than one global grid. Levels fit entirely in RAM. No streaming system.

**Why it matters here:** confirms that if the world fits in memory, streaming is a liability, not a feature. For Lost Spawns at 2M sections this is not viable. For a Teardown-style destructible small-world game, it would be. Our architecture must not force streaming on games that don't need it.

---

## Current state survey (2026-04-16)

Verified by reading source in `D:\users\tj\Projects\SpawnDev.VoxelEngine\SpawnDev.VoxelEngine\SpawnDev.VoxelEngine\`:

### What exists (primitives in isolation)

| Primitive | File | State |
|-----------|------|-------|
| `ChunkManager._sections` dictionary | `ChunkManager.cs:37` | works, but `LoadSectionAsync` is never called from here |
| `ManagedSection` state machine (CACHED etc.) | `ChunkManager.cs:24` | enum defined, transitions manual |
| `IChunkCacheProvider` interface | `Caching/IChunkCacheProvider.cs:16` | interface + default `LoadBatchAsync` fallback |
| `InMemoryChunkCache` | `Caching/IChunkCacheProvider.cs:63` | test/demo only, no eviction, no persistence |
| `VertexPool` + `EvictHighestScore` | `Buffers/VertexPool.cs:155` | bucket allocator with `distSq * bucketCount` scorer at line 200 |
| `BufferCompaction.Plan/ExecuteAsync` | `Buffers/BufferCompaction.cs:78,146` | 30% threshold at line 68 |
| `IndirectDrawBuffer` | `Buffers/IndirectDrawBuffer.cs:61,85` | end-swap `Add/Remove` |
| `StreamingBudget.ComputePriority` | `Adaptive/StreamingBudget.cs:111` | `(distSq, viewDot, lodUrgency)` → score |
| `SectionPriority` comparable struct | `Adaptive/StreamingBudget.cs:131` | defined, never instantiated |
| `LODSelector.SelectLOD` | `LOD/LODSelector.cs:28` | hysteresis + morph, never called from pipeline |
| `DrawDistance` | `Contracts.cs:208` | int config value, scaled by `ThermalManager.cs:49`, read by `CullingPipeline.cs:75` |

### What does not exist (the entire glue layer)

1. **No ChunkLoaderService / ChunkStreamer** - nothing owns "the player is here, these sections should be loaded."
2. **No viewer concept** - no `IViewer`, no `PlayerPosition` field anywhere. Every API takes `Vector3 cameraPos` as a parameter.
3. **No OPFS provider** - only `InMemoryChunkCache`. `VoxelEngineServiceExtensions.cs` does not register any cache provider; DI throws if none is supplied by the host.
4. **No async pump** - no `BackgroundService`, no `Task.Run`, no worker that drains a load queue.
5. **No priority queue instance** - `SectionPriority` exists but is never instantiated into a `SortedSet<>` or heap.
6. **No residency enforcer** - `VertexPool.Free`, `IndirectDrawBuffer.Remove`, and `ChunkManager.MarkUnloaded` are never called from a common eviction path.
7. **No per-frame LOD pump** - `LODSelector` is present but `SectionEntry.LodLevel` at `Contracts.cs:87` is never written from the library.

**All six missing pieces are wiring, not new algorithms.** Every algorithm we need already exists.

---

## Proposed architecture

### Design principles

1. **Independent ticket propagation.** Ports Minecraft's pattern without copying code. Any system (player, portal, explosion, structure build) can place a ticket. Lowest level wins.
2. **Stateless engine, stateful service.** The core engine keeps its stateless-viewer contract for tests. All viewer state lives in a new optional `ChunkLoaderService` that host apps opt into.
3. **Wire existing primitives first, add nothing new.** Phase 16.1 is 100% wiring. No new kernels, no new buffer types, no new compression. Just a service that calls what's already there.
4. **CPU-driven residency for v1.0.** GPU-driven (Nanite-style) is a Phase 13 follow-on. CPU decides which sections are resident; GPU is told.
5. **Decouple mesh residency from voxel residency.** A section's voxel data (cheap, ~1 KB palette-compressed) can live in CPU memory longer than its mesh (expensive, 10-100 KB vertex buffer). Meshes evict first on GPU pressure; voxel data stays to allow fast re-mesh without an OPFS round-trip.

### Ticket system

Introduce `SectionTicket`:

```
struct SectionTicket {
  TicketType Type;       // Player, Portal, ForceLoad, Structure, Unknown
  SectionCoord Origin;
  int Level;             // lower = more active
  int Radius;
  long ExpiryTick;       // -1 = no expiry
  int OwnerId;           // for removal
}
```

Levels (ported from Minecraft, tuned for our 16³ sections):

| Level | Name | Behavior |
|-------|------|----------|
| ≤28 | Rendered | full render, meshed, in IndirectDrawBuffer |
| 29-30 | Meshed | mesh exists in VertexPool, not yet in draw buffer |
| 31-32 | Loaded | voxel data in CPU memory, no mesh |
| 33-35 | Prefetched | async load in flight |
| ≥36 | Evicted | not resident |

Example: a `PLAYER` ticket at level 28, radius 0, propagates so sections at distance 1 get level 29, distance 2 get 30, etc. A player with draw distance 8 (128 blocks) reaches level 36 at distance 8.

### Residency state machine

Extend `ManagedSection` state machine:

```
Evicted → Prefetching → Loaded → Meshing → Meshed → Rendered
   ↑          ↓            ↓        ↓         ↓        ↓
   └──────────┴────────────┴────────┴─────────┴────────┘
             (any state can fall back to Evicted)
```

Every frame, the loader service:
1. Computes each section's target level from active tickets
2. Compares to current state level
3. Dispatches transitions (async for I/O-bound transitions, sync for GPU transitions)

### Viewer abstraction

```
interface IViewer {
  Vector3 Position { get; }
  Vector3 Forward { get; }
  int DrawDistance { get; }
  float Velocity { get; }  // for predictive prefetch
}
```

`ChunkLoaderService` takes `IEnumerable<IViewer>` (handles VR's two eyes, multiplayer split-screen, camera-vs-player-vs-spectator). Each frame it unions ticket coverage across all viewers. Tests pass a `MockViewer` - engine stays unit-testable with zero viewer state owned by the library core.

### Async pump (SpawnDev.BackgroundServices)

Use TJ's proven `SpawnDev.BackgroundServices` pattern (memory reference: `reference_background_services.md`). Service has a `Ready` dependency on `IChunkCacheProvider`, so a null-provider host still starts cleanly (the service just never dispatches).

```
class ChunkLoaderService : BackgroundService {
  private readonly PriorityQueue<SectionCoord, int> _loadQueue;
  private readonly IChunkCacheProvider _cache;

  protected override async Task ExecuteAsync(CancellationToken ct) {
    while (!ct.IsCancellationRequested) {
      var budget = _budget.AcquireFrameBudget();
      while (budget.HasCapacity && _loadQueue.TryDequeue(out var coord, out _)) {
        var data = await _cache.LoadSectionAsync(coord);
        _manager.MarkLoaded(coord, data);
        budget.Consume(1);
      }
      await Task.Yield();
    }
  }
}
```

Key detail: loads run off the render thread. On Wasm backend, this is the main thread but cooperative via `Task.Yield`. On desktop, true thread parallelism. `IChunkCacheProvider.LoadBatchAsync` batches multiple sections per OPFS read to amortize region-file decompression cost (matches AubsCraft's proven 310 MB/s region-file pattern).

### Residency enforcement (eviction)

Every frame after loader pump:
1. Walk sections with level ≥36 → call `ChunkManager.MarkUnloaded`, `VertexPool.Free`, `IndirectDrawBuffer.Remove`
2. If VertexPool is over budget, call existing `EvictHighestScore` using existing `distSq * bucketCount` scorer
3. If IndirectDrawBuffer fragmentation >30%, schedule `BufferCompaction.Plan/Execute`

Eviction score extension (add to `VertexPool.CreateDistanceScorer`):

```
score = baseDistSq * bucketCount
      - (hasPlayerTicket ? InfinityPenalty : 0)
      + (outOfFrustum ? FrustumBonus : 0)
      + (facingAway ? FacingBonus : 0)
```

Player-ticketed sections cannot be evicted even under pressure - they drop frames instead of visibility.

### Prefetch ring (load > render)

Lesson from Minecraft: render distance ≠ simulation distance. For us: render distance ≠ load distance.

| Ring | Radius (sections) | Radius (blocks) | Purpose |
|------|-------------------|-----------------|---------|
| Render | 8 | 128 | in IndirectDrawBuffer, meshed, drawn |
| Mesh | 12 | 192 | mesh exists, lazy add to draw buffer |
| Load | 16 | 256 | voxel data in CPU memory, no mesh yet |
| Prefetch | 20 | 320 | async load in flight, velocity-biased |

Prefetch bias: for a player moving at velocity `v`, bias prefetch ring forward by `v * lookaheadSeconds`. Cheap and effective - matches how Minecraft Bedrock handles fast riding/flying. Velocity source is the viewer's delta-position across frames.

### Morton ordering for section keys

Add a `SectionCoord.ToMorton()` helper that interleaves x/y/z bits into a single ulong. Use as the dictionary key - nearby sections cluster in memory. Lysenko's recommendation, costs nothing, measurable cache locality improvement for eviction sweeps.

### DI default provider

Ship a `NullChunkCacheProvider` that returns `null` for every `LoadSectionAsync` call. Register as the default in `AddVoxelEngine(..)`. Host apps override with their real provider (AubsCraft SignalR, Lost Spawns OPFS). Means DI resolves cleanly with zero host config, and tests can swap in `MockChunkCacheProvider` with one line.

---

## Tests (10 cases, expanded from implicit "should work")

1. **Ticket propagation**: place PLAYER ticket at (0,0,0), verify section (8,0,0) has level 36, section (0,0,0) has level 28.
2. **Ticket removal**: add then remove a ticket, verify section reverts to the next-highest ticket's level.
3. **Multi-viewer union**: two viewers at (0,0,0) and (100,0,0), verify sections near either get proper levels, sections between them get the lower level.
4. **Load pump dispatch**: enqueue 100 loads with budget 10/frame, verify exactly 10 load per frame tick.
5. **Residency enforcer frees GPU**: mark section evicted, verify VertexPool.Free called and IndirectDrawBuffer.Remove called.
6. **Eviction respects player ticket**: force VertexPool over budget, verify ticketed section is NOT evicted.
7. **Prefetch ring size**: player at origin with draw 8, verify load ring is 16, prefetch ring is 20.
8. **Velocity bias**: player moving +X at 10 blocks/s, verify prefetch ring asymmetry toward +X.
9. **Morton key clustering**: scan sections near origin, verify dictionary bucket locality (no direct test of locality, test that `SectionCoord.ToMorton()` round-trips).
10. **NullChunkCacheProvider**: default DI, zero host config, verify engine starts without throwing.

---

## Phased rollout

### Phase 16.1: Wiring (no new algorithms)

- `IViewer` interface + `SingleViewer` impl
- `SectionTicket` struct + `TicketManager` with level propagation
- `ChunkLoaderService : BackgroundService`
- `NullChunkCacheProvider` default DI registration
- Wire `StreamingBudget.ComputePriority` → `PriorityQueue`
- Wire `LODSelector.SelectLOD` per frame into `SectionEntry.LodLevel`
- Wire eviction: level ≥36 → `VertexPool.Free` + `IndirectDrawBuffer.Remove` + `ChunkManager.MarkUnloaded`
- Tests 1-6, 10

**Deliverable:** engine pulls a section from cache, meshes it, renders it, and evicts it cleanly. End-to-end streaming with a procedural `IChunkCacheProvider` driving the tests.

### Phase 16.2: Prefetch ring + velocity bias

- Add load/mesh/render ring distinction in `TicketManager`
- Velocity tracking on `IViewer`
- Velocity-biased prefetch
- Tests 7-8

### Phase 16.3: Morton ordering

- `SectionCoord.ToMorton()` helper
- Switch `Dictionary<SectionCoord, ManagedSection>` key to Morton-ordered ulong
- Test 9

### Phase 16.4: Multi-viewer (VR + split-screen ready)

- `ChunkLoaderService` accepts `IEnumerable<IViewer>`
- Ticket union across viewers
- Test 3

### Phase 16.5 (optional, v1.1 if scope-squeezed): GPU-driven residency

- Hi-Z occlusion feeding back into residency decisions
- GPU writes "this section was culled 60 frames in a row" → eligible for mesh eviction
- Nanite-style pattern, measurable win on dense scenes

---

## Risk analysis

**Risk: DI breaks for host apps that already register a custom provider.** Mitigation: use `TryAddSingleton` in `AddVoxelEngine` so host-supplied providers win. Verified pattern in every other SpawnDev lib.

**Risk: BackgroundService + Wasm = main thread contention.** Mitigation: `Task.Yield()` + small per-tick budget. AubsCraft already runs BackgroundServices on Wasm cleanly (verified). On desktop, true threading gives headroom.

**Risk: eviction thrashing at draw distance boundary.** Mitigation: hysteresis on level transitions (load at level 35, evict at level 37 - same pattern `LODSelector` already uses at `LOD/LODSelector.cs:40-51`).

**Risk: IChunkCacheProvider contract changes break AubsCraft.** Mitigation: AubsCraft is currently NOT using the library's loader path (has its own SignalR chunk delivery). Phase 16.1 adds new surface, doesn't change existing contract. AubsCraft can opt in later.

**Risk: OPFS region-file coordinator (Phase 8) not ready when Phase 16.1 ships.** Mitigation: Phase 16.1 uses `NullChunkCacheProvider` for tests, `InMemoryChunkCache` for demos, and a procedural `IChunkCacheProvider` stub that Lost Spawns can ship even before OPFS integration. OPFS is a Phase 8 deliverable that slots in under the existing interface.

**Risk: multi-viewer (Phase 16.4) complicates VR ship gate.** Mitigation: VR uses a single head position for residency decisions even with two eye cameras - both eyes see the same sections. Multi-viewer is really "multiple players sharing a world," not VR.

---

## Open questions for Captain

1. **Load vs render radius knobs** - one `DrawDistance` config that the engine derives all rings from (simpler), or four separate configs `RenderRadius / MeshRadius / LoadRadius / PrefetchRadius` (more control)? I recommend one `DrawDistance` with fixed ratios `{1.0, 1.5, 2.0, 2.5}` for v1.0, add overrides in v1.1 if profiling demands.

2. **Mesh vs voxel residency decoupling** - ship decoupled (voxel data stays in CPU longer than mesh on GPU) or coupled (evict voxel and mesh together for simplicity)? I recommend decoupled - the fast-remesh-without-OPFS path is a big Quest 3S win and costs ~16 KB extra CPU per section.

3. **Multi-viewer support in v1.0** - ship from day one (adds complexity to ticket propagation tests) or single-viewer only for v1.0 and multi-viewer in v1.1 when AubsCraft multiplayer lands? I recommend single-viewer for v1.0 with `IViewer` interface designed for later expansion. `IEnumerable<IViewer>` can accept a singleton collection today.

4. **Force-load tickets** - ship in v1.0 (structures, portals, `/forceload` command) or defer to v1.1? I recommend ship - the ticket system naturally supports them and they're load-bearing for Lost Spawns' procedural structures (military bases that must finish generating even if player walks away).

5. **Predictive prefetch aggressiveness** - cap at `v * 2s` lookahead (conservative, matches Minecraft Bedrock) or `v * 5s` (aggressive)? I recommend `v * 2s` for v1.0. Too aggressive wastes I/O on sections the player turns away from.

6. **Null cache provider** - ship `NullChunkCacheProvider` as the DI default so engine starts without host config, or require host to supply one (forces explicit choice)? I recommend ship Null - matches SpawnDev patterns, host can still supply a real one that wins via `TryAdd`.

7. **Frame budget target** - what's the ship gate for Quest 3S? I recommend: at 60 fps with 2000 active sections, streaming must consume ≤1ms/frame for loader pump + ≤0.5ms/frame for residency enforcer. If we miss, turn the knob on ring radii before turning the knob on compute.

---

## Summary

Phase 16 is 90% wiring and 10% new code. Every algorithm needed is already in the repo - VertexPool eviction, StreamingBudget priority, LODSelector hysteresis, BufferCompaction, IndirectDrawBuffer - all waiting for a service that calls them together.

The design adapts Minecraft's ticket-based level propagation (independently implemented, not copied), uses Lysenko's hash-map-of-sections pattern (already in place, add Morton ordering), keeps the engine stateless for tests (viewer state lives in an opt-in service), and ships a default `NullChunkCacheProvider` so DI resolves without host config.

Five phased deliverables, 10 test cases, 7 open questions for Captain.

Live long and prosper.

-- Tuvok
