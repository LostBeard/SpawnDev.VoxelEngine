# Palette Compression Research — Phase 15

**Author:** Tuvok (Claude CLI #3, Research/Planning)
**Date:** 2026-04-16
**Audience:** Data (VoxelEngine editor), Captain (TJ)
**Status:** Research complete. Architecture proposed. Awaiting review before Phase 15 kickoff.
**Companion doc:** `FLOOD-FILL-LIGHTING-RESEARCH.md` (Phase 14) — light data compresses the same way.

**References in this repo:**
- `Plans/PLAN-v1.0.0-LostSpawns.md` — Phase 15 spec (7 checkboxes)
- `Research/compression-techniques-quest3s.md` — Data's 2026-04-14 compression survey (palette math, TurboQuant-style operate-on-compressed insights). **Treat that as Part 1 of this topic.**
- `SpawnDev.VoxelEngine/PackedBlock.cs` — current 16-bit block format (12 type + 4 damage)
- `SpawnDev.VoxelEngine/ChunkManager.cs` — stores `int[] BlockData` per section, 16 KB each
- `SpawnDev.VoxelEngine/Caching/IChunkCacheProvider.cs` — boundary where compression must happen (Load/Save async)
- `SpawnDev.VoxelEngine/BlockRegistry.cs` — block properties for category classification

---

## 1. Problem Statement

Every loaded section holds a 16³ × 4-byte array = **16 KB of raw PackedBlock data per section** (or 8 KB if stored as ushort per v4.9.0+ sub-word). With `MaxLoadedSections = 4096`, that's 64 MB of block data RAM at worst. OPFS region files persist the same raw format — 16 KB per section × 1024 sections per region file = 16 MB per region on disk. Quest 3S on an 8 GB shared-RAM device cannot afford this.

Most natural-terrain sections use ≤ 8 unique block types (surface biomes ≈ 3-4, caves ≈ 5-8, underground ≈ 1). Storing 12 bits per voxel when the section only needs 2-3 bits is waste. Data's prior research quantifies the opportunity:

| Section Type | Unique Types | Uncompressed | Palette-Compressed | Ratio |
|---|---|---|---|---|
| Uniform (air/stone) | 1 | 16 KB | 2 B (palette only) | 8000x |
| Simple surface | 3-4 | 16 KB | 1 KB | 16x |
| Mixed terrain | 5-8 | 16 KB | 1.5 KB | 10x |
| Complex biome | 9-16 | 16 KB | 2 KB | 8x |
| Highly varied | 17+ | 16 KB | 16 KB | 1x (skip palette) |

But palette compression is not just a memory win. **Operating directly on compressed data** (TurboQuant-style) yields equally large compute wins — uniform sections skip meshing entirely, palette-category bitfields cull faces without per-voxel decode, and LOD reduction reads palette frequencies instead of scanning voxels.

Phase 15 ships the storage format. The compute wins follow.

---

## 2. Prior Art (Verified)

### 2.1 Minecraft 1.18+ PalettedContainer
The gold-standard reference for in-RAM section compression. Three strategies swap in/out dynamically as the section's unique count changes:

- **SingularPalette (0 bpe):** palette has one entry, no data array. Activates for uniform sections (all air, all stone).
- **ArrayPalette (4-bit floor):** linear-scan palette, 2-16 entries. Minimum 4 bits per entry even when 2-3 entries would fit in 1-2 bits — aligns with long-packed reads.
- **BiMapPalette (5-8 bpe):** hash-map palette, 17-256 entries.
- **IdListPalette (Direct, ≥9 bpe):** palette dropped, indices become raw global BlockState IDs.

**Bit packing:** **padded-aligned, not tightly packed.** Each index fits wholly within one 64-bit long; if the next index would overflow, the remaining high bits are padded to 0 and the next index starts at bit 0 of the next long. Trades ~5% storage waste for single-load GPU reads (no multi-word assembly).

**Promotion logic:** when a new block type is added, if it doesn't fit in the current palette, the section is rebuilt with the next-larger strategy. Demotion (shrinking palette) happens opportunistically on re-save or chunk unload.

Sources: `minecraft.wiki/w/Chunk_format`, yarn-mappings `PalettedContainer` (1.18-1.21).

### 2.2 Litematica Schematic Format
**Tightly-packed** long-array bit storage — indices can span long boundaries. Denser (no padding waste) but slower to random-access (multi-long read, shift, OR).
Good for offline format. **Wrong for GPU runtime.** We reject this.

Sources: Litemapy docs, Lite2Edit.

### 2.3 Dado et al. (CGF 2016) — "Geometry and Attribute Compression for Voxel Scenes"
Decouples geometry (DAG) from attributes (palette + Huffman-shaped Wavelet Trees). Variable-bit random access, explicitly "well-suited for GPU architectures." This is the academic reference for GPU-resident palette storage with heterogeneous bit widths.

Not adopting the Wavelet Trees part — Minecraft's padded-aligned long scheme is simpler and fast enough. Citing Dado for completeness and because it validates the palette approach from a rendering-research angle.

### 2.4 Molenaar (CGF 2023) — "Editing Compressed High-resolution Voxel Scenes with Attributes"
Direct GPU edit operations on palette-compressed data. Prior art for our "modify a block without fully decompressing the section" hot path.

### 2.5 Aokana (arXiv 2505.02017, 2025)
GPU-driven voxel rendering framework; uses palette + bit-buffer for attribute compression. Recent validation of the same design choices.

### 2.6 Sparse Voxel Octrees (SVO) / DAGs / Brickmaps — CONSIDERED, REJECTED for Phase 15
- SVO/DAG: optimized for static raytracing, weak for frequent edits and rasterization meshing.
- Brickmap (GigaVoxels, Crassin INRIA 2009): octree-of-bricks with LRU GPU cache. Great for macro-scale static worlds but overkill for per-section compression.
- Sparse 64-trees (dubiousconst282, 2024): ~0.19 bytes/voxel theoretical. Research-grade, not production.

**Reserve these for the macro tier** (super-section LOD, world cache, long-range streaming). Per-section is palette territory.

### 2.7 RLE
Wins on axis-coherent content (flat plains). Loses random access (O(log n) to find voxel[i]), which kills meshing kernels that need per-voxel indexed reads. Viable as a **secondary** compression atop palette (compress the palette-index array with RLE for disk storage), but not as primary runtime format. Open question — see §8.

---

## 3. Current VoxelEngine State (Survey)

| Component | Current Format | Observation |
|---|---|---|
| `PackedBlock.cs` | 16 bits (12 type + 4 damage), stored as `int` or `short` | Damage is almost always 0. Compressing type alone is simpler. |
| `ChunkManager.ManagedSection` | `int[] BlockData` (4096 ints = 16 KB per section) | Consumer of the format — needs to switch to `CompressedSection`. |
| `IChunkCacheProvider` | `Task<int[]?> LoadSectionAsync`, `Task SaveSectionAsync(coord, int[])` | **Boundary for compression.** Either the provider ships compressed bytes, or the engine compresses between provider and storage. |
| `BlockRegistry` | 32-byte BlockProperties per type, GPU storage buffer | Source for palette-category classification (opaque/transparent/plant/liquid → bitfield). |
| `VoxelMeshPipeline`, `GreedyMergeKernels` | Consume `ArrayView<int>` with raw PackedBlock | Kernels need a decode path. Either decode-per-read in the kernel or pre-decode before dispatch. |
| `_config.MaxLoadedSections = 4096`, `MaxTotalQuads = 1_000_000` | Current defaults | At 16 KB per section × 4096 = 64 MB block data alone. Palette brings this to ~4-8 MB for natural terrain. |

No palette compression code exists today. This is a greenfield Phase 15 addition.

---

## 4. Proposed Architecture

### 4.1 Storage Format

**Per-section compressed layout:**

```
Section header (8 bytes):
  [0]    strategy tag (u8)         0=Singular, 1=Array, 2=BiMap, 3=Direct
  [1]    bits per entry (u8)       0, 1, 2, 3, 4, 5, 6, 7, 8, or 12
  [2:3]  palette size (u16)        1, 2..16, 17..256, or 0 (Direct mode)
  [4:7]  flags (u32)               bit 0 = all-air, bit 1 = has-transparent,
                                    bit 2 = has-translucent, bit 3 = has-plant,
                                    bit 4 = has-liquid, bit 5 = has-emissive,
                                    bit 6 = single-type (shortcut),
                                    bits 7+ reserved

Palette (variable):
  Singular:  omitted (strategy implies type 0 for all-air, or the one type if flagged)
  Array:     2-16 * u16 (palette entries, each a 12-bit block type zero-extended)
  BiMap:     17-256 * u16 (palette entries)
  Direct:    omitted (indices are raw PackedBlock types)

Data (variable, padded-aligned long-packed):
  Singular:  omitted (4096 voxels all = palette[0])
  Array/BiMap/Direct:  4096 * bpe bits, packed padded-aligned into u64[]
```

**Bits-per-entry table (matches Minecraft 1.18+):**

| Palette Size | BPE | Notes |
|---|---|---|
| 1 | 0 | SingularPalette - no data array |
| 2-16 | 4 | ArrayPalette - 4-bit floor (single-long aligned) |
| 17-32 | 5 | BiMapPalette |
| 33-64 | 6 | BiMapPalette |
| 65-128 | 7 | BiMapPalette |
| 129-256 | 8 | BiMapPalette - last power-of-2 aligned |
| 257-4095 | 12 | DirectPalette - raw PackedBlock type, matches current format |

**Padded-aligned packing math:**
- 64-bit long, packed at `bpe` bits per entry
- Entries per long = `64 / bpe` (integer division, drops remainder)
- Longs required = `ceil(4096 / entriesPerLong)`
- Read: `(longs[idx / epl] >> ((idx % epl) * bpe)) & mask` - single load, two shifts, one AND

**Size compared to raw (16 KB = 4096 × 4 bytes as int; 8 KB if short):**

| Strategy | BPE | Data Bytes | + Palette | + Header | Total | vs 16 KB raw |
|---|---|---|---|---|---|---|
| Singular | 0 | 0 | 2 | 8 | 10 | **1600x** |
| Array (4 entries) | 4 | 2048 | 8 | 8 | 2064 | **7.75x** |
| Array (16 entries) | 4 | 2048 | 32 | 8 | 2088 | **7.66x** |
| BiMap (64) | 6 | 3072 | 128 | 8 | 3208 | **4.98x** |
| BiMap (256) | 8 | 4096 | 512 | 8 | 4616 | **3.47x** |
| Direct | 12 | 6144 | 0 | 8 | 6152 | **2.60x** |

Direct mode still beats raw 16 KB by 2.6x because we drop the 4-bit damage channel from the main stream (see §4.4). Raw is a 1x fallback only reached if we also store damage in-line.

### 4.2 Damage Handling

Damage is 4 bits per voxel. In the current format it rides alongside the 12-bit type in PackedBlock. In the palette scheme, keeping damage in palette entries would multiply palette size by up to 16x — defeating the point.

**Recommendation: separate parallel damage array.**

- 4096 voxels × 4 bits = 2048 bytes per section
- Stored RLE-encoded when damage is all-zero (which is 99%+ of sections) — degenerates to 1 byte for "all pristine"
- Decoded on the fly during meshing, or ignored if `DamageOverlay` is not in use

This is cheap and keeps the type-palette clean.

### 4.3 Light Handling (Phase 14 Interaction)

Per Phase 14 research, light data is 16 bits per voxel (4 sky + 4 block + 8 reserved for colored upgrade). Light does **not** compress well via palette — it's a continuous gradient, not a discrete set. But it's also not part of the block data, just parallel to it.

**Store light as a separate parallel `ushort[]` array per section, uncompressed.** 8 KB per section. At 4096 loaded sections, 32 MB light RAM. Acceptable.

Alternative (future optimization): RLE the light array for disk storage. Huge wins for underground sections where sky=0 and block=0 uniformly. Not urgent.

### 4.4 Section Flags (Operate-on-Compressed Metadata)

The flags field in the header is the key to compute wins:

- `all-air` → meshing emits 0 quads, skip entirely. Face-mask bitmap for neighbor visibility is trivially empty.
- `single-type` (but not air) → meshing emits exactly 6 merged quads (one per face) without running greedy merge. Face visibility determined by neighbor sections' palette classes.
- `has-transparent`, `has-translucent`, `has-plant`, `has-liquid` → drives which meshing passes are needed (TransparencyMesher, CrossQuadMesher) without scanning the palette.
- `has-emissive` → light propagation kernel skips sections with no emitters.

**Palette-category lookup table:** at palette-build time, derive a 4-byte-per-palette-entry class vector from BlockRegistry (opaque/transparent/translucent/plant/liquid/emissive bits). Kernels index by palette-local index for category decisions. Face-culling compares `category[thisIdx]` vs `category[neighborIdx]` instead of decoding full BlockProperties.

### 4.5 GPU-Parallel Palette Construction

When a section is generated or loaded as raw block data, we compress in-memory:

1. **Count unique types** — ILGPU kernel, shared-memory histogram per workgroup (each voxel atomically increments `palette_hist[type]`), then across-workgroup reduction. Result: sorted list of unique types. Use `RadixSort` for the sort pass (existing ILGPU Algorithms library).
2. **Build palette** — stream-compact the unique types (existing ILGPU `Scan` primitive). Store as `ushort[paletteSize]`.
3. **Build index map** — `indexMap[type] = paletteLocalIdx`. Temporary 4096-entry lookup.
4. **Pack indices** — ILGPU kernel, one thread per voxel, reads `indexMap[type[voxel]]`, packs bits into padded-aligned `u64[]`.

All GPU-side, no CPU readback. Output: compressed section in GPU buffer, ready for meshing.

### 4.6 GPU-Parallel Decode

For meshing, we want the meshing kernel to read compressed data directly. Two implementation paths:

- **(A) In-kernel decode.** Kernels accept `CompressedSection` (palette ArrayView + data ArrayView + header), read-per-voxel with a 5-line unpack helper. SWAR-friendly. Adds ~3-4 cycles per voxel read; probably negligible against meshing's other costs.
- **(B) Pre-dispatch decode.** Run a decode kernel that fills a scratch `int[]` with raw PackedBlock, then dispatch existing meshing kernels unchanged. Uses more VRAM (full 16 KB per actively-meshing section) but keeps existing kernels simple.

**Recommendation: start with (B) for compatibility, migrate to (A) for hot-path kernels** (greedy merge, face culling) where memory bandwidth dominates. Non-hot paths (explosion, structural integrity queries) can stay on decode-to-scratch.

### 4.7 Persistence (OPFS Region Files)

`IChunkCacheProvider.LoadSectionAsync` currently returns `Task<int[]?>`. Phase 15 changes this to `Task<CompressedSection?>`:

```csharp
public struct CompressedSection
{
    public SectionHeader Header;      // strategy, bpe, palette size, flags
    public ushort[]? Palette;         // null for Singular/Direct
    public ulong[]? Data;             // null for Singular
    public byte[]? DamageRle;         // null when all-zero
    public ushort[]? Light;           // null for "regenerate on load" case
}
```

Callers wanting raw `int[]` (legacy, tests, debug tools) call `.Decompress()` to get a 4096-entry array.

OPFS region files store the compressed byte blob directly. AubsCraft's existing region file benchmarks (310 MB/s read, 118 MB/s write) still apply but with smaller payloads — a region file of 1024 sections goes from 16 MB uncompressed to ~1-4 MB compressed. That's 4-16x less I/O per region read.

### 4.8 Block Modification on Compressed Data

When `Lost Spawns` / `AubsCraft` modifies a block:

1. Look up `type[newBlock]` in the section's palette.
2. If present, re-pack the single affected voxel's index. One long load, mask, OR, store. O(1).
3. If absent, **promote strategy** or grow palette:
   - Array → Array with larger palette (recompact if still ≤16 entries)
   - Array → BiMap (4 bpe → 5+ bpe, re-pack all 4096 entries, one-time cost)
   - BiMap → Direct (drop palette, re-pack at 12 bpe)
4. After promotion, write the new voxel at its palette-local index.

**Cost:** O(1) for non-promoting edits, O(4096) for promoting edits. Promotions are rare in gameplay (adding a new block type to a section), so amortized cost is tiny.

**Demotion** (shrinking palette when a block type becomes unused) happens opportunistically on chunk save or world tick, not on every edit. Matches Minecraft's approach.

### 4.9 PackedQuad Interaction

PackedQuad stores `blockType` at bits [23:34] (12 bits). This is the **post-decode global block type**, not a palette-local index — meshing runs on decoded type IDs for cross-section consistency. No change needed to PackedQuad format for Phase 15.

(Phase 14's 16-bit light field at [47:62] also fits regardless.)

---

## 5. Testing Strategy

Data's plan calls for 2 tests; I propose expanding to cover correctness at each strategy boundary:

1. **Unit: roundtrip uniform** — build Singular section from all-air input, decompress, verify 4096 zeros.
2. **Unit: roundtrip mixed** — 4-type section (air, stone, dirt, grass), compress to Array, decompress, byte-identical to input.
3. **Unit: roundtrip strategy boundaries** — build sections with 1, 2, 15, 16, 17, 64, 256, 257 unique types. Verify each chose the correct strategy and roundtrip is lossless.
4. **Unit: promotion** — Start with 2-type Array section, add 15 unique types one at a time, verify promotion to Array(16) → BiMap(32) → BiMap(256) → Direct at correct thresholds.
5. **Unit: demotion** — fill Array(16) section, remove 14 types (replace with air), trigger demote-on-save, verify strategy drops back to Array(2).
6. **Unit: padded-aligned pack math** — known input bytes, hand-compute expected long layout, verify kernel produces match.
7. **Unit: GPU decode matches CPU reference** — generate compressed sections on GPU, decode on GPU, CPU-decompress the same input, pixel-wise equal.
8. **Unit: size reduction** — assert 1-type section compresses to ≤ 16 bytes, 4-type ≤ 2.1 KB, 16-type ≤ 2.1 KB.
9. **Unit: face culling on palette categories** — build section with known palette, verify face culler identifies visible faces without decoding full block data (hit `category[]` lookup only).
10. **Unit: uniform section shortcut** — mesh `all-stone` section, verify exactly 6 merged quads emitted, verify greedy merge kernel NOT invoked (instrument counter).

All use existing test infrastructure, `GpuTestVerify` for GPU-side assertions, no large CPU readbacks.

---

## 6. Phased Rollout

### Phase 15.1 — In-Memory Palette (ship first)
- `CompressedSection` type, three strategies (Singular, Array, BiMap — skip Direct initially, fall back to `int[]` for >256 types)
- Padded-aligned long packing
- `ChunkManager` stores compressed; decodes on demand for meshing (option B, pre-dispatch decode)
- Tests 1-8
- Target: 8-10x RAM reduction for typical natural terrain

### Phase 15.2 — OPFS Region File Integration
- Change `IChunkCacheProvider` to compressed-section roundtrip
- Region files store compressed bytes
- Measure I/O savings on AubsCraft (benchmark vs current)
- Target: 4-16x less disk I/O per region read

### Phase 15.3 — Operate-on-Compressed Fast Paths
- Section flags drive kernel selection (empty / uniform / has-transparent shortcuts)
- In-kernel decode for greedy merge and face culling (option A)
- Palette-category face culling
- Uniform section emits 6 quads without greedy merge
- Tests 9-10
- Target: **50%+ meshing throughput improvement** for typical natural-terrain workloads

### Phase 15.4 — Damage Separation + Light Separation
- Damage moved to parallel array (RLE for all-zero common case)
- Light array stays separate, uncompressed, sized per Phase 14 layout
- Direct strategy drops damage from data stream → 2.6x compression even worst case

### Phase 15.5 — Direct Strategy (optional)
- Only needed if we see sections with 257+ unique types in practice. May never trigger for Lost Spawns/AubsCraft.

---

## 7. Risk Analysis

| Risk | Mitigation |
|---|---|
| Decode overhead exceeds memory-bandwidth savings for small sections | Measure. Pre-dispatch decode keeps existing kernels unchanged until measurements justify option A. |
| Promotion storm during terrain generation (adding types one at a time triggers re-packs) | Generation path builds palette once at end (offline compress), not incrementally. Only runtime edits promote. |
| Palette grows unbounded if random blocks placed (AubsCraft with server-driven block variety) | Direct strategy at 257+ is the safety valve. Measure palette sizes in AubsCraft's loaded world. |
| IChunkCacheProvider breaking change impacts AubsCraft + Lost Spawns simultaneously | Phase 15.1 keeps `int[]` path; 15.2 adds `CompressedSection` overload; migrate consumers gradually. |
| Regression in existing tests from kernel format changes | Pre-dispatch decode (option B) keeps all kernels running on raw `int[]`. Only hot-path kernels migrate in 15.3. |

---

## 8. Open Questions for Captain

1. **Should we support the Direct strategy at all?** Minecraft blocks can easily exceed 256 (with BlockState permutations). AubsCraft mirrors Minecraft, so yes. Lost Spawns uses a custom block set likely ≤ 200 types. **Recommendation: implement Direct anyway for AubsCraft forward-compat.**

2. **Damage channel: in-line vs parallel array?** Parallel array (my proposal) is cleaner but adds 2 KB per section. In-line means palette multiplies by 16 when any damaged block exists in the section. **Recommendation: parallel array, RLE for all-zero sections.**

3. **Light channel: palette-compressed or uncompressed parallel?** Light gradients are continuous; palette compression would bloat (each unique light level = palette entry). **Recommendation: uncompressed parallel `ushort[]` per section.** Revisit after Phase 14 measurements.

4. **Strategy demotion timing:** Minecraft demotes opportunistically (chunk save/unload). Should we demote more aggressively (e.g., periodic compact pass)? **Recommendation: opportunistic only for v1.0. Revisit if we see RAM pressure.**

5. **RLE layer on top of palette?** RLE-compressing the packed data array could win further on axis-coherent content (underground uniform stone bands). Adds decode complexity. **Recommendation: defer. Palette alone achieves the Phase 15 targets.**

6. **GPU palette construction performance.** Constructing the palette on GPU (histogram + scan + pack) adds ~3 dispatches per section. For newly-generated sections this is fine — generation is already multi-pass. For loaded sections the palette is already built on disk. Measure generation-side cost before committing to the full 4.5 pipeline.

7. **Block type identity across AubsCraft ↔ Lost Spawns.** Currently `PackedBlock` types are app-defined. Palette classes from BlockRegistry are app-defined. If we ship v1.0 with any assumption about type 0 = air, type 1 = stone, etc., both apps must honor it. **Recommendation: keep type 0 = air as universal. All other IDs app-defined. Palette category flags live in the BlockRegistry and travel with the app, not the engine.**

---

## 9. Summary

- **Reject:** Litematica tight packing (slow GPU reads), SVO/DAG/brickmap for 16³ sections (over-engineered), RLE as primary (no random access).
- **Accept:** Minecraft-style padded-aligned long-packed palette with three strategies (Singular/Array/BiMap, optional Direct). Damage to parallel array. Light stays separate.
- **Ship order:** 15.1 in-memory → 15.2 OPFS integration → 15.3 operate-on-compressed fast paths → 15.4 damage/light separation → 15.5 Direct fallback.
- **Biggest wins:** 8-10x RAM for natural terrain, 4-16x less disk I/O, 50%+ meshing throughput on uniform/simple-surface sections via operate-on-compressed shortcuts.
- **Risk profile:** low. Every hot path has a decode-to-scratch fallback (option B) that keeps existing kernels unchanged. Migration to in-kernel decode (option A) happens only where measurements justify it.
- **Verified facts only.** Every architecture claim ties back to Section 2. Where prior art is academic (Dado, Molenaar, Aokana) I cite it to show the approach is validated; we don't implement their specific Wavelet Tree or DAG machinery.

Ready for Data's review.

-- Tuvok 🖖
