# Review: PLAN-Complete-Implementation.md

**Reviewer:** Geordi (Claude CLI, AubsCraft renderer)
**Date:** 2026-04-14
**For:** Data, Captain, Tuvok

---

## Overall Assessment

This is excellent work, Data. The research is thorough, the sprint ordering is logical, and the technical decisions are well-reasoned. The architecture of "fix once, both projects benefit" is exactly right. I have specific concerns below - none are "this is wrong," they're all "this will bite us if we don't address it before implementation."

---

## Critical Issues (Must Fix Before Sprint 1)

### 1. PackedQuad Y-position only encodes 0-63 - AubsCraft uses 384 height

PackedQuad.cs allocates 6 bits for Y position (bits [6:11]), giving range 0-63. AubsCraft chunks are 384 blocks tall (Y range -64 to 320). Lost Spawns will likely have similar or larger height. The current encoding silently truncates Y positions above 63.

**Options:**
- A) Operate on 16x16x16 sections (not full columns) - Y only needs 0-15 per section, 4 bits. This is what I just built in AubsCraft's section-based refactor today. Frees 2 bits.
- B) Expand Y to 9 bits (0-511) by stealing from reserved bits. Loses future flexibility.
- C) Make PackedQuad configurable per-consumer. Over-engineered.

**Recommendation:** Option A. It aligns with the cave culling architecture (Phase 4A graph visibility operates on 16x16x16 sections), matches what AubsCraft just shipped today, and the face mask uint64 columns already represent 64 Y positions - section-sized. The kernel dispatch becomes per-section, the quad Y is section-local (0-15), and the consuming project maps section (cx, sy, cz) to world coordinates. This is the natural unit for everything in the plan.

**What AubsCraft already has (shipped today, commit fb9d0fd):**
- `MapRenderService._slots` keyed by `(int cx, int sy, int cz)` where sy = 0-23
- AABB per section: `minY = sy * 16 - 64`, `maxY = minY + 16` (tight 16x16x16 box)
- `MeshGenerationResult.SplitIntoSections()` bins vertices into 24 buckets by Y position
- Draw loop iterates sections with tight frustum tests
- Eviction, compaction, water sorting all operate on section keys
- This is the integration surface VoxelEngine should target. The library produces per-section PackedQuad arrays, AubsCraft uploads each to its section slot.

### 2. DefaultChunkHeight = 256 but AubsCraft is 384, face masks use uint64 (64 bits max)

VoxelMeshConstants.DefaultChunkHeight is 256 but AubsCraft sends 384. The face mask columns use `long` (64 bits), so the maximum height representable per column is 64. This means either:
- Chunks must be processed in vertical sections (64 tall max per face mask column), or
- Face masks need to be wider (ArrayView<long> becomes multiple longs per column)

This isn't really a bug - the binary greedy meshing naturally operates on 16x16x64 slices (one uint64 per column). But the plan doesn't explicitly state that a 384-tall chunk requires 6 vertical passes (384/64) or 24 section passes (384/16). This needs to be explicit so nobody builds assuming one kernel dispatch = one full column.

**Recommendation:** Make the section-based dispatch the documented contract. One kernel dispatch = one 16x16x16 section. 24 dispatches per AubsCraft column. The face mask uint64 has 48 unused bits per section (only 16 of 64 bits used), but that's fine - the alternative (packing multiple sections into one mask) adds complexity for no real gain.

### 3. The i64 race condition fix (1A) needs more thought

The proposed fix - split `long[]` into `int[]` pairs and use `Atomic.And` - is correct for the mask clearing, but the greedy merge kernel READS face masks non-atomically in multiple places:

- Line 104: `long mask = faceMasks[faceOffset + innerIdx];` (read)
- Line 122: `long nextMask = faceMasks[faceOffset + nextInnerIdx];` (read)
- Line 152: `faceMasks[clearIdx] &= ~(1L << yLayer);` (read-modify-write - this is the race)
- Line 243: `faceMasks[faceOffset + innerIdx] = mask;` (write - also a race)
- Line 262: `long nextMask = faceMasks[faceOffset + nextInnerIdx];` (read)
- Line 282: `faceMasks[faceOffset + nextInnerIdx] = nextMask;` (write - also a race)

The current kernel dispatches as `[chunkXZ, 6]` - one thread per layer per face. For +Y/-Y faces, different Y layers access the SAME face mask column (each thread checks a different bit in the same long). So thread for Y=5 and thread for Y=12 both read/write `faceMasks[faceOffset + innerIdx]` - that's a data race even with i32 atomics on the clear, because the read + extend-width + write-back at lines 243/282 is a non-atomic read-modify-write sequence.

**The real fix:** For +Y/-Y faces, there is NO parallelism across Y layers within the same column - each layer's greedy merge depends on the mask state left by previous layers (the extend-width loop at line 247-283 reads neighbor columns, checks bits, clears them, and writes back). Two Y-layer threads touching the same XZ column would corrupt each other's mask state.

**Concrete solution - split the kernel by face type:**

```
// Safe: one thread per layer along the normal axis. Each thread owns disjoint columns.
// +X/-X: dispatch [chunkXZ, 2] - layer = X position, threads access different X slices
// +Z/-Z: dispatch [chunkXZ, 2] - layer = Z position, threads access different Z slices

// Must be sequential per XZ column. Dispatch [chunkXZ * chunkXZ, 2] 
// +Y/-Y: one thread per (x,z) column. Thread iterates ALL Y layers sequentially.
// This is already how MergeXZPlane works (line 98: for z, while x) but the 
// DISPATCH gives each Y layer its own thread via index.X = layer = yLayer.
// Fix: for +Y/-Y, dispatch index.X as the XZ column index, not the Y layer.
```

For +Y/-Y: change dispatch from `[height, 1]` to `[chunkXZ * chunkXZ, 1]`. Each thread iterates Y=0..height-1 internally. The i64 issue goes away naturally because a single thread owns its column - no atomics needed for mask clearing within the same thread's column.

For +X/-X/+Z/-Z: the current dispatch `[chunkXZ, 1]` is safe - each thread owns a different layer along the normal axis, accessing disjoint sets of columns. The only shared state is the atomic quad counter, which is already correct.

The i32 split (FaceMaskSplitter.cs) is still needed for WebGPU i64 emulation - but only for the inter-column reads during extend-width, where thread T for column (x,z) reads neighbor column (x+w,z). With the per-column dispatch fix, these reads are safe because they're read-only peeks at a neighbor that no other thread is currently modifying (the neighbor's thread processes its own layers independently).

**Recommendation:** Redesign 1A as two changes: (a) Fix the dispatch to eliminate the Y-layer race, (b) Split face masks to i32 pairs for WebGPU compatibility. Both are needed. Neither alone is sufficient.

---

## Important Concerns (Address During Implementation)

### 4. Sprint ordering: Rendering (Sprint 3) before Culling (Sprint 5) is risky

The vertex pulling pipeline (3B) emits vertices from PackedQuads in WGSL. The culling pipeline (5C) produces a filtered set of visible chunks. If the rendering pipeline is built first without the culling output format in mind, the integration at Sprint 5 may require reworking the draw pipeline.

**Recommendation:** Define the `CullResult` struct and indirect draw command format in Sprint 2 (when building IndirectDrawBuffer). Rendering and culling both consume/produce these - agree on the contract early. No code rework needed later.

**Proposed contract (for team discussion):**
```csharp
// Input to culling pipeline - one entry per loaded section
struct SectionEntry {
    int Cx, Sy, Cz;           // section coordinates
    int QuadOffset;            // offset into PackedQuad buffer
    int QuadCount;             // number of quads in this section
    ulong Connectivity;        // 48-bit face-pair visibility (Phase 4A)
}

// Output from culling pipeline - compacted list of visible sections
struct CullResult {
    ArrayView<int> VisibleIndices;  // indices into SectionEntry array
    int VisibleCount;               // from atomic counter
    // Face mask per visible section (Phase 1C) can be a separate ArrayView<byte>
}

// Indirect draw commands - one per visible section (or per face group if face masking)
// Layout matches GPUDrawIndirectParameters
struct DrawCommand {
    uint VertexCount;    // quadCount * 6
    uint InstanceCount;  // 1
    uint FirstVertex;    // quadOffset * 6
    uint FirstInstance;  // 0 (or section index for per-instance data)
}
```
This format works for both AubsCraft (blocky, 24 sections per column) and Lost Spawns (HD, same section grid). The indirect draw buffer holds all DrawCommands contiguously - one GPUBuffer for all visible draws.

### 5. Face masking (1C) savings estimate is optimistic for the general case

"dot(faceNormal, cameraToChunk) < 0 -> skip group. 36% reduction." - the 36% number assumes evenly distributed faces. In practice, for Minecraft terrain viewed from above (the most common view), +Y faces dominate (every surface block has one). -Y faces are rare (only cave ceilings). Side faces are roughly equal. The actual distribution is heavily skewed, so face masking helps less for the common case and more for the edge case (underground).

This isn't wrong - face masking is still worth doing. But don't count on 36% as a baseline; 15-25% is more realistic for surface viewing. The big win is underground where it combines with graph visibility.

### 6. Hi-Z Occlusion (4B) may not be worth the complexity for WebGPU

Hi-Z requires reading back the depth buffer into a mip chain, then testing AABBs against it in a compute shader. On WebGPU, reading the depth attachment into a compute-readable texture requires a copy (depth textures can't be directly bound as storage). The mip chain generation is another N compute dispatches. For the savings (rejecting chunks behind mountains), graph-based visibility (4A) handles the same case more cheaply for voxel worlds specifically.

**Recommendation:** Make 4B optional/deferred. Graph visibility + frustum + fog handles 95% of cases for both AubsCraft and Lost Spawns. Hi-Z is the last 5% at high complexity cost. Build the pipeline orchestrator (4C) with a slot for Hi-Z but don't block v1.0 on it.

### 7. Geomorphing (5D) has no test criteria listed

Every other phase has specific test criteria. 5D just says "blend between LOD levels via per-chunk morph factor in vertex shader." Geomorphing is notoriously hard to test automatically - it's a visual smoothness thing. At minimum, test that morphFactor is 0.0 at LOD boundary near edge and 1.0 at far edge, and that vertex positions at morphFactor=0 match LOD N exactly and at morphFactor=1 match LOD N+1 exactly.

### 8. No explicit section/chunk coordinate system document

The plan references chunks, sections, layers, padded coordinates, inner coordinates, world coordinates - but there's no single source of truth for the coordinate system. AubsCraft uses: world Y = section_index * 16 - 64. The plan uses chunkXZ and height. When two consumers (AubsCraft, Lost Spawns) have different world coordinate offsets (-64 vs 0), the library needs a clear contract.

**Recommendation:** Add a `CoordinateConventions.md` in Research/ or a section in CLAUDE.md. Define: section-local (0-15), chunk-local (0-chunkXZ), world-space, padded (chunk-local + 1 border). Consumers map between world and chunk-local; the library only speaks chunk-local and section-local.

---

## Minor Notes

### 9. Streaming Upload Buffer (2C) - double-buffering may not be needed on WebGPU

WebGPU's `queue.writeBuffer` already handles synchronization internally - you can write to a buffer that's currently being read by a previous frame's commands. The driver handles the copy staging. Double-buffering is important on raw Vulkan/D3D12 but may be unnecessary overhead on WebGPU. Test before building.

### 10. Texture arrays vs atlas (Key Technical Decisions table)

"No mipmap bleeding, no per-tile padding, hardware tiling via fract()" - this is correct for Lost Spawns. For AubsCraft, the existing atlas (256x256, 16x16 grid) works fine and is already shipped. The library should support both - texture arrays as the preferred path, atlas as a compatibility option. Don't force AubsCraft to rewrite its texture pipeline on day 1 of migration.

### 11. "105/105 tests passing" in Context but git log shows WIP frustum tests

The Context section says "Two bugs remain (greedy merge i64 race on WebGPU, frustum test matrix math)" and also "105/105 tests passing." If the frustum tests are failing, the count should exclude them or note them as skipped. Honest accounting - the team needs to know the real number.

---

## What's Great

- **Bug-first sprint ordering.** Fix the races before building on top. Correct.
- **Triple verification pattern.** Naive reference + bitwise CPU reference + GPU kernel, all must match. This is the standard that caught 5 bugs in ILGPU v4.6.0.
- **Atomic counter rollback.** Already in the kernel code. Learned from AubsCraft's buffer overflow fix.
- **Research depth.** The masterplan doc maps every technique to specific ILGPU APIs. No handwaving.
- **DelegateSpecialization for LOD.** Using our own library features. Rule 4: Use Every Gear We Build.
- **The whole concept.** Extracting shared infrastructure is exactly what TJ's library-first philosophy demands. Both projects get better, bugs get fixed once.

---

## Summary for Captain

The plan is solid. Three things need resolution before anyone starts coding:

1. **PackedQuad height encoding** - decide on section-based (recommended) or expanded bits
2. **Greedy merge +Y/-Y race** - the i64 fix doesn't address the real data race between Y-layer threads
3. **CullResult format** - define early so rendering and culling don't diverge

Everything else is "address during implementation" level. Data did the research. The sprints are ordered correctly. Let's go.

-- Geordi
