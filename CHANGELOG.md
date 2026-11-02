# SpawnDev.VoxelEngine Changelog

## Unreleased - thin blocks, plants that stand in water

### Changed
- **Dependencies: SpawnDev.ILGPU 5.3.7 -> 5.4.14, SpawnDev.SpawnJS.Blazor 3.0.4 -> 3.0.7** (built as 0.2.0-rc.5-local.20). ILGPU 5.4.x brings O2 as the default optimization level, Velocity as the desktop CPU backend and the WebGPU loop / sub-view / sub-word fixes of 5.4.1-5.4.14.

### Added
- **`LightPropagator`: sky light and block light for a world that has none of its own, on the GPU.** `PropagateColumnAsync(volume, slot, column, negX, posX, negZ, posZ)` takes the column buffer and its four neighbours (null = not loaded: open sky) and writes the column's light bytes into a `LightVolume` slot without leaving the GPU (init per column, 15 relaxation steps over the 3 x 3 columns round it, pack, `LightVolume.CopyIn`). `SetBlockLight(passesLight, emission)`: per block type whether light passes and how much it gives off (0..15). Sky light enters at the top and falls straight down without loss; every other step costs one level. The corner columns are not taken. It returns `LightBorders` (which neighbours' cells within reach changed, in blocks or light: fingerprints of the 15-deep strips on each side, compared with the last time) so an edit can light the neighbours that need it and stop where nothing changed; `Forget(volume, slot)` for a column that is gone. Test (WebGPU): the light of a 4-section column with a roof, caves, a torch, lava and a missing neighbour is byte-for-byte what a flood fill written the other way round (nearest first, by level) gives; an edit beside +X reports +X; the same column again reports nothing.
- **`BlockRenderPipeline` render bundles: `BeginBundle` / `BindBundle` / `DrawInBundle` / `ExecuteBundle`, `FrameBindGroupVersion`.** Record the draws of a frame (pipeline, bind groups, draws) into a `GPURenderBundleEncoder` once and run it with one call in each pass: a draw costs a call into the browser (about 2 us on a desktop, several times that on a headset's CPU, hundreds of columns twice a frame), so a headset records once for both eyes and again only when the set of draws changes. The frame's uniforms are read when the bundle runs, so write the next eye's matrices (`BeginFrame`) between passes that are submitted one after the other. `ExecuteBundle` clears the pass's remembered pipeline (the browser clears it), so a `BindPass` after it sets it again. `FrameBindGroupVersion` changes when the frame bind group is made again (textures, light, biomes set): a bundle recorded before is stale. The wrapper's `GPURenderBundleEncoderDescriptor` has no colour formats, so a derived descriptor adds them. Test (WebGPU): a bundle with two draws run in two passes with different frame matrices draws the same pixels as the draws in the pass, and the two eyes differ.
- **`ColumnMesher.AmbientOcclusion`: per-vertex ambient occlusion baked into the quads (default off).** Every visible face of the opaque pass gets the darkness of its four corners (2 bits each in `PackedQuad`'s AO field, corner = a * 2 + b in the quad's own axes): the two cells beside the corner and the one between them in the layer in front of the face, both sides solid being as dark as all three. A new stage (`AoKernel`, one thread per section column, after the face cull) writes it for the visible faces; the greedy merge adds it to the merge key and merges only along an axis the occlusion does not change along (so every cell keeps its own value at the quad's corners): a flat floor is still one quad, a wall's foot or a floor's edge is smaller ones (about 30% more quads on rough test terrain). A missing neighbour column is open (no dark seam at the edge of the loaded world); water, plants and shaped blocks carry none. The block shader reads the four corners in the vertex stage, flips the quad's diagonal to run between the more alike pair, and multiplies the light by the interpolated shade (1, 0.8, 0.6, 0.4): no extra pass, no storage read in the fragment stage. Tests (WebGPU): a rough three-section column with a pillar across a section boundary and two of four neighbours loaded - every unit face has its cell's occlusion by the neighbours, the face set is unchanged, no quad's occlusion changes along an axis it was merged along; a flat floor is at most two quads and the floor beside a block is occluded; a quad's corners shade 0.4 / 0.8 / open at the right pixels, and the flipped diagonal keeps two opposite dark corners apart.
- **`LightPropagator`: emitters and dimming blocks.** `PropagateColumnAsync(..., columnX, columnZ, emitters)` takes `LightEmitter(X, Y, Z, Level)` records (world blocks, level 0..15: campfire 14, torch 12; the ones inside the column's 3 x 3 region count, up to `MaxEmitters` = 1,024, the brightest first, two in one cell give the brighter) for lights that are not blocks. `SetBlockLight(passesLight, emission, dims)`: `dims` is the levels a block takes off what passes through it over the one level every step costs (leaves 1, murk 3); sky light falling straight down loses exactly the dimming, so the ground under a canopy is darker. Test (WebGPU): a column with a leaf canopy and murk, emitters in the column and in a neighbour, a duplicate cell, a far one, an out-of-world one and a dark one, byte-for-byte against the flood fill; the leaf under open sky has 14.
- **`LightVolume.Init(device, queue, slotCount, side, columnHeight = 384, minY = -64)`**: a world 256 high from y = 0 passes `256, 0` and writes `BytesPerColumn` (65,536) bytes a column (`ColumnHeight`, `MinY`, `MaxSlots(columnHeight)`: 1,950 at 256). The shader reads both from the volume's parameters (`tune.z`, `tune.w`), so one pipeline lights worlds of any of these heights. `ColumnBytes` stays the default height's 98,304. `LightVolume.CopyIn(slot, GPUBuffer)` puts a column's light from a GPU buffer into a slot; `Data` can be copied from (CopySrc). Test: dark layer 1 of a 256-high column lights the face above y = 0, layer 65 is another cell, a second slot starts 65,536 bytes on, a wrong-size write is refused.
- **`ColumnMesher.CreateColumnBuffer(ReadOnlySpan<byte>)` / `(ReadOnlySpan<ushort>)` and `WriteColumnBuffer(existing, ...)`**: a column buffer from one byte (or ushort) per block, x + z * 16 + y * 256, and an edited column re-uploaded in place (the buffer stays; mesh again after it). The bytes pass through the .NET heap: for streamed worlds keep `CreateColumnBufferFromJS`. Test: the byte column meshes like the managed one, a rewrite in place meshes as the other terrain, both overloads, wrong lengths refused.
- **`SkyPipeline`: the sky behind the world.** One full-screen triangle drawn first in the pass (no depth test or write): every pixel works out the direction it looks in (`SkyUniforms.InverseViewProjection`: any eye of a headset too) and paints a gradient from the horizon's colour to the zenith's, a glow round the sun, the sun and the moon (squares of the game's own textures, the moon in its phase 0..7), stars at night (hashed points on the sphere, turning with the sun), and a cloud layer at a height (one map pixel = 12 blocks, sliding along x, seen from below only; rain: clouds everywhere, no sun, moon or stars). The sun, moon and cloud images come from an atlas (`EntityTextureAtlas`) the caller makes; `SkyUniforms` says where each is. Tests (PMT, WebGPU, mutation-checked): the gradient and below the horizon, the sun at its place (centre and 30 degrees right), the moon opposite it in the asked phase, stars at night and not by day or in rain, clouds where the map says, 12 blocks a pixel, sliding, none from above, rain; drawn behind a block in front of it.
- **Biome tint: `BiomeTintMap`, `BlockMaterial.TintGrass / TintFoliage / TintWater`.** Grass, leaves and water take the colour of the biome they stand in, per PIXEL: three 256 x 256 maps (grass, foliage, water), one texel per 4 x 4 blocks, wrapping every 1,024 blocks, sampled with linear filtering so the border of two biomes blends as in the game and not in 4-block steps. A material says which map tints it (flag bits 4-5, together with `FlagTinted` or `TintFaces`); without a kind the tint stays the material's own colour (birch and spruce leaves, lily pads). `BiomeTintMap.SetColumn(cx, cz, grass, foliage, water)` writes a column's 4 x 4 cells (16 colours each); `BlockRenderPipeline.SetBiomeTints(map)` binds it (the pipeline starts with plains colours everywhere, so ground not heard of yet is still green). Bindings 7-10 of the frame group. Tests (PMT): plains by default for each kind, a column's own colours and each kind reading its own map, a border blending across cell centres, negative columns and the wrap, a fixed tint and an untinted block ignoring the map.
- **`ParticleRenderPipeline`: smoke, flames, sparks in one instanced draw.** An emitter is a block position and a kind (`ParticleEmitterInstance`, 16 bytes); a kind (`ParticleKind`, 112 bytes) says how many particles live at once (up to 16), how long, which sprite frames (animated by age), the velocity, gravity or buoyancy, scatter, wobble, size and opacity over the life, tint, and whether it glows. The vertex shader places every particle from the kind and the frame clock alone (the Time of `BlockFrameUniforms`): nothing is simulated, nothing is stored per particle, and the CPU's job is the list of emitters near the camera. Camera-facing quads whose right and up come from the view-projection itself (so any eye of a headset, off-axis frusta included, gets billboards that face it); sprites fetched exactly; alpha blended, depth tested, not written; lit like the day (a quarter at night) unless they glow, fogged like the blocks. Draw it inside the block pass, after the opaque pass and the creatures. Tests (PMT, WebGPU, each mutation-checked): a sprite faces the camera from three directions and stands upright; the count of dots equals the kind's count; position follows the clock; gravity and buoyancy; animation frames follow the age in order; alpha blend, night dimming and glow; depth against a terrain face in front and behind.
- **`BlockRenderPipeline.BeginRenderPass(..., clearAlpha)`**: a clear alpha below 1 clears to the premultiplied see-through colour (augmented reality: what is not drawn shows the camera feed). Default 1, nothing changes for callers that do not pass it.
- **Shaped blocks: stairs, slabs, fences, doors... (`ShapeTable`, `BlockClass.Shaped`, `BlockMaterial.Shape`)**. A block that is not a cube is up to thirty-two BOXES of its cell, each with six faces
  (an atlas tile, a window of the tile in sixteenths, a quarter-turn count, a tint flag). A block type gets a slot in a `ShapeTable` (1,024 slots) and says so in its class
  (`BlockClass.ShapeSlot(slot)`: emits no cube faces, hides nothing) and its material (`BlockMaterial.Shape(slot)`). `ColumnMesher.SetShapes(table)`: a new kernel (`ShapeKernel`, one thread per cell,
  right after the opaque pass) emits each box face that can be seen as a one-by-one quad flagged `PackedQuad.ShapedFlag` with its box number, into the opaque list - faces on the cell boundary against an
  occluding neighbour are left out; `BlockRenderPipeline.SetShapes(table)` binds the boxes (binding 6) and the vertex shader places each face by its box and takes tile, window and turn from the table.
  No new pass, no new draw. Tests (PMT, WebGPU): the table packs; the mesher emits 6 + 5 faces for a two-box stair, loses the two against a stone neighbour, the stone shows its face toward the stair;
  a slab is half-high with the lower window of its tile on its side and its own tile on top; a second slot and a second box land where they are; a turned face turns its tile clockwise (all four turns);
  side faces are upright and unmirrored from the south and the north; 7 mutations caught (turn, cull, window, orientation, slot, box index).
- **`BlockMaterial.Thickness(sixteenths)`** (flag bits 9..12): a thin block - a carpet, rail, pressure plate, snow layer, closed trapdoor - is the cube squashed to n/16 of its
  height from its floor, tops and sides, the sides showing the top strip of their texture rather than the whole tile squeezed. Pair it with a class that hides nothing
  (`BlockClass.Cutout`) so what is beside and below still shows. Test: the middle of a 4/16 block is empty, just above its floor is solid, the side shows the tile's top rows;
  three mutations (no squash, no side-strip UV) caught.
- **Water is drawn from both sides** (the water pass no longer culls back faces): a swimmer sees the underside of the surface. Test: a water face from in front and from behind; mutation caught.
- **`BlockMaterial.Slope(direction)`** (flag bits 13..15, `SlopeEast/West/South/North`): a thin block sheared into a ramp whose far end is a block higher - a sloping rail. Ramps side by side
  merge across the slope, never along it (the next block of a ramp is a layer up), so every quad is one block long. Test: level block vs ramps rising each way at both ends; 3 mutations caught.
- **`BlockMaterial.FlagPlantFromBottom`**: a plant's crossed planes take face 5's tile and tint instead of face 0's, so a block can be water on faces 0..4 and a plant on face 5 -
  seagrass and kelp, which stand in a cell of water. With class `BlockClass.Plant | BlockClass.Water` the mesher already draws the water surface and the plant together
  (new mesher test: the surface over a plant in the top layer is there; plants 4 quads). Test: the plant shows face 5's tile, cube faces keep theirs, no flag = face 0; mutations caught.

## 0.2.0-rc.4 (2026-10-15) - the ENTITY pipeline: creatures drawn as textured model boxes, in one instanced draw

### Added
- **`EntityRenderPipeline`** draws every part of every model of a world in ONE instanced draw (36 vertices an instance: the shader expands a box).
  `EntityPartInstance` (128 bytes: a model matrix, the box's origin/size in model pixels, an overlay inflation, a mirror flag, a texture origin, a tint) is the
  whole CPU-side cost of a part; `SetParts` uploads a frame's worth (grows the buffer, caps at 131,072 parts and says how many it dropped).
  Model space is Minecraft's (y down, 16 px a block); the box texture is laid out the way Minecraft lays a model's boxes out (top and bottom over the four
  sides), so a vanilla model's texture offsets work unchanged. Mirrored boxes (the left limbs reuse the right limbs' texture) swap the side regions and run u backwards.
  Overlay layers (a hat, a coat, wool) inflate the geometry, not the texture lookup. Alpha under 0.1 is discarded; a tint multiplies the texture.
- **`EntityTextureAtlas`**: one 2D texture of any mix of image sizes packed by the caller, read with exact texels in atlas pixels (no table of sizes on the
  GPU); `WriteRegion` rewrites a rectangle while the atlas is in use (a player's skin arriving).
- `EntityTextureAtlas.WriteCanvas(x, y, OffscreenCanvas)` draws a canvas into the atlas (GPU to GPU where the browser can): a skin decoded and composed in a canvas never
  touches the .NET heap. The atlas texture now carries `RenderAttachment` (copyExternalImageToTexture needs it). Test: a region rewritten while in use, a canvas that does not fit refused.
- `BlockRenderPipeline.FrameBuffer`: the frame uniforms other pipelines share. The entity pipeline reads the SAME buffer, so a creature is lit and fogged exactly
  like the ground it stands on. Draw it inside the block pass, after the opaque and plant passes and before the water.
- Tests (WebGPU pixel readback, 12): each face of a box shows its own region of the cross layout; a mirrored box swaps the side regions; every side reads left to
  right and upright seen from outside; the top runs its texture frontward; tint, transparency, inflation, depth order across instances, fog, face shading, buffer
  growth. Eleven mutations (a flipped u, a flipped top v, no mirror swap, no inflation, no tint, no discard, one instance, no depth write, no fog, a flipped back u, no face
  shading) were each caught.

## 0.2.0-rc.3 (2026-10-15) - a texture ATLAS (no layer limit), animated tiles, per-face tint

AubsCraft's textures move from a hand-picked 80 to every block texture of the game (1,400+) with every frame of every animated one: several
thousand tiles. A WebGPU texture ARRAY holds 256 layers unless the device asks for more, and the device is created by SpawnDev.ILGPU.

### Changed (breaking within the RC; BlockRenderPipeline is new in 0.2.0)
- **`BlockTextureAtlas` replaces `BlockTextureArray`.** One 2D texture, a grid of equal square tiles; a tile's number (row-major from the top-left) is
  what a `BlockMaterial` face names, as a layer number was. `BlockTextureAtlas.FromRgba(device, queue, rgba, width, height, tileSize)` (a JS typed array
  or a `byte[]`) uploads it as it is: no slicing, no copy into the .NET heap. The shader fetches texels exactly (`textureLoad`, no sampler) and
  wraps the coordinate inside the tile itself, so a greedy-merged quad still tiles its texture once per block; there are no seams and no layer limit
  (a 2D texture side of 8192 px holds 512 tiles of 16 px a side). Tile size comes from the atlas, not a constant.
- `BlockFrameUniforms` is 176 bytes: `Time` (seconds), `AtlasTilesPerRow` and `AtlasTileSize` (filled in by the pipeline from the atlas in use).
- Bindings of group 0: 3 = the atlas (2D), 4 = the animation table, 5 = the timelines (the sampler is gone).

### Added
- **Per-face tint and a darken flag in `BlockMaterial`.** `BlockMaterial.TintFaces(mask)` (bit f = face f; OR it into `Flags`) tints only the named faces: the top of a
  grass block takes the biome colour, its sides do not (`FlagTinted` still tints them all). `FlagDarken` draws every face at 0.7 of its colour: what lies
  under water, without faking it through the tint (which cannot differ per face). Tests: top tinted and side not, a mask that does not name the face,
  `FlagTinted` still tinting the side, darken alone and with a tint; 3 mutations (mask shift, no darken, mask ignored) caught.
- **Animated tiles.** `BlockRenderPipeline.SetAnimations(IReadOnlyList<TextureAnimation>)`: `TextureAnimation(BaseTile, int[] Timeline)` lists the tile to
  show for each TICK (20 a second) of its loop; wherever a face names `BaseTile` the vertex shader draws `Timeline[tick % Timeline.Length]`, with
  `tick = floor(BlockFrameUniforms.Time * 20)`. A frame that lasts several ticks, repeats or plays out of order is just a longer list, which covers
  Minecraft's `.mcmeta` animations (frametime, frame lists) without the shader knowing about them. Setting a new atlas drops the animations.
- Tests (WebGPU pixel readback): 512 tiles (twice what an array holds) all reachable; tile size from the atlas; an animated tile at ticks 0-4, 22 and
  20000 shows the right frame while a static tile beside it never changes; bad timelines are rejected, a new atlas starts clean. The existing 24
  render tests (upright unmirrored faces, merged-quad tiling, cutout, water, fog, section origins...) pass unchanged on the new shader. Seven
  mutations (no modulo, wrong tick rate, wrong row math, no wrap, no zero check, a fixed tile size, no duplicate check) were each caught.


## 0.2.0-rc.2 (2026-10-15) - quad buffers shared by many meshes

AubsCraft on an Intel UHD 620 laptop: 837 draws AND 837 buffer binds a frame (one quad buffer per column and pass), 17.6 ms of the 19.1 ms
frame spent recording them. After `EmbedSectionIndex` removed the per-section draws, the per-mesh buffer was the remaining multiplier.

### Added
- **`QuadArena`**: a few large GPU buffers (blocks of 2 Mi quads = 16 MiB by default) that hold the quad lists of many meshes.
  `ColumnMesher.Arena` / `HeightmapMesher.Arena`: the compact pass writes each list straight to its place in a block (no extra copy, no
  buffer created per mesh), and `Draw(pass, list.FirstQuad, list.Count, slot)` draws it. Hundreds of columns share one buffer, so a pass
  binds once instead of once per column. A block is added when nothing fits (a list longer than a block gets a block of its own);
  disposing a mesh gives its ranges back, a second Dispose is harmless. Null (the default): each list keeps its own buffer, nothing changes.
- **`ColumnMesh.QuadList`**: `FirstQuad`, `IsPooled`, `ArenaBlock`, and `CopyToHostAsync()` (exactly the list's quads, pooled or not).
- **`BlockRenderPipeline.QuadBufferBinds` / `PipelineSwitches`**: running totals of what `BindPass` actually issued (it skips repeats, so its call count is not the number of binds); a renderer subtracts two readings for a frame. Tested in `BlockRender_BindPass_SkipsRepeats_ButNeverSkipsAChangeTest` (5 calls = 4 buffer binds + 3 switches).
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
