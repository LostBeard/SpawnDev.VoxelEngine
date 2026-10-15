using SpawnDev.ILGPU.WebGPU;
using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // The block texture ATLAS (one 2D texture, exact texel fetches, tiling done in the shader) and ANIMATED tiles (a timeline of tiles,
    // 20 ticks a second). What the texture array could not do: WebGPU guarantees 256 array layers, a game's block textures with every
    // animation frame are thousands of tiles.
    public abstract partial class VoxelEngineTestBase
    {
        // A solid-colour tile atlas: tile k is (k & 255, (k >> 8) * 80 + 40, 60), tileSize px, tilesPerRow x rows tiles
        private static byte[] BaSolidAtlas(int tileSize, int tilesPerRow, int rows, Func<int, (byte r, byte g, byte b, byte a)> colourOf)
        {
            int w = tileSize * tilesPerRow, h = tileSize * rows;
            var px = new byte[w * h * 4];
            for (int ty = 0; ty < rows; ty++)
                for (int tx = 0; tx < tilesPerRow; tx++)
                {
                    var (r, g, b, a) = colourOf(ty * tilesPerRow + tx);
                    for (int y = 0; y < tileSize; y++)
                        for (int x = 0; x < tileSize; x++)
                        {
                            int i = ((ty * tileSize + y) * w + tx * tileSize + x) * 4;
                            px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = a;
                        }
                }
            return px;
        }

        /// <summary>
        /// Render +Y faces of the given block types (type i drawn at x = i - 1) seen from straight above, 16 px a block, once per time in
        /// <paramref name="times"/> with the same pipeline; returns the pixel at the centre of each block for each time.
        /// </summary>
        private static async Task<List<(byte r, byte g, byte b)[]>> BaRenderTops(GPUDevice device, GPUQueue queue, byte[] atlas, int atlasW, int atlasH,
            int tileSize, BlockMaterial[] materials, IReadOnlyList<TextureAnimation>? animations, float[] times, int blocks)
        {
            using var textures = BlockTextureAtlas.FromRgba(device, queue, atlas, atlasW, atlasH, tileSize);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            if (animations != null) pipeline.SetAnimations(animations);

            // 4 blocks wide at 16 px: 90 degrees at height 2 above their tops sees 4 blocks
            var eye = new Vector3(blocks / 2f, 3f, 0.5f);
            var viewProj = BrView(eye, new Vector3(blocks / 2f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90;
            var quads = Enumerable.Range(0, blocks).Select(i => PackedQuad.Pack(i, 0, 0, 1, 1, 4, i + 1)).ToArray();
            var buffer = CreateQuadBuffer(device, queue, quads);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            var results = new List<(byte r, byte g, byte b)[]>();
            try
            {
                foreach (float t in times)
                {
                    var frame = BrNeutral(viewProj, eye);
                    frame.Time = t;
                    pipeline.BeginFrame(frame);
                    using var encoder = device.CreateCommandEncoder();
                    using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                    {
                        pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, buffer);
                        pipeline.Draw(pass, 0, quads.Length, 0);
                        pass.End();
                    }
                    using var cmd = encoder.Finish();
                    queue.Submit(new[] { cmd });
                    await queue.OnSubmittedWorkDone();
                    var pixels = await ReadBackPixels(device, queue, colorTex, 64, 64);
                    // 16 px a block, the row of blocks centred on the image: block i is centred at x = 32 + (i + 0.5 - blocks / 2) * 16
                    results.Add(Enumerable.Range(0, blocks).Select(i =>
                    {
                        var (r, g, b, _) = GetPixel(pixels, 64, (int)(32 + (i + 0.5f - blocks / 2f) * 16), 32);
                        return (r, g, b);
                    }).ToArray());
                }
            }
            finally
            {
                pipeline.ReleaseQuadBuffer(buffer);
                buffer.Destroy(); buffer.Dispose();
                DisposeTargets(colorTex, depthTex, colorView, depthView);
            }
            return results;
        }

        private static void BaExpect((byte r, byte g, byte b) got, int r, int g, int b, string what)
        {
            if (Math.Abs(got.r - r) > 3 || Math.Abs(got.g - g) > 3 || Math.Abs(got.b - b) > 3)
                throw new Exception($"{what}: ({got.r},{got.g},{got.b}), expected ({r},{g},{b})");
        }

        [TestMethod]
        public async Task BlockRender_Atlas_HoldsFarMoreTilesThanAnArrayCould_AndEveryOneIsReachableTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // 32 x 16 = 512 tiles of 4 px: twice the 256 layers WebGPU guarantees. Tile k = (k & 255, (k >> 8) * 80 + 40, 60).
            var atlas = BaSolidAtlas(4, 32, 16, k => ((byte)(k & 255), (byte)((k >> 8) * 80 + 40), 60, 255));
            var tiles = new[] { 3, 255, 256, 511 };
            var materials = new BlockMaterial[tiles.Length + 1];
            for (int i = 0; i < tiles.Length; i++) materials[i + 1] = BlockMaterial.Uniform((ushort)tiles[i]);
            var got = (await BaRenderTops(device, queue, atlas, 128, 64, 4, materials, null, new[] { 0f }, tiles.Length))[0];
            for (int i = 0; i < tiles.Length; i++)
                BaExpect(got[i], tiles[i] & 255, (tiles[i] >> 8) * 80 + 40, 60, $"tile {tiles[i]}");
        });

        [TestMethod]
        public async Task BlockRender_Atlas_TileSizeComesFromTheAtlas_NotAConstantTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // A 2 x 2 atlas of 8 px tiles: tile 3 is the bottom right one
            var atlas = BaSolidAtlas(8, 2, 2, k => k switch { 0 => (255, 0, 0, 255), 1 => (0, 255, 0, 255), 2 => (0, 0, 255, 255), _ => (255, 255, 0, 255) });
            var materials = new[] { BlockMaterial.Missing, BlockMaterial.Uniform(3), BlockMaterial.Uniform(1), BlockMaterial.Uniform(2), BlockMaterial.Uniform(0) };
            var got = (await BaRenderTops(device, queue, atlas, 16, 16, 8, materials, null, new[] { 0f }, 4))[0];
            BaExpect(got[0], 255, 255, 0, "tile 3 (bottom right)");
            BaExpect(got[1], 0, 255, 0, "tile 1 (top right)");
            BaExpect(got[2], 0, 0, 255, "tile 2 (bottom left)");
            BaExpect(got[3], 255, 0, 0, "tile 0 (top left)");
        });

        /// <summary>One quad of the given face code for block type 1 at the origin, seen from the given side; the pixel at the centre of the image.</summary>
        private static async Task<(byte r, byte g, byte b)> BaRenderFace(GPUDevice device, GPUQueue queue, BlockMaterial material, int face)
        {
            var atlas = BaSolidAtlas(4, 1, 1, _ => (255, 255, 255, 255));
            using var textures = BlockTextureAtlas.FromRgba(device, queue, atlas, 4, 4, 4);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(new[] { BlockMaterial.Missing, material });
            // +Y seen from above, +Z seen from the front: a block at the origin, 1 wide
            var (eye, target, up) = face == 4
                ? (new Vector3(0.5f, 3f, 0.5f), new Vector3(0.5f, 0f, 0.5f), -Vector3.UnitZ)
                : (new Vector3(0.5f, 0.5f, 3f), new Vector3(0.5f, 0.5f, 0f), Vector3.UnitY);
            pipeline.BeginFrame(BrNeutral(BrView(eye, target, up) * BrProj90, eye));
            var quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, face, 1) };
            var buffer = CreateQuadBuffer(device, queue, quads);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, buffer);
                    pipeline.Draw(pass, 0, 1, 0);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var (r, g, b, _) = GetPixel(await ReadBackPixels(device, queue, colorTex, 64, 64), 64, 32, 32);
                return (r, g, b);
            }
            finally
            {
                pipeline.ReleaseQuadBuffer(buffer);
                buffer.Destroy(); buffer.Dispose();
                DisposeTargets(colorTex, depthTex, colorView, depthView);
            }
        }

        /// <summary>One quad (face code) of block type 1 over an atlas of the given tile colours (16 px tiles, one row); the pixel at (px, py) of a 64 px image seen from the front (or above for face 4).</summary>
        private static async Task<(byte r, byte g, byte b)> BaRenderPixel(GPUDevice device, GPUQueue queue, byte[] atlas, int tiles, BlockMaterial material, int face, int px, int py, int cellX = 0, int cellZ = 0)
        {
            using var textures = BlockTextureAtlas.FromRgba(device, queue, atlas, 16 * tiles, 16, 16);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(new[] { BlockMaterial.Missing, material });
            var shift = new Vector3(cellX, 0, cellZ);
            var (eye, target, up) = face == 4
                ? (new Vector3(0.5f, 3f, 0.5f) + shift, new Vector3(0.5f, 0f, 0.5f) + shift, -Vector3.UnitZ)
                : (new Vector3(0.5f, 0.5f, 3f) + shift, new Vector3(0.5f, 0.5f, 0f) + shift, Vector3.UnitY);
            var frame = BrNeutral(BrView(eye, target, up) * BrProj90, eye);
            pipeline.BeginFrame(frame);
            var buffer = CreateQuadBuffer(device, queue, new[] { PackedQuad.Pack(cellX, 0, cellZ, 1, 1, face, 1) });
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    pipeline.BindPass(pass, face >= 6 ? BlockRenderPipeline.Pass.Cross : BlockRenderPipeline.Pass.Opaque, buffer);
                    pipeline.Draw(pass, 0, 1, 0);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var (r, g, b, _) = GetPixel(await ReadBackPixels(device, queue, colorTex, 64, 64), 64, px, py);
                return (r, g, b);
            }
            finally
            {
                pipeline.ReleaseQuadBuffer(buffer);
                buffer.Destroy(); buffer.Dispose();
                DisposeTargets(colorTex, depthTex, colorView, depthView);
            }
        }

        [TestMethod]
        public async Task BlockRender_ThinBlock_IsSquashedToItsFloor_AndItsSidesShowTheTopStripOfTheTileTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // One tile, 16 px: its top four rows red, the rest blue
            var atlas = BaSolidAtlas(16, 1, 1, _ => (0, 0, 255, 255));
            for (int y = 0; y < 4; y++) for (int x = 0; x < 16; x++) { int i = (y * 16 + x) * 4; atlas[i] = 255; atlas[i + 2] = 0; }
            // Seen from the front, the block spans rows 21..43 (y 0 at row 43): a full block shows its middle (row 32) and its lower part (row 40)
            var full = BlockMaterial.Uniform(0);
            var thin = BlockMaterial.Uniform(0, flags: BlockMaterial.Thickness(4));
            var fullMid = await BaRenderPixel(device, queue, atlas, 1, full, 2, 32, 32);
            if (fullMid.r + fullMid.g + fullMid.b < 60) throw new Exception("a full block should cover the middle of the view");
            // A block 4/16 thick: the middle (y = 0.5) is empty; just above its floor (y = 0.125, row 36) is the block
            var thinMid = await BaRenderPixel(device, queue, atlas, 1, thin, 2, 32, 32);
            if (thinMid.r + thinMid.g + thinMid.b > 10) throw new Exception($"the middle of a 4/16 block should be empty, got ({thinMid.r},{thinMid.g},{thinMid.b})");
            var thinLow = await BaRenderPixel(device, queue, atlas, 1, thin, 2, 32, 38);
            // The side shows the tile's top 4 rows (red), not the whole tile squeezed (mostly blue)
            if (thinLow.r < 120 || thinLow.b > 60) throw new Exception($"the side of a thin block should show the top strip of its tile (red), got ({thinLow.r},{thinLow.g},{thinLow.b})");
        });

        [TestMethod]
        public async Task BlockRender_ARamp_RisesAlongItsSlope_AndIsLevelWithoutOneTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Seen from the front (+Z side, 2 blocks from the camera) at 16 px a block: y 0 at row 40, y 1 at row 24, x 0.5 at column 32. A 4/16-thick block rising toward +X is at
            // x 0.9 (column 38) 0.9..1.15 high (rows 22-26) and at x 0.1 (column 26) 0.1..0.35 (rows 34-38). Level, it fills only rows 36-40.
            var atlas = BaSolidAtlas(16, 1, 1, _ => (255, 255, 255, 255));
            var level = BlockMaterial.Uniform(0, flags: BlockMaterial.Thickness(4));
            var east = BlockMaterial.Uniform(0, flags: BlockMaterial.Thickness(4) | BlockMaterial.Slope(BlockMaterial.SlopeEast));
            var west = BlockMaterial.Uniform(0, flags: BlockMaterial.Thickness(4) | BlockMaterial.Slope(BlockMaterial.SlopeWest));
            async Task<bool> Filled(BlockMaterial m, int col, int row) { var p = await BaRenderPixel(device, queue, atlas, 1, m, 2, col, row); return p.r + p.g + p.b > 60; }
            if (!await Filled(level, 26, 38) || !await Filled(level, 38, 38)) throw new Exception("a level 4/16 block fills row 38 at both ends");
            if (await Filled(level, 38, 24) || await Filled(level, 26, 24)) throw new Exception("a level 4/16 block must not reach row 24");
            if (!await Filled(east, 38, 24)) throw new Exception("the high end of a ramp rising toward +X must reach row 24 at column 38");
            if (await Filled(east, 26, 24)) throw new Exception("the low end of a ramp rising toward +X must not reach row 24");
            if (await Filled(east, 38, 38)) throw new Exception("a ramp has left the floor at its high end");
            if (!await Filled(east, 26, 36)) throw new Exception("the low end of a ramp is on the floor");
            if (!await Filled(west, 26, 24) || await Filled(west, 38, 24)) throw new Exception("a ramp rising toward -X is high at its -X end");
        });

        [TestMethod]
        public async Task BlockRender_ARamp_AwayFromTheOrigin_RisesByOneBlockAndNoMoreTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // The same ramp as above, in the cell (4, 4): the shear is measured from the cell's own corner
            var atlas = BaSolidAtlas(16, 1, 1, _ => (255, 255, 255, 255));
            var east = BlockMaterial.Uniform(0, flags: BlockMaterial.Thickness(4) | BlockMaterial.Slope(BlockMaterial.SlopeEast));
            async Task<bool> Filled(int col, int row) { var p = await BaRenderPixel(device, queue, atlas, 1, east, 2, col, row, 4, 4); return p.r + p.g + p.b > 60; }
            if (!await Filled(38, 24)) throw new Exception("the high end of the ramp reaches row 24 at column 38");
            if (await Filled(38, 8)) throw new Exception("and goes no higher than one block: row 8 (y 2) is empty");
            if (await Filled(26, 24)) throw new Exception("the low end stays low");
            if (!await Filled(26, 36)) throw new Exception("the low end is on the floor");
        });

        [TestMethod]
        public async Task BlockRender_ARamp_ThroughTheMesherAndTheArena_InASectionAboveTheFirst_RisesByOneBlockTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            // The real path: a column meshed by ColumnMesher (the section embedded in each quad, quads in the arena), a ramp cell at (4, 20, 4) in the second section, a stone block east of it
            const int sections = 2;
            using var arena = new QuadArena(accelerator);
            using var mesher = new ColumnMesher(accelerator, sections) { MissingNeighborsAreSolid = false, EmbedSectionIndex = true, Arena = arena };
            var flags = new int[] { BlockClass.None, BlockClass.Opaque, BlockClass.Cutout };      // air, stone, the thin block
            mesher.SetBlockClasses(flags);
            var col = new ushort[16 * 16 * 16 * sections];
            col[4 + 4 * 16 + 20 * 256] = 2;       // the ramp cell
            col[5 + 4 * 16 + 20 * 256] = 1;       // a stone block beside it, one block high
            using var gpu = CmUpload(accelerator, col);
            using var mesh = await mesher.MeshAsync(gpu);
            var atlas = BaSolidAtlas(16, 1, 1, _ => (255, 255, 255, 255));
            using var textures = BlockTextureAtlas.FromRgba(device, queue, atlas, 16, 16, 16);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            var ramp = BlockMaterial.Uniform(0, flags: BlockMaterial.Thickness(1) | BlockMaterial.Slope(BlockMaterial.SlopeWest));
            pipeline.SetMaterials(new[] { BlockMaterial.Missing, BlockMaterial.Uniform(0, new Vector3(0.3f, 0.3f, 1f)), ramp });
            pipeline.SetSectionOrigin(0, Vector3.Zero);
            // From the south at 16 px a block, 3 blocks from the ramp's z, at the ramp's floor + 0.5: the ramp (rising toward -X) is seen edge-on; its high end is at x 4, its low end x 5
            var eye = new Vector3(4.6f, 20.5f, 7f);
            pipeline.BeginFrame(BrNeutral(BrView(eye, new Vector3(4.6f, 20.5f, 4f), Vector3.UnitY) * BrProj90, eye));
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    var list = mesh.Opaque;
                    pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, list.Buffer!.GetGPUBuffer()!);
                    pipeline.Draw(pass, list.FirstQuad, list.Count, 0);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var px = await ReadBackPixels(device, queue, colorTex, 64, 64);
                // x 5 is column 32 + (5-4.6)*21 = 40 (the stone block's face); x 4.1 is column 32 - 10 = 22: the ramp's high end, a block above the floor
                // Scale: the ramp's +Z face is 3 blocks from the eye... the stone is 1 block high from y 20 to 21: rows (y 20 -> 32 + 0.5*21 = 42) to (y 21 -> 21)
                bool Filled(int col, int row) { var (r, g, b, _) = GetPixel(px, 64, col, row); return r + g + b > 60; }
                if (!Filled(40, 30)) throw new Exception("the stone block (one high) fills the middle of its column");
                if (Filled(40, 10)) throw new Exception("and nothing is above the stone");
                // The ramp's high end (x ~ 4.1, column ~ 22) rises to about the stone's top (row 21) and not a block above it (row 0..10)
                int top22 = Enumerable.Range(0, 64).FirstOrDefault(r => Filled(22, r), -1);
                int top40 = Enumerable.Range(0, 64).FirstOrDefault(r => Filled(40, r), -1);
                if (top22 < 0 || top22 > 24) throw new Exception($"the ramp's high end must reach about the stone's top: its highest row is {top22}, the stone's {top40}");
                for (int row = 0; row < 14; row++) if (Filled(22, row)) throw new Exception($"the ramp must not rise a block above the stone: pixel (22,{row}) is filled");
            }
            finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
        });

        [TestMethod]
        public async Task BlockRender_ARamp_BesideItsFlatNeighbours_InAFullHeightColumn_RisesByOneBlockTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            // The viewer's scene: a stone floor, a flat thin rail, the ramp rising west, a stone block with a flat thin rail on top, all in a 24-section column
            const int sections = 24;
            using var arena = new QuadArena(accelerator);
            using var mesher = new ColumnMesher(accelerator, sections) { MissingNeighborsAreSolid = false, EmbedSectionIndex = true, Arena = arena };
            mesher.SetBlockClasses(new int[] { BlockClass.None, BlockClass.Opaque, BlockClass.Cutout, BlockClass.Cutout });   // air, stone, ramp, flat thin rail
            var col = new ushort[16 * 16 * 16 * sections];
            int Y = 4;                                           // the floor level of the scene (y = -60 is index 4)
            for (int x = 0; x < 16; x++) for (int z = 0; z < 16; z++) col[x + z * 16 + (Y - 1) * 256] = 1;   // the floor
            col[3 + 10 * 16 + Y * 256] = 1;                      // the stone block
            col[3 + 10 * 16 + (Y + 1) * 256] = 3;                // the flat rail on it
            col[4 + 10 * 16 + Y * 256] = 2;                      // the ramp, rising toward -X
            col[5 + 10 * 16 + Y * 256] = 3;                      // the flat rail it comes from
            using var gpu = CmUpload(accelerator, col);
            using var mesh = await mesher.MeshAsync(gpu);
            var atlas = BaSolidAtlas(16, 1, 1, _ => (255, 255, 255, 255));
            using var textures = BlockTextureAtlas.FromRgba(device, queue, atlas, 16, 16, 16);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            var thin = BlockMaterial.Uniform(0, new Vector3(1f, 0.2f, 0.2f), flags: BlockMaterial.Thickness(1));
            var ramp = BlockMaterial.Uniform(0, flags: BlockMaterial.Thickness(1) | BlockMaterial.Slope(BlockMaterial.SlopeWest));
            pipeline.SetMaterials(new[] { BlockMaterial.Missing, BlockMaterial.Uniform(0, new Vector3(0.3f, 0.3f, 1f)), ramp, thin });
            pipeline.SetSectionOrigin(0, Vector3.Zero);
            // From the south at 16 px a block (2 blocks away), at the floor level + 0.5 (a block's height is 16 px): rows: y Y -> 40, Y+1 -> 24
            var eye = new Vector3(4.5f, Y + 0.5f, 12f);
            pipeline.BeginFrame(BrNeutral(BrView(eye, new Vector3(4.5f, Y + 0.5f, 10f), Vector3.UnitY) * BrProj90, eye));
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    var list = mesh.Opaque;
                    pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, list.Buffer!.GetGPUBuffer()!);
                    pipeline.Draw(pass, list.FirstQuad, list.Count, 0);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var px = await ReadBackPixels(device, queue, colorTex, 64, 64);
                bool Filled(int c, int r) { var (rr, g, b, _) = GetPixel(px, 64, c, r); return rr + g + b > 60; }
                // The stone block at x 3..4 (columns 8..24) is one block high (rows 24..40); the ramp, x 4..5 (columns 24..40), must not rise above the stone block's rail (about row 22)
                int highest = Enumerable.Range(0, 64).Where(r => Enumerable.Range(24, 16).Any(c => Filled(c, r))).DefaultIfEmpty(-1).Min();
                int stoneTop = Enumerable.Range(0, 64).Where(r => Enumerable.Range(9, 14).Any(c => Filled(c, r))).DefaultIfEmpty(-1).Min();
                if (highest < 0 || stoneTop < 0) throw new Exception($"nothing drawn: ramp columns {highest}, stone columns {stoneTop}");
                if (highest < stoneTop - 3) throw new Exception($"the ramp reaches row {highest}, above the stone block with its rail (row {stoneTop}): it rises more than one block");
            }
            finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
        });

        [TestMethod]
        public async Task BlockRender_PlantFromBottom_TakesItsTextureFromFaceFive_WhileTheCubeFacesKeepTheirOwnTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Tile 0 green (the water, in a real block), tile 1 red (the plant)
            var atlas = BaSolidAtlas(16, 2, 1, k => k == 0 ? ((byte)0, (byte)255, (byte)0, (byte)255) : ((byte)255, (byte)0, (byte)0, (byte)255));
            var m = BlockMaterial.Create(0, 0, 0, 0, 0, 1, flags: BlockMaterial.FlagPlantFromBottom);
            var plant = await BaRenderPixel(device, queue, atlas, 2, m, 6, 32, 32);
            if (plant.r < 150 || plant.g > 60) throw new Exception($"the plant should be the face-5 tile (red), got ({plant.r},{plant.g},{plant.b})");
            var top = await BaRenderPixel(device, queue, atlas, 2, m, 4, 32, 32);
            if (top.g < 150 || top.r > 60) throw new Exception($"the top face should keep its own tile (green), got ({top.r},{top.g},{top.b})");
            var ordinary = BlockMaterial.Create(0, 0, 0, 0, 0, 1);
            var plain = await BaRenderPixel(device, queue, atlas, 2, ordinary, 6, 32, 32);
            if (plain.g < 150 || plain.r > 60) throw new Exception($"without the flag a plant uses face 0 (green), got ({plain.r},{plain.g},{plain.b})");
        });

        [TestMethod]
        public async Task BlockRender_TintFaces_TintOnlyTheNamedFaces_TheTopOfGrassAndNotItsSidesTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // A white texture, a green tint on the TOP face only (bit 4): the top comes out green, a side stays white (lit as a side)
            var m = BlockMaterial.Uniform(0, new Vector3(0.5f, 1f, 0.5f), flags: BlockMaterial.TintFaces(1 << 4));
            BaExpect(await BaRenderFace(device, queue, m, 4), 134, 255, 134, "the top: white x tint (0.5, 1, 0.5) x 1.05, clamped");
            BaExpect(await BaRenderFace(device, queue, m, 2), 178, 178, 178, "a +Z side: white x 0.82 x 0.85, no tint");
            // The same tint on every face when the block is tinted outright
            var all = BlockMaterial.Uniform(0, new Vector3(0.5f, 1f, 0.5f), flags: BlockMaterial.FlagTinted);
            BaExpect(await BaRenderFace(device, queue, all, 2), 89, 178, 89, "FlagTinted tints the side too");
            // A mask that does not name the face leaves it alone
            var other = BlockMaterial.Uniform(0, new Vector3(0.5f, 1f, 0.5f), flags: BlockMaterial.TintFaces(1 << 5));
            BaExpect(await BaRenderFace(device, queue, other, 4), 255, 255, 255, "the top is not in the mask");
        });

        [TestMethod]
        public async Task BlockRender_Darken_ScalesTheColourBy07_ForWhatLiesUnderWaterTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            BaExpect(await BaRenderFace(device, queue, BlockMaterial.Uniform(0), 4), 255, 255, 255, "control: an untouched white top (1.05 clamps)");
            BaExpect(await BaRenderFace(device, queue, BlockMaterial.Uniform(0, flags: BlockMaterial.FlagDarken), 4), 187, 187, 187, "darkened to 0.7, then the top's 1.05: 0.735");
            BaExpect(await BaRenderFace(device, queue, BlockMaterial.Uniform(0, new Vector3(1f, 0.5f, 0.5f), flags: BlockMaterial.FlagTinted | BlockMaterial.FlagDarken), 4),
                187, 94, 94, "a tint and darkening together: (1, 0.5, 0.5) x 0.7 x 1.05");
        });

        [TestMethod]
        public async Task BlockRender_AnimatedTile_PlaysItsTimelineByTime_AndLeavesOtherTilesAloneTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // tile 0 is the animated one (its own colour, white, is never shown); 1 red, 2 green, 3 blue; the timeline lists a tile per TICK
            var atlas = BaSolidAtlas(4, 2, 2, k => k switch { 0 => (255, 255, 255, 255), 1 => (255, 0, 0, 255), 2 => (0, 255, 0, 255), _ => (0, 0, 255, 255) });
            var materials = new[] { BlockMaterial.Missing, BlockMaterial.Uniform(0), BlockMaterial.Uniform(1) };
            var animations = new[] { new TextureAnimation(0, new[] { 1, 1, 2, 3 }) };
            //             tick 0     tick 1      tick 2      tick 3      tick 4 = 0   tick 22 = 2   tick 20000 = 0 (a long session)
            var times = new[] { 0.00f, 0.05f, 0.10f, 0.15f, 0.20f, 1.10f, 1000.0f };
            var want = new[] { (255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 0) };
            var frames = await BaRenderTops(device, queue, atlas, 8, 8, 4, materials, animations, times, 2);
            for (int i = 0; i < times.Length; i++)
            {
                BaExpect(frames[i][0], want[i].Item1, want[i].Item2, want[i].Item3, $"animated tile at t={times[i]} s");
                BaExpect(frames[i][1], 255, 0, 0, $"the static tile 1 at t={times[i]} s");
            }
        });

        [TestMethod]
        public async Task BlockRender_SetAnimations_RejectsBadTimelines_AndANewAtlasDropsTheOldOnesTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var atlas = BaSolidAtlas(4, 2, 2, k => ((byte)(k * 60), 0, 0, 255));
            using var textures = BlockTextureAtlas.FromRgba(device, queue, atlas, 8, 8, 4);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);

            void Throws(string what, Action act)
            {
                try { act(); }
                catch (ArgumentException) { return; }          // ArgumentOutOfRangeException is one
                throw new Exception($"{what} must be rejected");
            }
            Throws("a base tile outside the atlas", () => pipeline.SetAnimations(new[] { new TextureAnimation(4, new[] { 0 }) }));
            Throws("a timeline naming a tile outside the atlas", () => pipeline.SetAnimations(new[] { new TextureAnimation(0, new[] { 1, 9 }) }));
            Throws("an empty timeline", () => pipeline.SetAnimations(new[] { new TextureAnimation(0, System.Array.Empty<int>()) }));
            Throws("one tile animated twice", () => pipeline.SetAnimations(new[] { new TextureAnimation(0, new[] { 1 }), new TextureAnimation(0, new[] { 2 }) }));

            // A rejected call changed nothing and a good one works; then a new atlas starts with no animations (its tile numbers are different)
            pipeline.SetAnimations(new[] { new TextureAnimation(0, new[] { 1, 2 }) });
            using var other = BlockTextureAtlas.FromRgba(device, queue, atlas, 8, 8, 4);
            pipeline.SetTextures(other);
            pipeline.SetAnimations(System.Array.Empty<TextureAnimation>());
        });
    }
}
