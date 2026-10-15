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
