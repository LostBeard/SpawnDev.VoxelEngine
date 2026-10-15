using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // The ENTITY pipeline: one instanced draw of textured boxes (Minecraft's model parts), the box texture laid out as Minecraft lays it out,
    // lit and fogged by the same frame uniforms as the blocks.
    public abstract partial class VoxelEngineTestBase
    {
        private static void ErPaint(byte[] atlas, int atlasW, int x0, int y0, int w, int h, byte r, byte g, byte b, byte a = 255)
        {
            for (int y = y0; y < y0 + h; y++)
                for (int x = x0; x < x0 + w; x++)
                {
                    int i = (y * atlasW + x) * 4;
                    atlas[i] = r; atlas[i + 1] = g; atlas[i + 2] = b; atlas[i + 3] = a;
                }
        }

        /// <summary>
        /// Draw the parts seen from <paramref name="eye"/> at <paramref name="target"/> (64 x 64 px, 90 degrees), neutral light (ambient 1), and return the
        /// pixels. <paramref name="up"/> is the screen's up.
        /// </summary>
        private static async Task<byte[]> ErRender(GPUDevice device, GPUQueue queue, byte[] atlas, int atlasW, int atlasH, EntityPartInstance[] parts,
            Vector3 eye, Vector3 target, Vector3 up, Vector3? clear = null, Vector4? fog = null)
        {
            using var textures = EntityTextureAtlas.FromRgba(device, queue, atlas, atlasW, atlasH);
            using var blocks = new BlockRenderPipeline();
            blocks.Init(device, queue, "bgra8unorm", maxSections: 4);
            using var entities = new EntityRenderPipeline();
            entities.Init(device, queue, "bgra8unorm", blocks.FrameBuffer);
            entities.SetTextures(textures);
            entities.SetParts(parts);

            var frame = BrNeutral(BrView(eye, target, up) * BrProj90, eye);
            if (fog.HasValue) { frame.FogColor = new Vector4(1, 0, 0, 0); frame.FogRange = fog.Value; }
            blocks.BeginFrame(frame);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = blocks.BeginRenderPass(encoder, colorView, depthView, clear ?? Vector3.Zero))
                {
                    entities.Draw(pass);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                return await ReadBackPixels(device, queue, colorTex, 64, 64);
            }
            finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
        }

        /// <summary>One 16 x 16 x 16 px box (a block) at the model origin, model px scaled to blocks, texture origin (0, 0).</summary>
        private static EntityPartInstance ErBlockBox(bool mirror = false, Vector3? tint = null, float inflate = 0f, Matrix4x4? model = null) => new()
        {
            Model = model ?? Matrix4x4.CreateScale(1f / 16f),
            OriginInflate = new Vector4(0, 0, 0, inflate),
            SizeFlags = new Vector4(16, 16, 16, mirror ? EntityPartInstance.FlagMirror : 0f),
            Uv = new Vector4(0, 0, 0, 0),
            Tint = new Vector4(tint ?? Vector3.One, 1),
        };

        // A 16 x 16 x 16 box's layout is 64 x 32: top (16,0) bottom (32,0), then right (0,16) front (16,16) left (32,16) back (48,16), each 16 x 16
        private static readonly (string name, int x, int y, byte r, byte g, byte b)[] ErRegions =
        {
            ("top", 16, 0, 255, 0, 0), ("bottom", 32, 0, 0, 255, 0),
            ("right", 0, 16, 0, 0, 255), ("front", 16, 16, 255, 255, 0), ("left", 32, 16, 0, 255, 255), ("back", 48, 16, 255, 0, 255),
        };

        private static byte[] ErRegionAtlas()
        {
            var atlas = new byte[64 * 32 * 4];
            foreach (var (_, x, y, r, g, b) in ErRegions) ErPaint(atlas, 64, x, y, 16, 16, r, g, b);
            return atlas;
        }

        /// <summary>The hue of a pixel by its largest channels (lighting scales all three alike): "R", "G", "B", "RG", "GB", "RB", or "-" when dark.</summary>
        private static string ErHue((byte r, byte g, byte b, byte a) px)
        {
            int max = Math.Max(px.r, Math.Max(px.g, px.b));
            if (max < 60) return "-";
            string s = "";
            if (px.r > max * 0.6) s += "R";
            if (px.g > max * 0.6) s += "G";
            if (px.b > max * 0.6) s += "B";
            return s;
        }

        private static string ErCentre(byte[] pixels) => ErHue(GetPixel(pixels, 64, 32, 32));

        [TestMethod]
        public async Task EntityRender_EachFaceOfABox_ShowsItsOwnRegionOfTheCrossLayoutTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // The block is at 0..1 in x and z, y points DOWN in model space: its "top" is the face at y = 0
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            var views = new (string face, Vector3 eye, Vector3 up, string expect)[]
            {
                ("front (z min)", c + new Vector3(0, 0, -2.5f), -Vector3.UnitY, "RG"),
                ("back (z max)", c + new Vector3(0, 0, 2.5f), -Vector3.UnitY, "RB"),
                ("right side (x min)", c + new Vector3(-2.5f, 0, 0), -Vector3.UnitY, "B"),
                ("left side (x max)", c + new Vector3(2.5f, 0, 0), -Vector3.UnitY, "GB"),
                ("top (y min)", c + new Vector3(0, -2.5f, 0), Vector3.UnitZ, "R"),
                ("bottom (y max)", c + new Vector3(0, 2.5f, 0), Vector3.UnitZ, "G"),
            };
            foreach (var (face, eye, up, expect) in views)
            {
                var pixels = await ErRender(device, queue, ErRegionAtlas(), 64, 32, new[] { ErBlockBox() }, eye, c, up);
                var got = ErCentre(pixels);
                if (got != expect) throw new Exception($"{face}: centre hue {got}, expected {expect}");
            }
        });

        [TestMethod]
        public async Task EntityRender_AMirroredBox_SwapsItsSideRegions_AndLeavesTheOthersTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            var mirrored = new[] { ErBlockBox(mirror: true) };
            var right = ErCentre(await ErRender(device, queue, ErRegionAtlas(), 64, 32, mirrored, c + new Vector3(-2.5f, 0, 0), c, -Vector3.UnitY));
            var left = ErCentre(await ErRender(device, queue, ErRegionAtlas(), 64, 32, mirrored, c + new Vector3(2.5f, 0, 0), c, -Vector3.UnitY));
            var front = ErCentre(await ErRender(device, queue, ErRegionAtlas(), 64, 32, mirrored, c + new Vector3(0, 0, -2.5f), c, -Vector3.UnitY));
            if (right != "GB") throw new Exception($"mirrored: the x-min side shows {right}, expected the left region (GB)");
            if (left != "B") throw new Exception($"mirrored: the x-max side shows {left}, expected the right region (B)");
            if (front != "RG") throw new Exception($"mirrored: the front shows {front}, still expected its own region (RG)");
        });

        [TestMethod]
        public async Task EntityRender_AFaceIsUprightAndUnmirrored_SeenFromOutsideTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // The front region split into quadrants: top-left red, top-right green, bottom-left blue, bottom-right white
            var atlas = ErRegionAtlas();
            ErPaint(atlas, 64, 16, 16, 8, 8, 255, 0, 0);
            ErPaint(atlas, 64, 24, 16, 8, 8, 0, 255, 0);
            ErPaint(atlas, 64, 16, 24, 8, 8, 0, 0, 255);
            ErPaint(atlas, 64, 24, 24, 8, 8, 255, 255, 255);
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            // From the front (z min) with model -y (the box's top) as screen up: u must grow to the screen's RIGHT
            var pixels = await ErRender(device, queue, atlas, 64, 32, new[] { ErBlockBox() }, c + new Vector3(0, 0, -1.2f), c, -Vector3.UnitY);
            string q(int x, int y) => ErHue(GetPixel(pixels, 64, x, y));
            if (q(24, 24) != "R" || q(40, 24) != "G" || q(24, 40) != "B" || q(40, 40) != "RGB")
                throw new Exception($"front quadrants (TL TR BL BR): {q(24, 24)} {q(40, 24)} {q(24, 40)} {q(40, 40)}, expected R G B RGB");
        });

        [TestMethod]
        public async Task EntityRender_EverySide_ReadsLeftToRight_SeenFromOutsideTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Each side region painted red on its left half and green on its right half: whichever side we look at from outside, red is on the screen's left
            var atlas = ErRegionAtlas();
            foreach (var (name, x, y, _, _, _) in ErRegions.Where(r => r.name is "right" or "front" or "left" or "back"))
            {
                ErPaint(atlas, 64, x, y, 8, 16, 255, 0, 0);
                ErPaint(atlas, 64, x + 8, y, 8, 16, 0, 255, 0);
            }
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            var sides = new (string side, Vector3 eye)[]
            {
                ("front", c + new Vector3(0, 0, -1.2f)), ("back", c + new Vector3(0, 0, 1.2f)),
                ("right", c + new Vector3(-1.2f, 0, 0)), ("left", c + new Vector3(1.2f, 0, 0)),
            };
            foreach (var (side, eye) in sides)
            {
                var pixels = await ErRender(device, queue, atlas, 64, 32, new[] { ErBlockBox() }, eye, c, -Vector3.UnitY);
                string l = ErHue(GetPixel(pixels, 64, 24, 32)), r = ErHue(GetPixel(pixels, 64, 40, 32));
                if (l != "R" || r != "G") throw new Exception($"{side}: screen left {l}, right {r}, expected R then G");
            }
        });

        [TestMethod]
        public async Task EntityRender_FaceShading_TopFullBright_SidesLess_BottomDarkestTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var atlas = new byte[64 * 32 * 4];
            for (int i = 0; i < atlas.Length; i++) atlas[i] = 255;   // all white
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            async Task<int> Shade(Vector3 eye, Vector3 up) => GetPixel(await ErRender(device, queue, atlas, 64, 32, new[] { ErBlockBox() }, eye, c, up), 64, 32, 32).r;
            int top = await Shade(c + new Vector3(0, -1.2f, 0), Vector3.UnitZ);       // world up: the model's y points down, so its top is at y minimum
            int side = await Shade(c + new Vector3(0, 0, -1.2f), -Vector3.UnitY);
            int bottom = await Shade(c + new Vector3(0, 1.2f, 0), Vector3.UnitZ);
            // The ENTITY's own up/down is what the shading follows once the model is placed in the world; here the test's model matrix does not flip y,
            // so "top" in the model is world y minimum: the normal in world space points down, and gets the bottom shade
            if (side < 205 || side > 230) throw new Exception($"side {side}, expected about 217 (0.85)");
            if (Math.Abs(Math.Min(top, bottom) - 178) > 6) throw new Exception($"the face looking along world -y: {Math.Min(top, bottom)}, expected about 178 (0.70)");
            if (Math.Max(top, bottom) < 245) throw new Exception($"the face looking along world +y: {Math.Max(top, bottom)}, expected full brightness");
        });

        [TestMethod]
        public async Task EntityRender_TheTopFace_RunsItsTextureFrontwardTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // The top region (16,0) 16 x 16: its LOWER edge (v large) is the one next to the front face's top edge, so the front (z min) is the lower half
            var atlas = ErRegionAtlas();
            ErPaint(atlas, 64, 16, 0, 16, 8, 255, 0, 0);     // upper half of the region: red
            ErPaint(atlas, 64, 16, 8, 16, 8, 0, 0, 255);     // lower half: blue
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            // Seen from above with the front (z min) at the bottom of the screen: up on screen is +z
            var pixels = await ErRender(device, queue, atlas, 64, 32, new[] { ErBlockBox() }, c + new Vector3(0, -1.2f, 0), c, Vector3.UnitZ);
            string q(int y) => ErHue(GetPixel(pixels, 64, 32, y));
            if (q(24) != "R" || q(40) != "B") throw new Exception($"top, screen upper / lower: {q(24)} / {q(40)}, expected R (back) / B (front)");
        });

        [TestMethod]
        public async Task EntityRender_Tint_MultipliesTheTexture_AndTransparentTexelsAreNotDrawnTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var atlas = new byte[64 * 32 * 4];
            ErPaint(atlas, 64, 16, 16, 16, 16, 255, 255, 255);              // an opaque white front
            ErPaint(atlas, 64, 16, 16, 16, 4, 255, 255, 255, a: 0);         // its top four rows transparent
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            var eye = c + new Vector3(0, 0, -1.2f);
            var tinted = await ErRender(device, queue, atlas, 64, 32, new[] { ErBlockBox(tint: new Vector3(1f, 0.5f, 0f)) }, eye, c, -Vector3.UnitY,
                clear: new Vector3(0, 0, 1));
            var mid = GetPixel(tinted, 64, 32, 40);
            // White x tint (1, 0.5, 0) x the side shade 0.85
            if (Math.Abs(mid.r - 217) > 4 || Math.Abs(mid.g - 108) > 4 || mid.b > 4) throw new Exception($"tinted: ({mid.r},{mid.g},{mid.b}), expected about (217,108,0)");
            // The transparent band shows what is behind (the clear colour, blue), not the box
            var band = GetPixel(tinted, 64, 32, 14);
            if (band.b < 200 || band.r > 20) throw new Exception($"transparent band: ({band.r},{band.g},{band.b}), expected the blue clear colour");
        });

        [TestMethod]
        public async Task EntityRender_Inflation_GrowsTheBox_WithoutMovingItsTextureTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var atlas = new byte[64 * 32 * 4];
            ErPaint(atlas, 64, 16, 16, 16, 16, 255, 0, 0);
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            // From 1.2 blocks away at 90 degrees the box (1 block) fills 32 of 64 px; inflated by 4 px (0.25 block each side) it fills 48
            var plain = await ErRender(device, queue, atlas, 64, 32, new[] { ErBlockBox() }, c + new Vector3(0, 0, -1.2f), c, -Vector3.UnitY);
            var fat = await ErRender(device, queue, atlas, 64, 32, new[] { ErBlockBox(inflate: 4f) }, c + new Vector3(0, 0, -1.2f), c, -Vector3.UnitY);
            int Count(byte[] px) { int n = 0; for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) if (GetPixel(px, 64, x, y).r > 100) n++; return n; }
            int a = Count(plain), b = Count(fat);
            if (b <= a * 1.3) throw new Exception($"inflated box covers {b} px, plain {a}: it did not grow");
            if (ErHue(GetPixel(fat, 64, 32, 32)) != "R") throw new Exception("the inflated box's face lost its texture");
        });

        [TestMethod]
        public async Task EntityRender_ManyParts_OneDraw_NearestWinsByDepthTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Two boxes in a line seen from the front: the near one (a red front) hides the far one (a green front), whatever order they were uploaded in
            var atlas = new byte[128 * 32 * 4];
            ErPaint(atlas, 128, 16, 16, 16, 16, 255, 0, 0);
            ErPaint(atlas, 128, 64 + 16, 16, 16, 16, 0, 255, 0);
            var near = ErBlockBox();
            var far = ErBlockBox(model: Matrix4x4.CreateScale(1f / 16f) * Matrix4x4.CreateTranslation(0, 0, 3));
            far.Uv = new Vector4(64, 0, 0, 0);
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            foreach (var order in new[] { new[] { near, far }, new[] { far, near } })
            {
                var pixels = await ErRender(device, queue, atlas, 128, 32, order, c + new Vector3(0, 0, -2f), c, -Vector3.UnitY);
                if (ErCentre(pixels) != "R") throw new Exception($"centre {ErCentre(pixels)}, expected the near box (R) in front");
            }
        });

        [TestMethod]
        public async Task EntityRender_Fog_BlendsDistantPartsTowardTheFogColourTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var atlas = new byte[64 * 32 * 4];
            ErPaint(atlas, 64, 16, 16, 16, 16, 255, 255, 255);
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            // Fog from 0 to 0.5 block: the box face, 0.7 blocks from the eye, is fully fogged (red)
            var pixels = await ErRender(device, queue, atlas, 64, 32, new[] { ErBlockBox() }, c + new Vector3(0, 0, -1.2f), c, -Vector3.UnitY, fog: new Vector4(0, 0.5f, 0, 0));
            var p = GetPixel(pixels, 64, 32, 32);
            if (p.r < 240 || p.g > 15 || p.b > 15) throw new Exception($"fogged centre ({p.r},{p.g},{p.b}), expected the fog colour (red)");
        });

        [TestMethod]
        public async Task EntityRender_AtlasRegions_CanBeRewrittenFromACanvas_WhileTheAtlasIsInUseTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // The back region (48, 16) starts green; a 16 x 16 canvas drawn red then replaces it (a skin arriving)
            var atlas = ErRegionAtlas();
            ErPaint(atlas, 64, 48, 16, 16, 16, 0, 255, 0);
            using var textures = EntityTextureAtlas.FromRgba(device, queue, atlas, 64, 32);
            using var blocks = new BlockRenderPipeline();
            blocks.Init(device, queue, "bgra8unorm", maxSections: 4);
            using var entities = new EntityRenderPipeline();
            entities.Init(device, queue, "bgra8unorm", blocks.FrameBuffer);
            entities.SetTextures(textures);
            entities.SetParts(new[] { ErBlockBox() });
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            var eye = c + new Vector3(0, 0, 1.2f);   // seen from the back

            async Task<string> Centre()
            {
                blocks.BeginFrame(BrNeutral(BrView(eye, c, -Vector3.UnitY) * BrProj90, eye));
                var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
                try
                {
                    using var encoder = device.CreateCommandEncoder();
                    using (var pass = blocks.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero)) { entities.Draw(pass); pass.End(); }
                    using var cmd = encoder.Finish();
                    queue.Submit(new[] { cmd });
                    await queue.OnSubmittedWorkDone();
                    return ErHue(GetPixel(await ReadBackPixels(device, queue, colorTex, 64, 64), 64, 32, 32));
                }
                finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
            }

            var before = await Centre();
            if (before != "G") throw new Exception($"before the rewrite the back is {before}, expected G");
            using var canvas = new OffscreenCanvas(16, 16);
            using var ctx = canvas.Get2DContext();
            ctx.FillStyle = "rgb(255,0,0)";
            ctx.FillRect(0, 0, 16, 16);
            textures.WriteCanvas(48, 16, canvas);
            var after = await Centre();
            if (after != "R") throw new Exception($"after the rewrite the back is {after}, expected R");
            bool threw = false;
            try { textures.WriteCanvas(56, 0, canvas); } catch (ArgumentOutOfRangeException) { threw = true; }
            if (!threw) throw new Exception("a canvas that does not fit the atlas was accepted");
        });

        [TestMethod]
        public async Task EntityRender_MorePartsThanTheBufferHolds_GrowsIt_AndDrawsThemAllTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var atlas = ErRegionAtlas();
            using var textures = EntityTextureAtlas.FromRgba(device, queue, atlas, 64, 32);
            using var blocks = new BlockRenderPipeline();
            blocks.Init(device, queue, "bgra8unorm", maxSections: 4);
            using var entities = new EntityRenderPipeline();
            entities.Init(device, queue, "bgra8unorm", blocks.FrameBuffer, initialParts: 4);
            entities.SetTextures(textures);
            var many = Enumerable.Range(0, 5000).Select(_ => ErBlockBox()).ToArray();
            int drawn = entities.SetParts(many);
            if (drawn != 5000 || entities.PartCount != 5000 || entities.DroppedParts != 0) throw new Exception($"drew {drawn}, PartCount {entities.PartCount}, dropped {entities.DroppedParts}");
            entities.SetParts(many.AsSpan(0, 3));
            if (entities.PartCount != 3) throw new Exception($"a smaller frame left PartCount {entities.PartCount}");
        });
    }
}
