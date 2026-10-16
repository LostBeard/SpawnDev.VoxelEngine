using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // The SKY: its gradient, the sun and the moon in their phases, stars at night, a layer of clouds - one full-screen triangle drawn first in the pass.
    public abstract partial class VoxelEngineTestBase
    {
        private const int SkSize = 64;

        // An atlas for the tests: sun (16 x 16, yellow), moon phases (8 frames of 8 x 8 in 4 columns, each its own flat colour), clouds (2 x 2, only the first pixel opaque white)
        private static (byte[] rgba, int w, int h, Vector4 sun, Vector4 moon, Vector4 clouds) SkAtlas()
        {
            const int w = 64, h = 32;
            var px = new byte[w * h * 4];
            ErPaint(px, w, 0, 0, 16, 16, 255, 255, 0);                              // the sun
            for (int f = 0; f < 8; f++) ErPaint(px, w, 16 + (f % 4) * 8, (f / 4) * 8, 8, 8, (byte)(30 + f * 30), (byte)(200 - f * 20), 220);   // the moon's phases
            ErPaint(px, w, 0, 16, 1, 1, 255, 255, 255);                             // the cloud map: its first pixel is a cloud
            ErPaint(px, w, 1, 16, 1, 1, 255, 255, 255, 0);                          // the others are not
            ErPaint(px, w, 0, 17, 2, 1, 255, 255, 255, 0);
            return (px, w, h, new Vector4(0, 0, 16, 16), new Vector4(16, 0, 8, 8), new Vector4(0, 16, 2, 2));
        }

        private static SkyUniforms SkDay(Vector4 sun, Vector4 moon, Vector4 clouds) => new()
        {
            Camera = new Vector4(0, 0, 0, 1),
            SunDirection = new Vector4(-0.6f, -0.8f, 0, 1),            // under the horizon behind the camera, its moon high over the +x horizon and out of a level view: nothing in view unless a test says so
            Zenith = new Vector4(1, 0, 0, 0),                       // red overhead
            Horizon = new Vector4(0, 0, 1, 0),                      // blue at the horizon
            Cloud = new Vector4(1e9f, 0, 0, 0),                     // no clouds unless a test says so
            CloudColor = new Vector4(1, 1, 1, 1),
            SunRect = sun, MoonRect = moon, CloudRect = clouds,
        };

        private static async Task<byte[]> SkRender(GPUDevice device, GPUQueue queue, SkyUniforms sky, Vector3 eye, Vector3 target, Vector3 up, bool wall = false)
        {
            var (rgba, aw, ah, _, _, _) = SkAtlas();
            using var atlas = EntityTextureAtlas.FromRgba(device, queue, rgba, aw, ah);
            using var blocks = new BlockRenderPipeline();
            blocks.Init(device, queue, "bgra8unorm", maxSections: 4);
            using var skyPipe = new SkyPipeline();
            skyPipe.Init(device, queue, "bgra8unorm");
            skyPipe.SetTextures(atlas);
            using var wallTextures = BlockTextureAtlas.FromRgba(device, queue, BrAtlas(), 32, 32, BrTile);
            GPUBuffer? wallBuffer = null;
            if (wall)
            {
                blocks.SetTextures(wallTextures);
                var materials = new BlockMaterial[2];
                materials[1] = BlockMaterial.Uniform(BrLayerCyan);
                blocks.SetMaterials(materials);
                wallBuffer = CreateQuadBuffer(device, queue, new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 3, 1) });
            }
            var viewProj = BrView(eye, target, up) * BrProj90;
            blocks.BeginFrame(BrNeutral(viewProj, eye));
            Matrix4x4.Invert(viewProj, out var inverse);
            sky.InverseViewProjection = inverse;
            sky.Camera = new Vector4(eye, 1);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, SkSize, SkSize);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = blocks.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    skyPipe.Draw(pass, in sky);
                    if (wallBuffer != null) { blocks.BindPass(pass, BlockRenderPipeline.Pass.Opaque, wallBuffer); blocks.Draw(pass, 0, 1, 0); }
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                return await ReadBackPixels(device, queue, colorTex, SkSize, SkSize);
            }
            finally { if (wallBuffer != null) { wallBuffer.Destroy(); wallBuffer.Dispose(); } DisposeTargets(colorTex, depthTex, colorView, depthView); }
        }

        private static (byte r, byte g, byte b, byte a) SkAt(byte[] px, int x, int y) => GetPixel(px, SkSize, x, y);

        [TestMethod]
        public async Task Sky_RunsFromTheHorizonsColourToTheZeniths_AndBelowTheHorizonIsTheHorizonsTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var (_, _, _, sun, moon, clouds) = SkAtlas();
            // Looking along +x, level: the top row is 45 degrees up, the middle the horizon, the bottom 45 degrees down
            var px = await SkRender(device, queue, SkDay(sun, moon, clouds), Vector3.Zero, Vector3.UnitX, Vector3.UnitY);
            var bottom = SkAt(px, 32, 62); var middle = SkAt(px, 32, 32); var top = SkAt(px, 32, 1);
            if (bottom.b < 245 || bottom.r > 10) throw new Exception($"below the horizon it is the horizon's blue: ({bottom.r},{bottom.g},{bottom.b})");
            if (middle.b < 240 || middle.r > 25) throw new Exception($"at the horizon it is blue: ({middle.r},{middle.g},{middle.b})");
            if (top.r < 180 || top.b > 70) throw new Exception($"up at 45 degrees it is mostly the zenith's red: ({top.r},{top.g},{top.b})");
            // Looking straight up: the zenith
            var up = await SkRender(device, queue, SkDay(sun, moon, clouds), Vector3.Zero, Vector3.UnitY, -Vector3.UnitZ);
            var centre = SkAt(up, 32, 32);
            if (centre.r < 250 || centre.b > 5) throw new Exception($"straight up it is the zenith's red: ({centre.r},{centre.g},{centre.b})");
        });

        [TestMethod]
        public async Task Sky_TheSunIsWhereItIs_AndTheMoonOppositeItInThePhaseAskedForTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var (_, _, _, sun, moon, clouds) = SkAtlas();
            // The sun dead ahead: a yellow square at the centre; 30 degrees to the right (+z when looking along +x), it is to the right of the centre by tan 30 / tan 45 of half the picture
            var ahead = SkDay(sun, moon, clouds); ahead.SunDirection = new Vector4(1, 0, 0, 1);
            var a = await SkRender(device, queue, ahead, Vector3.Zero, Vector3.UnitX, Vector3.UnitY);
            var c = SkAt(a, 32, 32);
            if (c.r < 240 || c.g < 240) throw new Exception($"the sun dead ahead is yellow at the centre: ({c.r},{c.g},{c.b})");
            var corner = SkAt(a, 2, 2);
            if (corner.g > 40) throw new Exception($"away from the sun the sky is not yellow: ({corner.r},{corner.g},{corner.b})");
            var side = SkDay(sun, moon, clouds);
            side.SunDirection = new Vector4(MathF.Cos(MathF.PI / 6), 0, MathF.Sin(MathF.PI / 6), 1);
            var s = await SkRender(device, queue, side, Vector3.Zero, Vector3.UnitX, Vector3.UnitY);
            int x = 32 + (int)MathF.Round(MathF.Tan(MathF.PI / 6) * 32);
            var at = SkAt(s, x, 32);
            if (at.r < 240 || at.g < 240) throw new Exception($"the sun 30 degrees to the right is at x {x}: ({at.r},{at.g},{at.b})");
            if (SkAt(s, 32 - 8, 32).g > 40) throw new Exception("and not on the left of the centre");

            // The moon opposite the sun, in each phase its own flat colour: the sun behind the camera, the moon dead ahead
            for (int phase = 0; phase < 8; phase += 3)
            {
                var night = SkDay(sun, moon, clouds);
                night.SunDirection = new Vector4(-1, 0, 0, 0);
                night.Zenith.W = phase;
                var m = await SkRender(device, queue, night, Vector3.Zero, Vector3.UnitX, Vector3.UnitY);
                var got = SkAt(m, 32, 32);
                int er = 30 + phase * 30, eg = 200 - phase * 20;
                if (Math.Abs(got.r - er) > 8 || Math.Abs(got.g - eg) > 8 || got.b < 205) throw new Exception($"moon phase {phase}: ({got.r},{got.g},{got.b}), expected ({er},{eg},220)");
            }
        });

        [TestMethod]
        public async Task Sky_ShowsStarsAtNightOnly_AndNotInTheRainTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var (_, _, _, sun, moon, clouds) = SkAtlas();
            int Bright(byte[] px) { int n = 0; for (int y = 0; y < SkSize; y++) for (int x = 0; x < SkSize; x++) { var p = SkAt(px, x, y); if (p.r > 120 && p.g > 120 && p.b > 120) n++; } return n; }
            var starry = SkDay(sun, moon, clouds);
            starry.Zenith = new Vector4(0, 0, 0, 0); starry.Horizon = new Vector4(0, 0, 0, 1); starry.SunDirection = new Vector4(0, -1, 0, 0);
            // Looking up at the zenith, wide: 90 degrees across, so a few thousand stars of the whole sky show a few of them
            var night = await SkRender(device, queue, starry, Vector3.Zero, Vector3.UnitY, -Vector3.UnitZ);
            int stars = Bright(night);
            if (stars < 3) throw new Exception($"a night sky overhead shows stars, saw {stars} bright pixels");
            var day = starry; day.Horizon.W = 0f; day.SunDirection.W = 1f;
            if (Bright(await SkRender(device, queue, day, Vector3.Zero, Vector3.UnitY, -Vector3.UnitZ)) != 0) throw new Exception("no stars by day");
            var rain = starry; rain.Cloud.Z = 1f;
            // The rain's clouds are white over everything: look at the stars' share against the grey, not at their count; with rain the sky shows no stars through
            var rainy = await SkRender(device, queue, rain, Vector3.Zero, Vector3.UnitY, -Vector3.UnitZ);
            int rainStars = 0; var rc = SkAt(rainy, 5, 5);
            for (int y = 0; y < SkSize; y++) for (int x = 0; x < SkSize; x++) { var p = SkAt(rainy, x, y); if (p.r > rc.r + 60) rainStars++; }
            if (rainStars != 0) throw new Exception($"no stars in the rain, saw {rainStars}");
        });

        [TestMethod]
        public async Task Sky_CloudsAreWhereTheMapSays_TwelveBlocksAPixel_AndSlideTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var (_, _, _, sun, moon, clouds) = SkAtlas();
            var layer = SkDay(sun, moon, clouds);
            layer.Cloud = new Vector4(10f, 0f, 0f, 0f);     // a layer 10 blocks up
            layer.Zenith = new Vector4(0, 0, 1, 0); layer.Horizon = new Vector4(0, 0, 1, 0);   // blue sky
            // Straight up from x = 6, z = 6: the map pixel (0, 0): a cloud (white)
            var over = await SkRender(device, queue, layer, new Vector3(6, 0, 6), new Vector3(6, 1, 6), -Vector3.UnitZ);
            var c = SkAt(over, 32, 32);
            if (c.r < 245 || c.g < 245) throw new Exception($"over map pixel (0,0) there is a cloud: ({c.r},{c.g},{c.b})");
            // From x = 18: pixel (1, 0): clear sky
            var clear = await SkRender(device, queue, layer, new Vector3(18, 0, 6), new Vector3(18, 1, 6), -Vector3.UnitZ);
            var k = SkAt(clear, 32, 32);
            if (k.r > 10 || k.b < 245) throw new Exception($"over map pixel (1,0) the sky is clear: ({k.r},{k.g},{k.b})");
            // The layer slid 12 blocks along x: from x = 6 the pixel overhead is now the map's pixel (1, 0): clear
            var slid = layer; slid.Cloud.Y = 12f;
            var moved = await SkRender(device, queue, slid, new Vector3(6, 0, 6), new Vector3(6, 1, 6), -Vector3.UnitZ);
            var m = SkAt(moved, 32, 32);
            if (m.r > 10) throw new Exception($"after sliding a pixel along, the cloud is gone from overhead: ({m.r},{m.g},{m.b})");
            // A camera above the layer sees none of it
            var above = await SkRender(device, queue, layer, new Vector3(6, 20, 6), new Vector3(6, 21, 6), -Vector3.UnitZ);
            if (SkAt(above, 32, 32).r > 10) throw new Exception("from above the layer it is not drawn");
            // Rain: clouds over everything, even where the map has none
            var rain = layer; rain.Cloud.Z = 1f; rain.CloudColor = new Vector4(0.5f, 0.5f, 0.5f, 1f);
            var wet = await SkRender(device, queue, rain, new Vector3(18, 0, 6), new Vector3(18, 1, 6), -Vector3.UnitZ);
            var w = SkAt(wet, 32, 32);
            if (w.r < 100 || w.r > 160) throw new Exception($"in the rain the sky is cloud-grey even over a clear pixel: ({w.r},{w.g},{w.b})");
        });

        [TestMethod]
        public async Task Sky_IsDrawnBehindEverythingElseTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var (_, _, _, sun, moon, clouds) = SkAtlas();
            // The sun dead ahead and a cyan wall of a block in front of it: the wall shows, the sun does not
            var ahead = SkDay(sun, moon, clouds); ahead.SunDirection = new Vector4(0, 0, -1, 1);
            // Looking along +z, so the sun is ahead at +z
            ahead.SunDirection = new Vector4(0, 0, 1, 1);
            var px = await SkRender(device, queue, ahead, new Vector3(0.5f, 0.5f, -6f), new Vector3(0.5f, 0.5f, 0.5f), Vector3.UnitY, wall: true);
            string hue = ErHue(SkAt(px, 32, 32));
            if (hue != "GB") throw new Exception($"the wall (cyan) hides the sun behind it, the middle shows {hue}");
            // Around the wall the sun is there (a corner of the sun's square, off the block)
            var edge = SkAt(px, 35, 32);
            if (edge.r < 200 || edge.g < 200) throw new Exception($"the sun shows beside the wall: ({edge.r},{edge.g},{edge.b})");
        });
    }
}
