using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // RAIN AND SNOW: a streak for each column of a window round the camera, from the column's ground to a little above the camera.
    public abstract partial class VoxelEngineTestBase
    {
        private const int WxSize = 64;

        // An atlas for the tests: the rain is an opaque white 2 x 2, the snow an opaque red 2 x 2 beside it
        private static (byte[] rgba, int w, int h) WxAtlas()
        {
            const int w = 4, h = 2;
            var px = new byte[w * h * 4];
            ErPaint(px, w, 0, 0, 2, 2, 255, 255, 255);
            ErPaint(px, w, 2, 0, 2, 2, 255, 0, 0);
            return (px, w, h);
        }

        private static uint WxCell(uint kind, int groundY) => (kind << 16) | (uint)(groundY + 64);

        // A 12 x 12 window of columns, all of the same kind and ground, round a camera at (0.5, 0.5, 0.5) looking along +x
        private static async Task<byte[]> WxRender(GPUDevice device, GPUQueue queue, uint kind, int groundY, float strength = 1f, Func<int, int, uint>? cell = null)
        {
            var (rgba, aw, ah) = WxAtlas();
            using var atlas = EntityTextureAtlas.FromRgba(device, queue, rgba, aw, ah);
            using var blocks = new BlockRenderPipeline();
            blocks.Init(device, queue, "bgra8unorm", maxSections: 4);
            using var weather = new WeatherPipeline();
            weather.Init(device, queue, "bgra8unorm");
            weather.SetTextures(atlas);
            const int n = 12;
            var cells = new uint[n * n];
            for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                    cells[z * n + x] = cell != null ? cell(x, z) : WxCell(kind, groundY);
            weather.SetCells(cells, n);
            var eye = new Vector3(0.5f, 0.5f, 0.5f);
            var viewProj = BrView(eye, eye + Vector3.UnitX, Vector3.UnitY) * BrProj90;
            blocks.BeginFrame(BrNeutral(viewProj, eye));
            var uniforms = new WeatherUniforms
            {
                ViewProjection = viewProj,
                Camera = new Vector4(eye, 40f),
                Params = new Vector4(strength, 1f, n, 8f),
                Window = new Vector4(-5, -5, 0, 0),
                RainRect = new Vector4(0, 0, 2, 2),
                SnowRect = new Vector4(2, 0, 2, 2),
            };
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, WxSize, WxSize);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = blocks.BeginRenderPass(encoder, colorView, depthView, new Vector3(0, 0, 0)))
                {
                    weather.Draw(pass, in uniforms);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                return await ReadBackPixels(device, queue, colorTex, WxSize, WxSize);
            }
            finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
        }

        private static (int white, int red, int total) WxCount(byte[] px)
        {
            int white = 0, red = 0;
            for (int y = 0; y < WxSize; y++)
                for (int x = 0; x < WxSize; x++)
                {
                    var p = GetPixel(px, WxSize, x, y);
                    if (p.r > 40 && p.g > 40 && p.b > 40) white++;
                    else if (p.r > 40 && p.g < 30 && p.b < 30) red++;
                }
            return (white, red, WxSize * WxSize);
        }

        [TestMethod]
        public async Task Weather_RainStreaksFallInTheColumnsRoundTheCamera_SnowIsItsOwnTextureTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var rain = WxCount(await WxRender(device, queue, WeatherPipeline.KindRain, groundY: -20));
            if (rain.white < 100) throw new Exception($"rain over open ground fills the view with streaks: {rain.white} white pixels");
            if (rain.red != 0) throw new Exception($"rain is not the snow texture: {rain.red} red pixels");
            var snow = WxCount(await WxRender(device, queue, WeatherPipeline.KindSnow, groundY: -20));
            if (snow.red < 100) throw new Exception($"snow over open ground fills the view with flakes: {snow.red} red pixels");
            if (snow.white != 0) throw new Exception($"snow is not the rain texture: {snow.white} white pixels");
        });

        [TestMethod]
        public async Task Weather_FallsNowhereWhereTheGroundIsAboveTheViewOrNothingFallsOrItIsOffTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // A roof high over the camera (ground above the streak's top: 0.5 + 8): nothing falls under it
            var roofed = WxCount(await WxRender(device, queue, WeatherPipeline.KindRain, groundY: 40));
            if (roofed.white != 0) throw new Exception($"nothing falls where the ground is above the view: {roofed.white}");
            var none = WxCount(await WxRender(device, queue, WeatherPipeline.KindNone, groundY: -20));
            if (none.white != 0 || none.red != 0) throw new Exception("a column that nothing falls on is empty");
            var off = WxCount(await WxRender(device, queue, WeatherPipeline.KindRain, groundY: -20, strength: 0f));
            if (off.white != 0) throw new Exception($"no strength, no rain: {off.white}");
        });

        [TestMethod]
        public async Task Weather_AColumnIsRainOrSnowOnItsOwn_AndTheFarOnesFadeTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Window x columns -5..6: the ones with x < 6 (world x < 1) rain, the rest snow: both appear
            var mixed = WxCount(await WxRender(device, queue, 0, 0, cell: (x, z) => z < 6 ? WxCell(WeatherPipeline.KindRain, -20) : WxCell(WeatherPipeline.KindSnow, -20)));
            if (mixed.white < 30 || mixed.red < 30) throw new Exception($"a view over both shows both: {mixed.white} white, {mixed.red} red");
            // Weaker is thinner: the streaks are less opaque, so the view is darker (many streaks overlap, so count the brightness, not the pixels)
            double Mean(byte[] px) { double sum = 0; for (int y = 0; y < WxSize; y++) for (int x = 0; x < WxSize; x++) sum += GetPixel(px, WxSize, x, y).r; return sum / (WxSize * WxSize); }
            double strong = Mean(await WxRender(device, queue, WeatherPipeline.KindRain, groundY: -20, strength: 1f));
            double weak = Mean(await WxRender(device, queue, WeatherPipeline.KindRain, groundY: -20, strength: 0.15f));
            if (weak >= strong - 10) throw new Exception($"weaker rain is darker: {weak:F1} against {strong:F1}");
        });
    }
}
