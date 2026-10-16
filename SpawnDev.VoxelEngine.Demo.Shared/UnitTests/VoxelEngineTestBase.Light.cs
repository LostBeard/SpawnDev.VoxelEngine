using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // LIGHT: each pixel lit by the sky light and block light of the cell in front of its face, from a LightVolume.
    public abstract partial class VoxelEngineTestBase
    {
        // The top face of the block at (0, 0, 0) is lit by the cell above it: (0, 1, 0), at y + 64 = 65 of column (0, 0)
        private const int LtCellIndex = 65 * 256;

        private static byte[] LtColumn(byte atTheCell, byte elsewhere = 0xF0)
        {
            var data = new byte[LightVolume.ColumnBytes];
            System.Array.Fill(data, elsewhere);
            data[LtCellIndex] = atTheCell;
            return data;
        }

        // The cyan top face under 50% ambient light, seen from above, with a light volume set up by the test (null: the pipeline's own, off)
        private static async Task<(byte r, byte g, byte b, byte a)> LtRender(GPUDevice device, GPUQueue queue, Action<LightVolume>? setup, float darkFloor = 0.12f)
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            var eye = new Vector3(0.5f, 2.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            frame.Ambient = new Vector4(0.5f, 0.5f, 0.5f, 0f);
            using var textures = BlockTextureAtlas.FromRgba(device, queue, BrAtlas(), 32, 32, BrTile);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            using var light = new LightVolume();
            if (setup != null)
            {
                light.Init(device, queue, 4, 3);
                light.DarkFloor = darkFloor;
                setup(light);
                pipeline.SetLight(light);
            }
            pipeline.BeginFrame(frame);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            GPUBuffer? buffer = null;
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    buffer = CreateQuadBuffer(device, queue, new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) });
                    pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, buffer);
                    pipeline.Draw(pass, 0, 1, 0);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var pixels = await ReadBackPixels(device, queue, colorTex, 64, 64);
                return GetPixel(pixels, 64, 32, 32);
            }
            finally
            {
                if (buffer != null) { buffer.Destroy(); buffer.Dispose(); }
                DisposeTargets(colorTex, depthTex, colorView, depthView);
            }
        }

        private static void LtExpect((byte r, byte g, byte b, byte a) got, int r, int g, int b, string what, int tolerance = 6)
        {
            if (Math.Abs(got.r - r) > tolerance || Math.Abs(got.g - g) > tolerance || Math.Abs(got.b - b) > tolerance)
                throw new Exception($"{what}: ({got.r},{got.g},{got.b}), expected ({r},{g},{b}) +-{tolerance}");
        }

        [TestMethod]
        public async Task Light_SkyLightDimsAndBlockLightWarms_ByTheCellInFrontOfTheFaceTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // No light volume: the world as it was (cyan under half ambient: 0, 128, 128)
            LtExpect(await LtRender(device, queue, null), 0, 128, 128, "no light volume");

            // The cell has open sky and no block light: the same
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); v.Write(v.Assign(0, 0), LtColumn(0xF0)); }), 0, 128, 128, "open sky");

            // No sky, no block light: down to the floor (0.12 of the light)
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); v.Write(v.Assign(0, 0), LtColumn(0x00)); }), 0, 15, 15, "dark");
            // and a floor of its own
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); v.Write(v.Assign(0, 0), LtColumn(0x00)); }, darkFloor: 0.5f), 0, 64, 64, "dark with a higher floor");

            // Sky light 8 of 15: the game's curve gives 0.8205, so 0.12 + 0.88 * 0.8205 = 0.842 of the light
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); v.Write(v.Assign(0, 0), LtColumn(0x80)); }), 0, 108, 108, "sky light 8");

            // Block light 15 in the dark: the warm colour of a torch, (1, 0.84, 0.6), times the cyan
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); v.Write(v.Assign(0, 0), LtColumn(0x0F)); }), 0, 214, 153, "block light 15");
            // Block light 7: dimmer: levelBrightness(7) = 1 - 0.5333 / 2.4 = 0.7778
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); v.Write(v.Assign(0, 0), LtColumn(0x07)); }), 0, 214 * 778 / 1000, 153 * 778 / 1000, "block light 7", 8);
            // Both: the brighter of the two lights, per channel (sky 15 gives 128 across; the torch gives 214 green, 153 blue)
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); v.Write(v.Assign(0, 0), LtColumn(0xFF)); }), 0, 214, 153, "sky and block both full");

            // A column the volume has no slot for is lit as open sky
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); }), 0, 128, 128, "no slot for the column");
            // A window that does not take in the column
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(10, 10); v.Assign(0, 0); }), 0, 128, 128, "outside the window");
            // Switched off, though the data is there
            LtExpect(await LtRender(device, queue, v => { v.SetWindow(0, 0); v.Write(v.Assign(0, 0), LtColumn(0x00)); v.Enabled = false; }), 0, 128, 128, "switched off");
        });

        [TestMethod]
        public async Task Light_AColumnKeepsItsSlot_UntilTheWindowMovesPastItTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var v = new LightVolume();
            v.Init(device, queue, 2, 3);
            v.SetWindow(0, 0);
            int a = v.Assign(0, 0), b = v.Assign(1, 0);
            if (a < 0 || b < 0 || a == b) throw new Exception($"two columns, two slots: {a}, {b}");
            if (v.Assign(0, 0) != a) throw new Exception("assigning again gives the same slot");
            if (v.Assign(-1, 0) != -1) throw new Exception("no slot is left");
            if (v.Count != 2) throw new Exception($"two columns have slots, not {v.Count}");
            // Move the window east by two: column 0 falls out of it, column 1 stays (window 3 wide: centre 2 covers columns 1..3)
            var evicted = new List<(int, int)>();
            v.SetWindow(2, 0, evicted);
            if (evicted.Count != 1 || evicted[0] != (0, 0)) throw new Exception($"column 0 is evicted: {string.Join(",", evicted)}");
            if (v.Has(0, 0) || !v.Has(1, 0)) throw new Exception("column 1 keeps its slot, column 0 does not");
            // The freed slot is there for the next column
            if (v.Assign(3, 0) != a) throw new Exception("the freed slot is used again");
            if (v.Assign(0, 0) != -1) throw new Exception("a column outside the window gets none");
            v.Release(1, 0);
            if (v.Count != 1) throw new Exception("released");
        });
    }
}
