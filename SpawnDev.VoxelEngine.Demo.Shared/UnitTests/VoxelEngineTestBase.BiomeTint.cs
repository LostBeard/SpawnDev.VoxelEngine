using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // BIOME TINT: blocks that take a biome colour (grass, foliage, water) are multiplied, per pixel, by the colour map's value at their place; the maps are one texel per 4 x 4 blocks, filtered
    // linearly (a border blends) and wrapping every 1,024 blocks.
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>
        /// Draw the top of column (cx, cz) - 16 x 16 blocks of one white-textured material - seen from 8 blocks straight above its middle (the view is exactly the column: 4 pixels a block), with the
        /// biome map as <paramref name="setup"/> leaves it, and read the picture back. Screen x grows with world x.
        /// </summary>
        private static async Task<byte[]> BtRender(GPUDevice device, GPUQueue queue, uint materialFlags, Func<BiomeTintMap, Task>? setup, int cx = 0, int cz = 0, Vector3? tint = null)
        {
            using var atlas = BlockTextureAtlas.FromRgba(device, queue, new byte[] { 255, 255, 255, 255 }, 1, 1, 1);
            using var blocks = new BlockRenderPipeline();
            blocks.Init(device, queue, "bgra8unorm", maxSections: 4);
            blocks.SetTextures(atlas);
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(0, tint ?? new Vector3(0.2f, 0.4f, 0.6f), materialFlags);
            blocks.SetMaterials(materials);
            blocks.SetSectionOrigin(0, new Vector3(cx * 16, 0, cz * 16));
            using var map = BiomeTintMap.Create(device, queue);
            if (setup != null) await setup(map);
            blocks.SetBiomeTints(map);

            var eye = new Vector3(cx * 16 + 8, 8.5f, cz * 16 + 8);
            var frame = BrNeutral(BrView(eye, new Vector3(eye.X, 0, eye.Z), -Vector3.UnitZ) * BrProj90, eye);
            blocks.BeginFrame(frame);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            var buffer = CreateQuadBuffer(device, queue, new[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1) });
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = blocks.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    blocks.BindPass(pass, BlockRenderPipeline.Pass.Opaque, buffer);
                    blocks.Draw(pass, 0, 1, 0);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                return await ReadBackPixels(device, queue, colorTex, 64, 64);
            }
            finally { buffer.Destroy(); buffer.Dispose(); DisposeTargets(colorTex, depthTex, colorView, depthView); }
        }

        // The pixel at world block x (within the column) along the middle row
        private static (byte r, byte g, byte b, byte a) BtAt(byte[] px, float blockX) => GetPixel(px, 64, (int)MathF.Round((blockX - 8f) / 8f * 32f + 31.5f), 32);

        private static uint[] BtFill(uint rgb) { var a = new uint[16]; System.Array.Fill(a, rgb); return a; }

        private static void BtExpect((byte r, byte g, byte b, byte a) got, int r, int g, int b, string what, int tolerance = 3)
        {
            if (Math.Abs(got.r - r) > tolerance || Math.Abs(got.g - g) > tolerance || Math.Abs(got.b - b) > tolerance)
                throw new Exception($"{what}: ({got.r},{got.g},{got.b}), expected ({r},{g},{b}) +-{tolerance}");
        }

        [TestMethod]
        public async Task BiomeTint_UntouchedMaps_ShowPlainsColours_ForEachKindTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var grass = await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, null);
            BtExpect(BtAt(grass, 8), 0x91, 0xBD, 0x59, "grass over plains");
            var foliage = await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintFoliage, null);
            BtExpect(BtAt(foliage, 8), 0x77, 0xAB, 0x2F, "foliage over plains");
            var water = await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintWater, null);
            BtExpect(BtAt(water, 8), 0x3F, 0x76, 0xE4, "water over plains");
        });

        [TestMethod]
        public async Task BiomeTint_AColumnsColours_TintOnlyThatColumn_AndEachKindReadsItsOwnMapTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            Task Paint(BiomeTintMap m) { m.SetColumn(0, 0, BtFill(0xFF0000), BtFill(0x00FF00), BtFill(0x0000FF)); return Task.CompletedTask; }
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, Paint), 8), 255, 0, 0, "column (0,0) grass is red");
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintFoliage, Paint), 8), 0, 255, 0, "column (0,0) foliage is green");
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintWater, Paint), 8), 0, 0, 255, "column (0,0) water is blue");
            // The next column over still has plains
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, Paint, cx: 1), 8), 0x91, 0xBD, 0x59, "column (1,0) is untouched");
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, Paint, cz: 1), 8), 0x91, 0xBD, 0x59, "column (0,1) is untouched");
        });

        [TestMethod]
        public async Task BiomeTint_ABorder_BlendsSmoothly_AndEachCellKeepsItsOwnColourAtItsCentreTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Cells (i, j) at j * 4 + i, i along x: the left two cell columns red, the right two blue. Cell centres are 2, 6, 10, 14 blocks across
            Task Paint(BiomeTintMap m)
            {
                var g = new uint[16];
                for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) g[j * 4 + i] = i < 2 ? 0xFF0000u : 0x0000FFu;
                m.SetColumn(0, 0, g, g, g);
                return Task.CompletedTask;
            }
            var px = await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, Paint);
            BtExpect(BtAt(px, 2), 255, 0, 0, "block x 2 (a cell centre)", 8);
            BtExpect(BtAt(px, 14), 0, 0, 255, "block x 14 (a cell centre)", 8);
            BtExpect(BtAt(px, 8), 128, 0, 128, "block x 8 (between two cell centres): half and half", 14);
            var a = BtAt(px, 6.5f); var b = BtAt(px, 8f); var c = BtAt(px, 9.5f);
            if (!(a.r > b.r && b.r > c.r)) throw new Exception($"red should fade across the border: {a.r} {b.r} {c.r}");
            if (!(a.b < b.b && b.b < c.b)) throw new Exception($"blue should grow across the border: {a.b} {b.b} {c.b}");
        });

        [TestMethod]
        public async Task BiomeTint_NegativeColumns_AndTheWrap_LandWhereTheyShouldTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            Task Negative(BiomeTintMap m) { m.SetColumn(-1, -1, BtFill(0x00FF00), BtFill(0x00FF00), BtFill(0x00FF00)); return Task.CompletedTask; }
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, Negative, cx: -1, cz: -1), 8), 0, 255, 0, "column (-1,-1) is green");
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, Negative, cx: 0, cz: 0), 8), 0x91, 0xBD, 0x59, "column (0,0) is not touched by column (-1,-1)");
            // The maps wrap every 64 columns (1,024 blocks): column 64 is column 0's texels
            Task Wrapped(BiomeTintMap m) { m.SetColumn(64, 0, BtFill(0xFF0000), BtFill(0xFF0000), BtFill(0xFF0000)); return Task.CompletedTask; }
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, Wrapped, cx: 64), 8), 255, 0, 0, "column (64,0) shows what was written to it");
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted | BlockMaterial.TintGrass, Wrapped, cx: 0), 8), 255, 0, 0, "column (0,0) shares the texels of column (64,0)");
            if (BiomeTintMap.CellOf(-1) != 255 || BiomeTintMap.CellOf(-4) != 255 || BiomeTintMap.CellOf(-5) != 254 || BiomeTintMap.CellOf(1024) != 0 || BiomeTintMap.CellOf(1023) != 255)
                throw new Exception("CellOf wraps wrongly");
        });

        [TestMethod]
        public async Task BiomeTint_AMaterialWithoutAKind_KeepsItsOwnColour_AndTheMapNeverTouchesIt() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            Task Paint(BiomeTintMap m) { m.SetColumn(0, 0, BtFill(0xFF0000), BtFill(0xFF0000), BtFill(0xFF0000)); return Task.CompletedTask; }
            // A fixed tint (birch leaves): the material's colour, whatever the biome
            BtExpect(BtAt(await BtRender(device, queue, BlockMaterial.FlagTinted, Paint, tint: new Vector3(0.2f, 0.4f, 0.6f)), 8), 54, 107, 161, "a fixed tint ignores the map (the top face brightens it by 1.05)");
            // Not tinted at all: white texture as it is
            BtExpect(BtAt(await BtRender(device, queue, 0u, Paint), 8), 255, 255, 255, "an untinted block ignores the map");
        });
    }
}
