using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // A block that makes its own light (lava, glowstone, a torch) is as bright at midnight as at noon.
    public abstract partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task BlockRender_AnEmissiveBlockIsBrightInTheDark_AndAnotherIsNotTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[3];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            materials[2] = BlockMaterial.Uniform(BrLayerCyan, flags: BlockMaterial.FlagEmissive);
            var eye = new Vector3(0.5f, 2.5f, 0.5f);
            // Midnight: no sun, no ambient light, no fill
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            frame.Ambient = new Vector4(0, 0, 0, 0);
            var ordinary = await BrRender(device, queue, materials, new[]
            {
                new BrDraw { Pass = BlockRenderPipeline.Pass.Opaque, Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) } },
            }, frame, Vector3.Zero);
            BrExpect(ordinary, 64, 32, 32, 0, 0, 0, "an ordinary block is black in the dark");

            var glowing = await BrRender(device, queue, materials, new[]
            {
                new BrDraw { Pass = BlockRenderPipeline.Pass.Opaque, Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 2) } },
            }, frame, Vector3.Zero);
            // The top face's own brightness (1.05, capped at 1) times the texture (0, 255, 255)
            BrExpect(glowing, 64, 32, 32, 0, 255, 255, "an emissive block shows its colour in the dark");

            // A side face keeps its shading (the game does too): 0.88 and the side darkening of 0.85
            var side = await BrRender(device, queue, materials, new[]
            {
                new BrDraw { Pass = BlockRenderPipeline.Pass.Opaque, Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 2, 2) } },
            }, BrNeutralOf(frame, new Vector3(0.5f, 0.5f, 3f), new Vector3(0.5f, 0.5f, 1f)), Vector3.Zero);
            var (r, g, b, _) = GetPixel(side, 64, 32, 32);
            if (g < 150 || g > 240 || r > 10) throw new Exception($"an emissive side is bright but shaded: ({r},{g},{b})");

            // And in the water pass, over nothing: bright too (alpha 1)
            materials[2] = BlockMaterial.Uniform(BrLayerCyan, flags: BlockMaterial.FlagEmissive | BlockMaterial.FlagFlatShading, alpha: 1f);
            var water = await BrRender(device, queue, materials, new[]
            {
                new BrDraw { Pass = BlockRenderPipeline.Pass.Water, Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 2) } },
            }, frame, Vector3.Zero);
            BrExpect(water, 64, 32, 32, 0, 255, 255, "an emissive block in the water pass is bright in the dark");
        });

        private static BlockFrameUniforms BrNeutralOf(BlockFrameUniforms like, Vector3 eye, Vector3 target)
        {
            var frame = BrNeutral(BrView(eye, target, Vector3.UnitY) * BrProj90, eye);
            frame.Ambient = like.Ambient;
            return frame;
        }
    }
}
