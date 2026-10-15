using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // BlockRenderPipeline: textured, tinted, lit, fogged blocks from PackedQuad buffers, on a real WebGPU device with pixel readback.
    // Lighting in these tests is NEUTRAL (ambient 1, no sun, no fill, no fog) unless a test is about lighting, so a pixel's value is
    // the texel times the known per-face brightness: sides 0.88|0.82 x 0.85, top 1.0 (1.05 clamps), bottom 0.70 x 0.70.
    public abstract partial class VoxelEngineTestBase
    {
        // ---- test textures: a 32x32 atlas of four 16x16 tiles (layer = row * 2 + column) ---------------------------------------
        // layer 0: quadrants TL red, TR green, BL blue, BR yellow (8x8 each)  - tells a mirrored or rotated face from a right one
        // layer 1: solid cyan   layer 2: left half transparent, right half white   layer 3: solid orange
        private const int BrTile = 16;
        private const int BrLayerQuadrants = 0, BrLayerCyan = 1, BrLayerHalfClear = 2, BrLayerOrange = 3;

        private static byte[] BrAtlas()
        {
            var px = new byte[32 * 32 * 4];
            void Set(int x, int y, byte r, byte g, byte b, byte a)
            {
                int i = (y * 32 + x) * 4;
                px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = a;
            }
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                {
                    // tile 0 (atlas x 0..15, y 0..15): quadrants
                    if (y < 8) Set(x, y, x < 8 ? (byte)255 : (byte)0, x < 8 ? (byte)0 : (byte)255, 0, 255);
                    else Set(x, y, x < 8 ? (byte)0 : (byte)255, x < 8 ? (byte)0 : (byte)255, x < 8 ? (byte)255 : (byte)0, 255);
                    // tile 1 (x 16..31, y 0..15): cyan
                    Set(16 + x, y, 0, 255, 255, 255);
                    // tile 2 (x 0..15, y 16..31): left half transparent, right half white
                    if (x < 8) Set(x, 16 + y, 255, 0, 255, 0); else Set(x, 16 + y, 255, 255, 255, 255);
                    // tile 3 (x 16..31, y 16..31): orange
                    Set(16 + x, 16 + y, 255, 128, 0, 255);
                }
            return px;
        }

        private static float BrFaceFactor(int face) => face switch
        {
            0 or 1 => 0.88f * 0.85f,
            2 or 3 => 0.82f * 0.85f,
            4 => 1.0f,                 // 1.05 clamps to 1 on a white base
            _ => 0.70f * 0.70f,        // bottom
        };

        /// <summary>Neutral lighting: ambient 1, no sun, no fill light, no fog.</summary>
        private static BlockFrameUniforms BrNeutral(Matrix4x4 viewProj, Vector3 camera) => new()
        {
            ViewProjection = viewProj,
            CameraPosition = new Vector4(camera, 1),
            SunDirection = new Vector4(0, 1, 0, 0),
            SunColor = new Vector4(0, 0, 0, 0),
            Ambient = new Vector4(1, 1, 1, 0),
            FogColor = new Vector4(0, 0, 0, 0),
            FogRange = new Vector4(1e9f, 2e9f, 0, 0),
        };

        private static Matrix4x4 BrView(Vector3 eye, Vector3 target, Vector3 up) => Matrix4x4.CreateLookAt(eye, target, up);
        private static Matrix4x4 BrProj90 => ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.05f);

        private sealed class BrDraw
        {
            public BlockRenderPipeline.Pass Pass = BlockRenderPipeline.Pass.Opaque;
            public long[] Quads = System.Array.Empty<long>();
            public int Section;
        }

        /// <summary>Atlas textures + materials into a fresh pipeline, render the draws into a size x size target, read it back (BGRA).</summary>
        private static async Task<byte[]> BrRender(GPUDevice device, GPUQueue queue, BlockMaterial[] materials, BrDraw[] draws,
            BlockFrameUniforms frame, Vector3 clear, int size = 64, Vector3[]? sectionOrigins = null)
        {
            using var textures = BlockTextureArray.FromAtlas(device, queue, BrAtlas(), 32, 32, BrTile);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 16);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            if (sectionOrigins != null) pipeline.SetSectionOrigins(0, sectionOrigins);
            pipeline.BeginFrame(frame);

            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, size, size);
            var buffers = new List<GPUBuffer>();
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, clear);
                foreach (var d in draws)
                {
                    var buffer = CreateQuadBuffer(device, queue, d.Quads);
                    buffers.Add(buffer);
                    pipeline.BindPass(pass, d.Pass, buffer);
                    pipeline.Draw(pass, 0, d.Quads.Length, d.Section);
                }
                pass.End();
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                return await ReadBackPixels(device, queue, colorTex, size, size);
            }
            finally
            {
                foreach (var b in buffers) { b.Destroy(); b.Dispose(); }
                DisposeTargets(colorTex, depthTex, colorView, depthView);
            }
        }

        private static void BrExpect(byte[] pixels, int size, int x, int y, float r, float g, float b, string what, int tolerance = 6)
        {
            var (pr, pg, pb, _) = GetPixel(pixels, size, x, y);
            int er = (int)MathF.Round(r), eg = (int)MathF.Round(g), eb = (int)MathF.Round(b);
            if (Math.Abs(pr - er) > tolerance || Math.Abs(pg - eg) > tolerance || Math.Abs(pb - eb) > tolerance)
                throw new Exception($"{what}: pixel ({x},{y}) is ({pr},{pg},{pb}), expected ({er},{eg},{eb}) +-{tolerance}");
        }

        // =====================================================================
        // Struct layouts (every backend)
        // =====================================================================

        [TestMethod]
        public void BlockRender_StructSizes_MatchTheShaderTest()
        {
            if (Marshal.SizeOf<BlockMaterial>() != 32) throw new Exception($"BlockMaterial is {Marshal.SizeOf<BlockMaterial>()} bytes, the WGSL struct is 32");
            if (Marshal.SizeOf<BlockFrameUniforms>() != BlockFrameUniforms.SizeInBytes)
                throw new Exception($"BlockFrameUniforms is {Marshal.SizeOf<BlockFrameUniforms>()} bytes, the WGSL struct is {BlockFrameUniforms.SizeInBytes}");
            var m = BlockMaterial.Create(1, 2, 3, 4, 5, 6);
            if (m.Layers01 != (1u | (2u << 16)) || m.Layers23 != (3u | (4u << 16)) || m.Layers45 != (5u | (6u << 16)))
                throw new Exception($"Face layers pack wrongly: {m.Layers01:X8} {m.Layers23:X8} {m.Layers45:X8}");
        }

        // =====================================================================
        // Texture orientation: every face, seen from outside, shows the texture upright and unmirrored
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_EveryFace_ShowsTheTextureUprightAndUnmirroredTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerQuadrants);
            // face, eye, target, screen-up: one block at the origin, the camera 0.5 in front of the face centre (a 90 degree view then shows exactly the face)
            var cases = new (int face, Vector3 eye, Vector3 target, Vector3 up)[]
            {
                (0, new(1.5f, 0.5f, 0.5f), new(1f, 0.5f, 0.5f), Vector3.UnitY),
                (1, new(-0.5f, 0.5f, 0.5f), new(0f, 0.5f, 0.5f), Vector3.UnitY),
                (2, new(0.5f, 0.5f, 1.5f), new(0.5f, 0.5f, 1f), Vector3.UnitY),
                (3, new(0.5f, 0.5f, -0.5f), new(0.5f, 0.5f, 0f), Vector3.UnitY),
                (4, new(0.5f, 1.5f, 0.5f), new(0.5f, 1f, 0.5f), -Vector3.UnitZ),
                (5, new(0.5f, -0.5f, 0.5f), new(0.5f, 0f, 0.5f), Vector3.UnitZ),
            };
            foreach (var (face, eye, target, up) in cases)
            {
                var frame = BrNeutral(BrView(eye, target, up) * BrProj90, eye);
                var pixels = await BrRender(device, queue, materials,
                    new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, face, 1) } } },
                    frame, Vector3.Zero);
                float f = 255f * BrFaceFactor(face);
                string w = $"face {face}";
                BrExpect(pixels, 64, 16, 16, f, 0, 0, w + " top-left should be the red quadrant");
                BrExpect(pixels, 64, 48, 16, 0, f, 0, w + " top-right should be the green quadrant");
                BrExpect(pixels, 64, 16, 48, 0, 0, f, w + " bottom-left should be the blue quadrant");
                BrExpect(pixels, 64, 48, 48, f, f, 0, w + " bottom-right should be the yellow quadrant");
            }
        });

        // =====================================================================
        // A greedy-merged quad tiles its texture once per block
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_MergedQuad_TilesTheTexturePerBlockTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerQuadrants);
            // one 4x4 quad of +Y; the camera 2 above its top, 90 degrees: 4 blocks across 64 px, 16 px per block, the texture's quadrants 8 px
            var eye = new Vector3(2f, 3f, 2f);
            var frame = BrNeutral(BrView(eye, new Vector3(2f, 1f, 2f), -Vector3.UnitZ) * BrProj90, eye);
            var pixels = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 4, 4, 4, 1) } } }, frame, Vector3.Zero);

            BrExpect(pixels, 64, 4, 4, 255, 0, 0, "block (0,0), red quadrant");
            BrExpect(pixels, 64, 52, 52, 255, 0, 0, "block (3,3), red quadrant: the texture repeats");
            BrExpect(pixels, 64, 28, 44, 255, 255, 0, "block (1,2), yellow quadrant");
            BrExpect(pixels, 64, 60, 4, 0, 255, 0, "block (3,0), green quadrant");
            BrExpect(pixels, 64, 4, 60, 0, 0, 255, "block (0,3), blue quadrant (left half, bottom half)");
        });

        // =====================================================================
        // Alpha cutout
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_AlphaCutout_DiscardsTransparentTexelsTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerHalfClear);
            var eye = new Vector3(0.5f, 1.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            var pixels = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) } } }, frame, new Vector3(0, 0, 1));

            BrExpect(pixels, 64, 16, 32, 0, 0, 255, "the transparent left half shows the clear colour (blue) through the quad");
            BrExpect(pixels, 64, 48, 32, 255, 255, 255, "the opaque right half is white");
        });

        // =====================================================================
        // Water pass: alpha blended over what is behind it
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_WaterPass_BlendsWithTheAlphaOverTheSceneTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[3];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            materials[2] = BlockMaterial.Uniform(BrLayerOrange, flags: BlockMaterial.FlagFlatShading, alpha: 0.6f);
            var eye = new Vector3(0.5f, 2.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            frame.Ambient = new Vector4(0.5f, 0.5f, 0.5f, 0); // water lighting is ambient x 1.2 = 0.6
            // opaque cyan top at y=1; water top (one block) at y=1 too would z-fight, so the water block sits one up: its top face at y=2
            var pixels = await BrRender(device, queue, materials, new[]
            {
                new BrDraw { Pass = BlockRenderPipeline.Pass.Opaque, Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) } },
                new BrDraw { Pass = BlockRenderPipeline.Pass.Water, Quads = new[] { PackedQuad.Pack(0, 1, 0, 1, 1, 4, 2) } },
            }, frame, Vector3.Zero);

            // opaque cyan: (0, 255, 255) x ambient 0.5 = (0, 128, 128). water: orange (255, 128, 0) x 0.6 lit = (153, 77, 0), alpha 0.6 over it
            float dst = 255f * 0.5f;
            BrExpect(pixels, 64, 32, 32, 153 * 0.6f, 77 * 0.6f + dst * 0.4f, dst * 0.4f, "water blended 0.6 over the opaque cyan", tolerance: 9);

            // and with nothing behind it, over the clear colour
            var over = await BrRender(device, queue, materials, new[]
            {
                new BrDraw { Pass = BlockRenderPipeline.Pass.Water, Quads = new[] { PackedQuad.Pack(0, 1, 0, 1, 1, 4, 2) } },
            }, frame, Vector3.Zero);
            BrExpect(over, 64, 32, 32, 153 * 0.6f, 77 * 0.6f, 0, "water over black", tolerance: 9);
        });

        [TestMethod]
        public async Task BlockRender_WaterPass_DoesNotWriteDepthTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Water drawn FIRST, opaque after it, from above: the opaque block (below the water) is still drawn into the pixels the water
            // covers, because water does not write depth. (Were depth written, the opaque block behind it would be rejected.)
            var materials = new BlockMaterial[3];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            materials[2] = BlockMaterial.Uniform(BrLayerOrange, flags: BlockMaterial.FlagFlatShading, alpha: 0.5f);
            var eye = new Vector3(0.5f, 2.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            var pixels = await BrRender(device, queue, materials, new[]
            {
                new BrDraw { Pass = BlockRenderPipeline.Pass.Water, Quads = new[] { PackedQuad.Pack(0, 1, 0, 1, 1, 4, 2) } },
                new BrDraw { Pass = BlockRenderPipeline.Pass.Opaque, Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) } },
            }, frame, Vector3.Zero);
            var (r, g, b, _) = GetPixel(pixels, 64, 32, 32);
            if (b < 60) throw new Exception($"The opaque block behind the water is missing ({r},{g},{b}): the water pass wrote depth");
        });

        [TestMethod]
        public async Task BlockRender_WaterTop_SitsATenthBelowTheBlockTopTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // A water block's top face is 0.1 below the block's top. Seen from a slant, the face's far edge lands on a screen row we can
            // PROJECT: render the top with and without FlagLowerTop and find its first lit row; each must match the projected row of
            // the far edge at y = 1.0 (plain) and y = 0.9 (lowered), and the two rows must differ.
            var eye = new Vector3(0.5f, 3f, -1.5f);
            var vp = BrView(eye, new Vector3(0.5f, 1f, 0.5f), Vector3.UnitY) * BrProj90;
            var frame = BrNeutral(vp, eye);
            var draw = new[] { new BrDraw { Pass = BlockRenderPipeline.Pass.Water, Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) } } };

            // 256 px: at 64 px the two surfaces project less than a pixel apart and no test could tell them apart
            const int size = 256;
            float ProjectedRow(float y)
            {
                var clip = Vector4.Transform(new Vector4(0.5f, y, 1f, 1f), vp); // the far edge (z = 1) of the top face, mid-width
                return (1f - clip.Y / clip.W) * 0.5f * size;
            }
            int FirstLitRow(byte[] px)
            {
                for (int y = 0; y < size; y++) { var (r, g, b, _) = GetPixel(px, size, size / 2, y); if (r > 40) return y; }
                return -1;
            }

            var plainMaterial = new BlockMaterial[2];
            plainMaterial[1] = BlockMaterial.Uniform(BrLayerOrange, flags: BlockMaterial.FlagFlatShading, alpha: 1f);
            var loweredMaterial = new BlockMaterial[2];
            loweredMaterial[1] = BlockMaterial.Uniform(BrLayerOrange, flags: BlockMaterial.FlagFlatShading | BlockMaterial.FlagLowerTop, alpha: 1f);

            int plainRow = FirstLitRow(await BrRender(device, queue, plainMaterial, draw, frame, Vector3.Zero, size));
            int loweredRow = FirstLitRow(await BrRender(device, queue, loweredMaterial, draw, frame, Vector3.Zero, size));
            float expectPlain = ProjectedRow(1.0f), expectLowered = ProjectedRow(0.9f);

            if (Math.Abs(plainRow - expectPlain) > 1.5f)
                throw new Exception($"Without the flag the far edge's first lit row is {plainRow}, projected {expectPlain:F1}");
            if (Math.Abs(loweredRow - expectLowered) > 1.5f)
                throw new Exception($"With FlagLowerTop the far edge's first lit row is {loweredRow}, projected {expectLowered:F1} for y = 0.9");
            if (Math.Abs(expectPlain - expectLowered) < 2f)
                throw new Exception("The test geometry cannot tell 1.0 from 0.9 apart: the projected rows are " + $"{expectPlain:F1} and {expectLowered:F1}");
        });

        // =====================================================================
        // Plants: two crossed planes, visible from both sides
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_CrossPlants_AreDoubleSidedTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            var quads = new[]
            {
                PackedQuad.Pack(0, 0, 0, 1, 1, BlockRenderShaders.FaceCrossA, 1),
                PackedQuad.Pack(0, 0, 0, 1, 1, BlockRenderShaders.FaceCrossB, 1),
            };
            // Cyan x plant brightness 1.0 x the vertical-ish normal step 0.85 = (0, 217, 217)
            foreach (var eye in new[] { new Vector3(-1.5f, 0.5f, 0.5f), new Vector3(2.5f, 0.5f, 0.5f), new Vector3(0.5f, 0.5f, -1.5f), new Vector3(0.5f, 0.5f, 2.5f) })
            {
                var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 0.5f, 0.5f), Vector3.UnitY) * BrProj90, eye);
                var pixels = await BrRender(device, queue, materials,
                    new[] { new BrDraw { Pass = BlockRenderPipeline.Pass.Cross, Quads = quads } }, frame, Vector3.Zero);
                BrExpect(pixels, 64, 32, 32, 0, 255 * 0.85f, 255 * 0.85f, $"plant seen from {eye}", tolerance: 8);
            }
        });

        // =====================================================================
        // Section placement through the instance index
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_SectionOrigins_PlaceEachSectionTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            // One quad buffer, drawn as section 0 (origin 0,0,0) and section 3 (origin 4,0,0): two blocks 4 apart, nothing between
            var eye = new Vector3(3f, 5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(3f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            var origins = new[] { new Vector3(0, 0, 0), new Vector3(9, 9, 9), new Vector3(9, 9, 9), new Vector3(4, 0, 0) };
            var quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) };
            using var textures = BlockTextureArray.FromAtlas(device, queue, BrAtlas(), 32, 32, BrTile);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 16);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            pipeline.SetSectionOrigins(0, origins);
            pipeline.BeginFrame(frame);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            using var buffer = CreateQuadBuffer(device, queue, quads);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero);
                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, buffer);
                pipeline.Draw(pass, 0, 1, 0);
                pipeline.Draw(pass, 0, 1, 3);
                pass.End();
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var pixels = await ReadBackPixels(device, queue, colorTex, 64, 64);
                // The camera is 4 above the tops, so x spans -1..7 over 64 px: 8 px per block. Block x 0..1 -> px 8..16, x 4..5 -> px 40..48.
                BrExpect(pixels, 64, 12, 32, 0, 255, 255, "section 0's block");
                BrExpect(pixels, 64, 44, 32, 0, 255, 255, "section 3's block, 4 blocks over");
                BrExpect(pixels, 64, 28, 32, 0, 0, 0, "nothing between them (x 2..3)");
                BrExpect(pixels, 64, 60, 32, 0, 0, 0, "and no block at the (9,9,9) origins that sections 1 and 2 were given: they were never drawn");
            }
            finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
        });

        // =====================================================================
        // Materials: tint, flat colour, missing
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_Materials_TintFlatColourAndMissingTypeTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[4];
            materials[1] = BlockMaterial.Uniform(BrLayerOrange, tint: new Vector3(0f, 1f, 0.5f), flags: BlockMaterial.FlagTinted);
            materials[2] = BlockMaterial.Uniform(BlockMaterial.NoTexture, tint: new Vector3(0.2f, 0.4f, 0.6f));
            materials[3] = BlockMaterial.Uniform(BrLayerOrange, tint: new Vector3(0f, 1f, 0f)); // not tinted: the tint must NOT apply
            // Three +Y blocks side by side (types 1, 2, 3), seen from 1.5 above their tops: the view is 3 blocks wide, 21.3 px each
            var eye = new Vector3(1.5f, 2.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(1.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            var quads = new[]
            {
                PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1), PackedQuad.Pack(1, 0, 0, 1, 1, 4, 2), PackedQuad.Pack(2, 0, 0, 1, 1, 4, 3),
            };
            var pixels = await BrRender(device, queue, materials, new[] { new BrDraw { Quads = quads } }, frame, Vector3.Zero, size: 64);
            // A top face is brightened x1.05 before the texture (clamped to 1): a tint of (0,1,0.5) becomes (0,1,0.525)
            BrExpect(pixels, 64, 10, 32, 0, 255 * 0.5f, 0, "tinted orange (255,128,0) x tint (0,1,0.525) = (0,128,0)");
            // An untextured block IS its tint colour, x1.05 on a top face: (0.2,0.4,0.6) -> (0.21,0.42,0.63)
            BrExpect(pixels, 64, 32, 32, 255 * 0.21f, 255 * 0.42f, 255 * 0.63f, "an untextured block is its (top-brightened) tint colour");
            BrExpect(pixels, 64, 54, 32, 255, 128, 0, "a textured block without FlagTinted ignores its tint");
        });

        [TestMethod]
        public async Task BlockRender_UnregisteredBlockType_DrawsMagentaTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[1]; // only type 0
            var eye = new Vector3(0.5f, 1.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            var pixels = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 7) } } }, frame, Vector3.Zero);
            BrExpect(pixels, 64, 32, 32, 255, 0, 255, "a block type with no material is untextured magenta");
        });

        // =====================================================================
        // Fog and light
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_Fog_BlendsQuadraticallyBetweenStartAndEndTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            var eye = new Vector3(0.5f, 1.5f, 0.5f); // 0.5 above the top face: the centre pixel is at distance 0.5
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            frame.FogColor = new Vector4(1, 0, 0, 0);
            frame.FogRange = new Vector4(0.1f, 1.0f, 0, 0);
            var pixels = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) } } }, frame, Vector3.Zero);
            // f = (0.5 - 0.1) / 0.9 = 0.4444; f^2 = 0.1975. colour = cyan (0,255,255) mixed toward red (255,0,0)
            float f2 = (0.4f / 0.9f) * (0.4f / 0.9f);
            BrExpect(pixels, 64, 32, 32, 255 * f2, 255 * (1 - f2), 255 * (1 - f2), "fog at distance 0.5 in range [0.1, 1.0]", tolerance: 8);

            frame.FogRange = new Vector4(0.01f, 0.2f, 0, 0); // fully fogged at 0.5
            var full = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) } } }, frame, Vector3.Zero);
            BrExpect(full, 64, 32, 32, 255, 0, 0, "fully fogged past the end distance");
        });

        [TestMethod]
        public async Task BlockRender_SunAndAmbient_LightTheFacesTheyShouldTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            // Sun straight down, no ambient, no fill: a top face is lit by the sun, a side face is not
            var top = new Vector3(0.5f, 1.5f, 0.5f);
            var side = new Vector3(1.5f, 0.5f, 0.5f);
            BlockFrameUniforms Lit(Vector3 eye, Vector3 target, Vector3 up) => new()
            {
                ViewProjection = BrView(eye, target, up) * BrProj90,
                CameraPosition = new Vector4(eye, 1),
                SunDirection = new Vector4(0, 1, 0, 0),
                SunColor = new Vector4(1, 1, 1, 0.5f),   // sun strength 0.5
                Ambient = new Vector4(0, 0, 0, 0),
                FogColor = Vector4.Zero,
                FogRange = new Vector4(1e9f, 2e9f, 0, 0),
            };
            var topPx = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) } } },
                Lit(top, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ), Vector3.Zero);
            // cyan x top 1.0 x (sun 1.0 x N.L 1.0 x strength 0.5) = (0, 128, 128)
            BrExpect(topPx, 64, 32, 32, 0, 128, 128, "a top face in the sun", tolerance: 8);

            var sidePx = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 0, 1) } } },
                Lit(side, new Vector3(1f, 0.5f, 0.5f), Vector3.UnitY), Vector3.Zero);
            BrExpect(sidePx, 64, 32, 32, 0, 0, 0, "a side face with the sun overhead and no ambient is dark", tolerance: 4);
        });

        // =====================================================================
        // Depth
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_ReversedZ_NearerQuadWinsWhateverTheDrawOrderTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[3];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            materials[2] = BlockMaterial.Uniform(BrLayerOrange);
            var eye = new Vector3(0.5f, 3.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            var near = PackedQuad.Pack(0, 1, 0, 1, 1, 4, 2); // orange top at y=2
            var far = PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1);  // cyan top at y=1
            foreach (var order in new[] { new[] { near, far }, new[] { far, near } })
            {
                var draws = order.Select(q => new BrDraw { Quads = new[] { q } }).ToArray();
                var pixels = await BrRender(device, queue, materials, draws, frame, Vector3.Zero);
                BrExpect(pixels, 64, 32, 32, 255, 128, 0, "the nearer (orange) quad is on top");
            }
        });

        // =====================================================================
        // The atlas slicer: layer = row * tilesX + column
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_TextureArrayFromAtlas_MapsTilesToLayersRowMajorTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var textures = BlockTextureArray.FromAtlas(device, queue, BrAtlas(), 32, 32, BrTile);
            if (textures.LayerCount != 4 || textures.TileSize != 16)
                throw new Exception($"A 32x32 atlas of 16px tiles must be 4 layers of 16, got {textures.LayerCount} of {textures.TileSize}");

            // layer 3 = the tile at column 1, row 1 = orange; layer 1 = column 1, row 0 = cyan
            var materials = new BlockMaterial[3];
            materials[1] = BlockMaterial.Uniform(3);
            materials[2] = BlockMaterial.Uniform(1);
            var eye = new Vector3(1f, 2f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(1f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            var pixels = await BrRender(device, queue, materials, new[]
            {
                new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1), PackedQuad.Pack(1, 0, 0, 1, 1, 4, 2) } },
            }, frame, Vector3.Zero);
            BrExpect(pixels, 64, 16, 32, 255, 128, 0, "layer 3 is the orange tile (column 1, row 1)");
            BrExpect(pixels, 64, 48, 32, 0, 255, 255, "layer 1 is the cyan tile (column 1, row 0)");
        });

        // =====================================================================
        // Quad buffers that come and go (a viewer re-meshes and evicts columns all the time)
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_ReleaseQuadBuffer_ForgetsOneBindGroup_AndALaterBufferStillDrawsTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            var eye = new Vector3(0.5f, 1.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);

            using var textures = BlockTextureArray.FromAtlas(device, queue, BrAtlas(), 32, 32, BrTile);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 16);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            pipeline.SetSectionOrigin(0, Vector3.Zero);

            async Task<byte[]> DrawOnce(GPUBuffer buffer)
            {
                pipeline.BeginFrame(frame);
                var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
                try
                {
                    using var encoder = device.CreateCommandEncoder();
                    using var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero);
                    pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, buffer);
                    pipeline.Draw(pass, 0, 1, 0);
                    pass.End();
                    using var cmd = encoder.Finish();
                    queue.Submit(new[] { cmd });
                    await queue.OnSubmittedWorkDone();
                    return await ReadBackPixels(device, queue, colorTex, 64, 64);
                }
                finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
            }

            var quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1) };
            float f = 255f * BrFaceFactor(4);

            var a = CreateQuadBuffer(device, queue, quads);
            BrExpect(await DrawOnce(a), 64, 32, 32, 0, f, f, "first buffer draws");
            if (pipeline.CachedQuadBindGroups != 1) throw new Exception($"One bound buffer must have one cached bind group, got {pipeline.CachedQuadBindGroups}");

            pipeline.ReleaseQuadBuffer(a);
            a.Destroy(); a.Dispose();
            if (pipeline.CachedQuadBindGroups != 0) throw new Exception($"ReleaseQuadBuffer must drop the buffer's bind group, {pipeline.CachedQuadBindGroups} remain");

            var b = CreateQuadBuffer(device, queue, quads);
            try
            {
                BrExpect(await DrawOnce(b), 64, 32, 32, 0, f, f, "a buffer made after the release draws");
                if (pipeline.CachedQuadBindGroups != 1) throw new Exception($"Expected 1 cached bind group again, got {pipeline.CachedQuadBindGroups}");

                // Releasing a buffer that was never bound, or one already released, is harmless and does not touch the others
                var never = CreateQuadBuffer(device, queue, quads);
                pipeline.ReleaseQuadBuffer(never);
                pipeline.ReleaseQuadBuffer(never);
                never.Destroy(); never.Dispose();
                if (pipeline.CachedQuadBindGroups != 1) throw new Exception($"Releasing an unbound buffer changed the cache: {pipeline.CachedQuadBindGroups}");
            }
            finally { b.Destroy(); b.Dispose(); }
        });

        [TestMethod]
        public void BlockRender_TextureArrayFromAtlas_RejectsAnAtlasThatIsNotWholeTilesTest()
        {
            // A 30x32 atlas cannot be 16px tiles, and 10 bytes are not a 2x2 RGBA image: both are caught before any GPU work
            bool threw = false;
            try { BlockTextureArray.FromAtlas(null!, null!, new byte[30 * 32 * 4], 30, 32, 16); }
            catch (ArgumentException) { threw = true; }
            if (!threw) throw new Exception("A 30x32 atlas of 16px tiles must be rejected");
        }

        // =====================================================================
        // One draw for a whole column: the quad carries its section, the shader lifts it
        // =====================================================================

        [TestMethod]
        public async Task BlockRender_EmbeddedSectionIndex_LiftsTheQuadSixteenBlocksPerSectionTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            // One +X face of the block at the origin; the camera sits 16 blocks up (section 1) at the same X offset, looking at that height
            var eye = new Vector3(1.5f, 16.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(1f, 16.5f, 0.5f), Vector3.UnitY) * BrProj90, eye);
            float f = 255f * BrFaceFactor(0);

            var lifted = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 0, 1, sectionIndex: 1) } } }, frame, Vector3.Zero);
            BrExpect(lifted, 64, 32, 32, 0, f, f, "a quad of section 1 is drawn 16 blocks up");

            var notLifted = await BrRender(device, queue, materials,
                new[] { new BrDraw { Quads = new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 0, 1) } } }, frame, Vector3.Zero);
            BrExpect(notLifted, 64, 32, 32, 0, 0, 0, "the same quad without a section index stays at the origin: nothing here");
        });

        [TestMethod]
        public async Task BlockRender_BindPass_SkipsRepeats_ButNeverSkipsAChangeTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var materials = new BlockMaterial[3];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            materials[2] = BlockMaterial.Uniform(BrLayerOrange, alpha: 1f);

            using var textures = BlockTextureArray.FromAtlas(device, queue, BrAtlas(), 32, 32, BrTile);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 16);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            pipeline.SetSectionOrigins(0, new[] { Vector3.Zero, new Vector3(0, 0, 0) });

            // Three blocks side by side along +Z seen from +X: cyan (opaque pass, buffer A), orange (cross pass? no: water pass, buffer B),
            // cyan again (opaque pass, buffer A again, after the pass changed twice). All three must show.
            var eye = new Vector3(3f, 0.5f, 2f);
            pipeline.BeginFrame(BrNeutral(BrView(eye, new Vector3(1f, 0.5f, 2f), Vector3.UnitY) * BrProj90, eye));

            var bufferA = CreateQuadBuffer(device, queue, new[]
            {
                PackedQuad.Pack(0, 0, 0, 1, 1, 0, 1),   // z 0
                PackedQuad.Pack(0, 0, 2, 1, 1, 0, 1),   // z 2
            });
            var bufferB = CreateQuadBuffer(device, queue, new[] { PackedQuad.Pack(0, 0, 1, 1, 1, 0, 2) }); // z 1, type 2 (orange)
            var bufferC = CreateQuadBuffer(device, queue, new[] { PackedQuad.Pack(0, 0, 3, 1, 1, 0, 1) }); // z 3, another buffer of the same pass kind as A
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 96, 32);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero);
                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, bufferA);
                pipeline.Draw(pass, 0, 1, 0);
                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, bufferA);   // an exact repeat: skipped inside
                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, bufferC);   // the buffer changes, the pass kind does not
                pipeline.Draw(pass, 0, 1, 0);
                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Water, bufferB);    // pass and buffer change
                pipeline.Draw(pass, 0, 1, 0);
                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, bufferA);   // back to the first pass and buffer
                pipeline.Draw(pass, 1, 1, 0);
                pass.End();
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var pixels = await ReadBackPixels(device, queue, colorTex, 96, 32);
                // A 90 degree view of 96 px at distance 2 is 24 px per block: seen from +X the screen right is -Z, so with the eye at z=2
                // the blocks z=0, 1, 2, 3 are at x = 84, 60, 36, 12
                var c = BrFaceFactor(0) * 255f;
                BrExpect(pixels, 96, 84, 16, 0, c, c, "z=0: cyan from the first opaque draw");
                BrExpect(pixels, 96, 60, 16, 255, 128, 0, "z=1: orange from the water-pass draw (full alpha, the water pass has no per-face shade)", tolerance: 12);
                BrExpect(pixels, 96, 36, 16, 0, c, c, "z=2: cyan from the opaque draw after two changes");
                BrExpect(pixels, 96, 12, 16, 0, c, c, "z=3: cyan from the second opaque buffer (same pass kind)");
                // What BindPass really issued: A (first), C (buffer change), B (pass + buffer), A again (pass + buffer) = 4 buffer binds and 3
                // pipeline switches. The exact repeat of A cost nothing, and the five BindPass calls are not the number of binds.
                if (pipeline.QuadBufferBinds != 4 || pipeline.PipelineSwitches != 3)
                    throw new Exception($"Expected 4 quad-buffer binds and 3 pipeline switches from 5 BindPass calls, got {pipeline.QuadBufferBinds} and {pipeline.PipelineSwitches}");
            }
            finally
            {
                pipeline.ReleaseQuadBuffer(bufferA); pipeline.ReleaseQuadBuffer(bufferB); pipeline.ReleaseQuadBuffer(bufferC);
                bufferA.Destroy(); bufferA.Dispose(); bufferB.Destroy(); bufferB.Dispose(); bufferC.Destroy(); bufferC.Dispose();
                DisposeTargets(colorTex, depthTex, colorView, depthView);
            }
        });    }
}
