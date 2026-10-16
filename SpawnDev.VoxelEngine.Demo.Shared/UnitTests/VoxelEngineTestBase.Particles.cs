using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // The PARTICLE pipeline: every emitter's slots are camera-facing quads placed and faded by the vertex shader from the kind's numbers and the frame clock alone.
    public abstract partial class VoxelEngineTestBase
    {
        // A sprite atlas: four 8 x 8 quadrants (top-left red, top-right green, bottom-left blue, bottom-right yellow) in a 16 x 16 frame
        private static byte[] PrQuadrantAtlas()
        {
            var atlas = new byte[16 * 16 * 4];
            ErPaint(atlas, 16, 0, 0, 8, 8, 255, 0, 0);
            ErPaint(atlas, 16, 8, 0, 8, 8, 0, 255, 0);
            ErPaint(atlas, 16, 0, 8, 8, 8, 0, 0, 255);
            ErPaint(atlas, 16, 8, 8, 8, 8, 255, 255, 0);
            return atlas;
        }

        // One kind that stays put: one particle, a very long life, constant size and opacity, the whole 16 x 16 atlas as its sprite
        private static ParticleKind PrStill(float size, float alpha = 1f, float count = 1f, float life = 100000f) => new()
        {
            Counts = new Vector4(count, life, 0, 1),
            Sprite = new Vector4(0, 0, 16, 16),
            Motion = Vector4.Zero,
            Scatter = Vector4.Zero,
            Fade = new Vector4(size, size, alpha, alpha),
            Tint = Vector4.One,
            Origin = new Vector4(0.5f, 0.5f, 0.5f, 0),
        };

        private static async Task<byte[]> PrRender(GPUDevice device, GPUQueue queue, byte[] atlas, int atlasW, int atlasH, ParticleKind[] kinds, ParticleEmitterInstance[] emitters,
            Vector3 eye, Vector3 target, Vector3 up, float time = 0f, float sun = 1f, Vector3? clear = null, bool wall = false)
        {
            using var textures = EntityTextureAtlas.FromRgba(device, queue, atlas, atlasW, atlasH);
            using var blocks = new BlockRenderPipeline();
            blocks.Init(device, queue, "bgra8unorm", maxSections: 4);
            using var particles = new ParticleRenderPipeline();
            particles.Init(device, queue, "bgra8unorm", blocks.FrameBuffer);
            particles.SetTextures(textures);
            particles.SetKinds(kinds);
            particles.SetEmitters(emitters);

            var frame = BrNeutral(BrView(eye, target, up) * BrProj90, eye);
            frame.Time = time;
            frame.SunColor = new Vector4(0, 0, 0, sun);
            // A wall: one cyan block face on the plane z = 0 (x and y 0..1), facing the camera at negative z
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
            blocks.BeginFrame(frame);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = blocks.BeginRenderPass(encoder, colorView, depthView, clear ?? Vector3.Zero))
                {
                    if (wallBuffer != null) { blocks.BindPass(pass, BlockRenderPipeline.Pass.Opaque, wallBuffer); blocks.Draw(pass, 0, 1, 0); }
                    particles.Draw(pass);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                return await ReadBackPixels(device, queue, colorTex, 64, 64);
            }
            finally { if (wallBuffer != null) { wallBuffer.Destroy(); wallBuffer.Dispose(); } DisposeTargets(colorTex, depthTex, colorView, depthView); }
        }

        [TestMethod]
        public async Task Particles_ASprite_FacesTheCamera_FromAnyDirection_AndStandsUprightTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // A 2 x 2 block sprite in the middle of block (0,0,0): seen from the front, the side, and diagonally from above, the same square shows with its top-left quadrant red, top-right green,
            // bottom-left blue, bottom-right yellow (the camera's up is the screen's up, so the sprite is never turned or mirrored)
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            var kinds = new[] { PrStill(2f) };
            var emitters = new[] { new ParticleEmitterInstance(0, 0, 0, 0) };
            var eyes = new (string name, Vector3 eye, Vector3 up)[]
            {
                ("front", c + new Vector3(0, 0, -3f), Vector3.UnitY),
                ("side", c + new Vector3(3f, 0, 0), Vector3.UnitY),
                ("diagonal from above", c + new Vector3(2f, 2.5f, -2f), Vector3.UnitY),
            };
            foreach (var (name, eye, up) in eyes)
            {
                var pixels = await PrRender(device, queue, PrQuadrantAtlas(), 16, 16, kinds, emitters, eye, c, up);
                string Hue(int x, int y) => ErHue(GetPixel(pixels, 64, x, y));
                // The square is ~ 20 px wide around the centre (a little less diagonally): probe 6 px from the centre
                string got = $"{Hue(26, 26)} {Hue(38, 26)} {Hue(26, 38)} {Hue(38, 38)}";
                if (got != "R G B RG") throw new Exception($"{name}: quadrants (TL TR BL BR) read {got}, expected R G B RG");
            }
        });

        [TestMethod]
        public async Task Particles_ASlotPastTheKindsCount_DrawsNothing_AndEverySlotBelowItDoesTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Particles rising 1 block a second over a 2 s life, 0.2 block each, seen from the front: the slots are spread evenly through the life, so N alive show as N dots on the centre column
            var atlas = new byte[16 * 16 * 4];
            ErPaint(atlas, 16, 0, 0, 16, 16, 255, 255, 255);
            ParticleKind Rising(float count) => new()
            {
                Counts = new Vector4(count, 2f, 0, 1),
                Sprite = new Vector4(0, 0, 16, 16),
                Motion = new Vector4(0, 1f, 0, 0),
                Scatter = Vector4.Zero,
                Fade = new Vector4(0.2f, 0.2f, 1f, 1f),
                Tint = Vector4.One,
                Origin = new Vector4(0.5f, 0.5f, 0.5f, 0),
            };
            var c = new Vector3(0.5f, 1.5f, 0.5f);
            var eye = c + new Vector3(0, 0, -3f);
            foreach (int n in new[] { 1, 2, 4 })
            {
                var pixels = await PrRender(device, queue, atlas, 16, 16, new[] { Rising(n) }, new[] { new ParticleEmitterInstance(0, 0, 0, 0) }, eye, c, Vector3.UnitY, time: 0.37f);
                int runs = 0; bool inRun = false;
                for (int y = 0; y < 64; y++)
                {
                    bool lit = GetPixel(pixels, 64, 32, y).r > 128;
                    if (lit && !inRun) runs++;
                    inRun = lit;
                }
                if (runs != n) throw new Exception($"{n} particles alive show as {runs} dots on the centre column");
            }
        });

        [TestMethod]
        public async Task Particles_FollowTheFrameClock_RiseWithTimeTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // One particle, 1 block a second up, a life of 4 s (it stays in view): half a second later it is half a block higher on screen (5.3 px at this distance)
            var atlas = new byte[16 * 16 * 4];
            ErPaint(atlas, 16, 0, 0, 16, 16, 255, 255, 255);
            var kind = new ParticleKind
            {
                Counts = new Vector4(1, 4f, 0, 1),
                Sprite = new Vector4(0, 0, 16, 16),
                Motion = new Vector4(0, 1f, 0, 0),
                Fade = new Vector4(0.3f, 0.3f, 1f, 1f),
                Tint = Vector4.One,
                Origin = new Vector4(0.5f, 0.5f, 0.5f, 0),
            };
            var c = new Vector3(0.5f, 2.5f, 0.5f);
            var eye = c + new Vector3(0, 0, -3f);
            int CentreY(byte[] px) { long sum = 0, n = 0; for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) if (GetPixel(px, 64, x, y).r > 128) { sum += y; n++; } return n == 0 ? -1 : (int)(sum / n); }
            int at0 = CentreY(await PrRender(device, queue, atlas, 16, 16, new[] { kind }, new[] { new ParticleEmitterInstance(0, 0, 0, 0) }, eye, c, Vector3.UnitY, time: 100f));
            int at1 = CentreY(await PrRender(device, queue, atlas, 16, 16, new[] { kind }, new[] { new ParticleEmitterInstance(0, 0, 0, 0) }, eye, c, Vector3.UnitY, time: 100.5f));
            if (at0 < 0 || at1 < 0) throw new Exception($"the particle is not on screen ({at0}, {at1})");
            if (at0 - at1 < 3 || at0 - at1 > 8) throw new Exception($"half a second of rise moved the particle {at0 - at1} px up, expected about 5");
        });

        [TestMethod]
        public async Task Particles_AnimationFrames_FollowTheAge_AndTheSpritesAreFetchedExactlyTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // Two 8 x 8 frames side by side, red then green, a life of 1000 s: sampled a quarter life apart the particle shows both, red in the first half of its life and green in the second
            var atlas = new byte[16 * 8 * 4];
            ErPaint(atlas, 16, 0, 0, 8, 8, 255, 0, 0);
            ErPaint(atlas, 16, 8, 0, 8, 8, 0, 255, 0);
            var kind = new ParticleKind
            {
                Counts = new Vector4(1, 1000f, 0, 2),
                Sprite = new Vector4(0, 0, 8, 8),
                Fade = new Vector4(2f, 2f, 1f, 1f),
                Tint = Vector4.One,
                Origin = new Vector4(0.5f, 0.5f, 0.5f, 0),
            };
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            var eye = c + new Vector3(0, 0, -3f);
            var seen = new List<string>();
            for (int i = 0; i < 4; i++)
            {
                var px = await PrRender(device, queue, atlas, 16, 8, new[] { kind }, new[] { new ParticleEmitterInstance(3, 4, 5, 0) }, new Vector3(3.5f, 4.5f, 5.5f) + new Vector3(0, 0, -3f), new Vector3(3.5f, 4.5f, 5.5f), Vector3.UnitY, time: i * 250f);
                seen.Add(ErHue(GetPixel(px, 64, 32, 32)));
            }
            if (!seen.Contains("R") || !seen.Contains("G")) throw new Exception($"a quarter life apart the particle showed {string.Join(" ", seen)}: both frames expected");
            // The cycle of 4 samples must read as R-run then G-run (frames follow the age in order), not interleave
            int changes = 0;
            for (int i = 0; i < 4; i++) if (seen[i] != seen[(i + 1) % 4]) changes++;
            if (changes != 2) throw new Exception($"the frames {string.Join(" ", seen)} are not in age order (a ring of two runs changes twice)");
        });

        [TestMethod]
        public async Task Particles_AlphaBlend_NightDimsAParticle_AndAGlowingOneIsNotDimmedTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var atlas = new byte[16 * 16 * 4];
            ErPaint(atlas, 16, 0, 0, 16, 16, 255, 255, 255);
            var c = new Vector3(0.5f, 0.5f, 0.5f);
            var eye = c + new Vector3(0, 0, -3f);
            var emitters = new[] { new ParticleEmitterInstance(0, 0, 0, 0) };

            // Half-opaque white over blue: the middle of both
            var half = await PrRender(device, queue, atlas, 16, 16, new[] { PrStill(3f, alpha: 0.5f) }, emitters, eye, c, Vector3.UnitY, sun: 1f, clear: new Vector3(0, 0, 1));
            var (hr, hg, hb, _) = GetPixel(half, 64, 32, 32);
            if (Math.Abs(hr - 128) > 6 || Math.Abs(hg - 128) > 6 || hb < 250) throw new Exception($"half white over blue: ({hr},{hg},{hb}), expected about (128,128,255)");

            // Night (sun 0): a particle keeps a quarter of its colour; one that glows keeps all of it
            var night = await PrRender(device, queue, atlas, 16, 16, new[] { PrStill(3f) }, emitters, eye, c, Vector3.UnitY, sun: 0f);
            int dim = GetPixel(night, 64, 32, 32).r;
            if (Math.Abs(dim - 64) > 6) throw new Exception($"a white particle at night: {dim}, expected about 64 (a quarter)");
            var glowKind = PrStill(3f); glowKind.Counts.Z = ParticleKind.FlagEmissive;
            var glow = await PrRender(device, queue, atlas, 16, 16, new[] { glowKind }, emitters, eye, c, Vector3.UnitY, sun: 0f);
            int bright = GetPixel(glow, 64, 32, 32).r;
            if (bright < 250) throw new Exception($"a glowing particle at night: {bright}, expected full brightness");
        });

        [TestMethod]
        public async Task Particles_AreHiddenByTerrainInFrontOfThem_AndShowInFrontOfItTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // A cyan block face on the plane z = 0, camera at z = -3 looking at its middle; a white 2 x 2 sprite behind it (z 2.5) is hidden, one in front of it (z -1.5) shows over it
            var atlas = new byte[16 * 16 * 4];
            ErPaint(atlas, 16, 0, 0, 16, 16, 255, 255, 255);
            var kinds = new[] { PrStill(1.2f) };
            var eye = new Vector3(0.5f, 0.5f, -3f);
            var target = new Vector3(0.5f, 0.5f, 0.5f);
            var behind = await PrRender(device, queue, atlas, 16, 16, kinds, new[] { new ParticleEmitterInstance(0, 0, 2, 0) }, eye, target, Vector3.UnitY, wall: true);
            string behindHue = ErHue(GetPixel(behind, 64, 32, 32));
            if (behindHue != "GB") throw new Exception($"a sprite behind the wall: the middle shows {behindHue}, expected the wall's cyan (GB)");
            var front = await PrRender(device, queue, atlas, 16, 16, kinds, new[] { new ParticleEmitterInstance(0, 0, -2, 0) }, eye, target, Vector3.UnitY, wall: true);
            string frontHue = ErHue(GetPixel(front, 64, 32, 32));
            if (frontHue != "RGB") throw new Exception($"a sprite in front of the wall: the middle shows {frontHue}, expected the sprite's white (RGB)");
        });

        [TestMethod]
        public async Task Particles_Gravity_PullsAParticleDownAndBuoyancyUpTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // No initial velocity, a life of 2 s, gravity of -2 or +2 blocks a second squared: a quarter of a second apart the particle has gone down (or up) on screen
            var atlas = new byte[16 * 16 * 4];
            ErPaint(atlas, 16, 0, 0, 16, 16, 255, 255, 255);
            ParticleKind WithGravity(float g) => new()
            {
                Counts = new Vector4(1, 2f, 0, 1),
                Sprite = new Vector4(0, 0, 16, 16),
                Motion = new Vector4(0, 0, 0, g),
                Fade = new Vector4(0.3f, 0.3f, 1f, 1f),
                Tint = Vector4.One,
                Origin = new Vector4(0.5f, 0.5f, 0.5f, 0),
            };
            var c = new Vector3(0.5f, 2.5f, 0.5f);
            var eye = c + new Vector3(0, 0, -3f);
            int CentreY(byte[] px) { long sum = 0, n = 0; for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) if (GetPixel(px, 64, x, y).r > 128) { sum += y; n++; } return n == 0 ? -1 : (int)(sum / n); }
            foreach (float g in new[] { -2f, 2f })
            {
                // Emitter high up for falling (it stays in view), low for rising
                var emitter = new[] { new ParticleEmitterInstance(0, g < 0 ? 4 : 0, 0, 0) };
                int a = CentreY(await PrRender(device, queue, atlas, 16, 16, new[] { WithGravity(g) }, emitter, eye, c, Vector3.UnitY, time: 103f));
                int b = CentreY(await PrRender(device, queue, atlas, 16, 16, new[] { WithGravity(g) }, emitter, eye, c, Vector3.UnitY, time: 103.25f));
                if (a < 0 || b < 0) throw new Exception($"gravity {g}: the particle is off screen ({a}, {b})");
                int moved = b - a;   // screen y grows downward
                if (g < 0 && moved < 1) throw new Exception($"gravity {g}: the particle moved {moved} px down, expected it to fall");
                if (g > 0 && moved > -1) throw new Exception($"gravity {g}: the particle moved {moved} px down, expected it to rise");
            }
        });
    }
}
