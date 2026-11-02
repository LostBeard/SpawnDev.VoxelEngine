using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // RENDER BUNDLES: the draws of a frame recorded once and run in each eye, with the same picture as drawing them in the pass.
    public abstract partial class VoxelEngineTestBase
    {
        // Two eyes (two camera places) looking at two blocks' top faces, drawn straight into the pass or from one bundle run in both: the pixels of both eyes
        private static async Task<byte[]> BundleRender(GPUDevice device, GPUQueue queue, bool bundled, int bundleRuns = 2)
        {
            var materials = new BlockMaterial[3];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            materials[2] = BlockMaterial.Uniform(BrLayerCyan);
            using var textures = BlockTextureAtlas.FromRgba(device, queue, BrAtlas(), 32, 32, BrTile);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            var eyes = new[] { new Vector3(0.5f, 2.5f, 0.5f), new Vector3(1.5f, 2.5f, 1.2f) };
            var all = new List<byte>();
            GPUBuffer? buffer = null;
            GPURenderBundle? bundle = null;
            try
            {
                buffer = CreateQuadBuffer(device, queue, new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1), PackedQuad.Pack(1, 0, 0, 1, 1, 4, 2) });
                if (bundled)
                {
                    var recorder = pipeline.BeginBundle();
                    pipeline.BindBundle(recorder, BlockRenderPipeline.Pass.Opaque, buffer);
                    pipeline.DrawInBundle(recorder, 0, 1, 0);
                    pipeline.DrawInBundle(recorder, 1, 1, 0);   // a second draw, the buffer already bound
                    bundle = recorder.Finish();
                }
                for (int e = 0; e < bundleRuns; e++)
                {
                    var eye = eyes[e % eyes.Length];
                    var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
                    frame.Ambient = new Vector4(0.5f, 0.5f, 0.5f, 0f);
                    pipeline.BeginFrame(frame);   // written between the passes: the bundle reads it when it runs
                    var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
                    try
                    {
                        using var encoder = device.CreateCommandEncoder();
                        using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                        {
                            if (bundled) pipeline.ExecuteBundle(pass, bundle!);
                            else
                            {
                                pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, buffer);
                                pipeline.Draw(pass, 0, 1, 0);
                                pipeline.Draw(pass, 1, 1, 0);
                            }
                            pass.End();
                        }
                        using var cmd = encoder.Finish();
                        queue.Submit(new[] { cmd });
                        await queue.OnSubmittedWorkDone();
                        all.AddRange(await ReadBackPixels(device, queue, colorTex, 64, 64));
                    }
                    finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
                }
            }
            finally
            {
                bundle?.Dispose();
                if (buffer != null) { buffer.Destroy(); buffer.Dispose(); }
            }
            return all.ToArray();
        }

        [TestMethod]
        public async Task BlockRender_ABundleRunInTwoEyes_DrawsWhatThePassDrawsTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            var direct = await BundleRender(device, queue, false);
            var bundled = await BundleRender(device, queue, true);
            if (direct.Length != bundled.Length) throw new Exception($"{direct.Length} bytes direct, {bundled.Length} bundled");
            int different = 0, lit = 0;
            for (int i = 0; i < direct.Length; i += 4)
            {
                if (direct[i] != bundled[i] || direct[i + 1] != bundled[i + 1] || direct[i + 2] != bundled[i + 2]) different++;
                if (direct[i + 1] > 40) lit++;
            }
            if (lit < 400) throw new Exception($"something is drawn: {lit} lit pixels");
            if (different != 0) throw new Exception($"{different} pixels differ between drawing straight and running the bundle");
            // The two eyes saw different pictures (the frame's matrices were rewritten between the runs and the bundle used each)
            int apart = 0;
            for (int i = 0; i < direct.Length / 2; i += 4)
                if (bundled[i + 1] != bundled[direct.Length / 2 + i + 1]) apart++;
            if (apart < 50) throw new Exception($"the second run used the first's matrices: only {apart} pixels differ between the eyes");
        });
    }
}
