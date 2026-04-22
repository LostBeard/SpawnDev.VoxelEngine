using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Phase 3 hybrid render pass tests. The target scene composites greedy-meshed
    // blocky content (VertexPullPipeline) and Dual Marching Cubes smooth terrain
    // (SdfRenderPipeline) into the same frame. Both pipelines share a color/depth
    // target pair and identical 128-byte UniformData so per-frame camera/fog/time
    // state can be uploaded once and reused. Reversed-Z depth resolves visibility
    // between the two representations - closer surfaces win regardless of which
    // pipeline drew them.
    public abstract partial class VoxelEngineTestBase
    {
        // ===================================================================
        // CPU-only: uniform layout compatibility
        // ===================================================================

        /// <summary>
        /// Hybrid rendering requires both pipelines to read the same 128-byte
        /// uniform block so the per-frame camera/fog/time state can be uploaded
        /// once and bound to both. Binary-compatible layout is a hard contract.
        /// </summary>
        [TestMethod]
        public void HybridRender_BlockAndSdfUniformLayoutsAreBinaryCompatibleTest()
        {
            int blockSize = Marshal.SizeOf<VertexPullPipeline.UniformData>();
            int sdfSize = Marshal.SizeOf<SdfRenderPipeline.UniformData>();

            if (blockSize != 128)
                throw new Exception(
                    $"VertexPullPipeline.UniformData must be 128 bytes for WGSL struct " +
                    $"alignment, got {blockSize}");
            if (sdfSize != 128)
                throw new Exception(
                    $"SdfRenderPipeline.UniformData must be 128 bytes for WGSL struct " +
                    $"alignment, got {sdfSize}");
            if (blockSize != sdfSize)
                throw new Exception(
                    $"Block uniform ({blockSize} bytes) and SDF uniform ({sdfSize} bytes) " +
                    "must be binary-compatible so hybrid render can share per-frame " +
                    "camera/fog/time state across both pipelines");
        }

        // ===================================================================
        // WebGPU: blocks + SDF coexist in one frame
        // ===================================================================

        /// <summary>
        /// Core hybrid render test. Draws a blocky +Y face and a larger smooth SDF
        /// ground quad into the same color+depth target, with reversed-Z depth
        /// resolving overlap. Verifies both pixel regions appear: neutral-gray
        /// block pixels AND warm rocky-tan SDF pixels, proving the two pipelines
        /// compose without destroying each other's output.
        /// </summary>
        [TestMethod]
        public async Task HybridRender_BlockAndSdfInSameFrame_BothVisibleTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var blockPipeline = new VertexPullPipeline();
            blockPipeline.Init(device, queue, "bgra8unorm");
            using var sdfPipeline = new SdfRenderPipeline();
            sdfPipeline.Init(device, queue, "bgra8unorm");

            // Blocky geometry: one 16x16 +Y face at section (0,0,0). Block type 1
            // (stone, ~0.5 gray). With sectionOffset=(0,0,0) this sits at world
            // (0..16, 1, 0..16) - a slab elevated above the SDF ground by 1 unit.
            var blockQuads = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1) };
            using var blockBuffer = CreateQuadBuffer(device, queue, blockQuads);

            // Smooth geometry: a flat ground quad at y=0 extending (-30..30, -30..30)
            // in world space. +Y normal so the SDF shader's sun lights it warm tan.
            // Quad is wider than the block's 0..16 XZ extent AND wider than the oblique
            // camera's view footprint at y=0 so SDF pixels remain visible along every
            // edge where the block does not occlude them.
            var positions = new float[]
            {
                -30f, 0f,  30f,
                 30f, 0f,  30f,
                 30f, 0f, -30f,
                -30f, 0f, -30f,
            };
            var normals = new float[]
            {
                0f, 1f, 0f,
                0f, 1f, 0f,
                0f, 1f, 0f,
                0f, 1f, 0f,
            };
            var quadIndices = new int[] { 0, 1, 2, 3 };
            var (posBuf, normBuf, idxBuf) = UploadSdfMesh(device, queue, positions, normals, quadIndices);

            try
            {
                int width = 64, height = 64;
                // Oblique view: high and offset so we can see the block top AND
                // the SDF ground sweeping around it. LookAt target at block center
                // so the block fills the center of the frame.
                var camPos = new Vector3(20f, 18f, 20f);
                var view = Matrix4x4.CreateLookAt(camPos, new Vector3(8f, 0.5f, 8f), Vector3.UnitY);
                var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2.2f, 1f, 0.1f);
                var vp = view * proj;

                // Same camera/lighting uploaded to both pipelines (proves shared layout).
                blockPipeline.UpdateUniforms(vp, Vector3.Zero, 1f, Vector3.Zero, 0f, new Vector3(0.5f), 0f, camPos);
                sdfPipeline.UpdateUniforms(vp, Vector3.Zero, 1f, Vector3.Zero, 0f, new Vector3(0.5f), 0f, camPos);

                var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, width, height);
                try
                {
                    using var encoder = device.CreateCommandEncoder();

                    // Pass 1: clear both color+depth to background, draw blocky content.
                    using (var blockPass = blockPipeline.BeginRenderPass(
                        encoder, colorView, depthView,
                        clearColor: true, clearColorValue: Vector3.Zero))
                    {
                        blockPipeline.DrawSection(blockPass, blockBuffer, blockQuads.Length);
                        blockPass.End();
                    }

                    // Pass 2: load prior color+depth, draw SDF. Reversed-Z depth test
                    // keeps block pixels on top where the two overlap and lets SDF
                    // fill the surrounding area where the block does not cover.
                    using (var sdfPass = sdfPipeline.BeginRenderPass(
                        encoder, colorView, depthView,
                        clearColor: false))
                    {
                        sdfPipeline.DrawSection(sdfPass, posBuf, normBuf, idxBuf, 1);
                        sdfPass.End();
                    }

                    using var cmd = encoder.Finish();
                    queue.Submit(new[] { cmd });
                    await queue.OnSubmittedWorkDone();

                    var pixels = await ReadBackPixels(device, queue, colorTex, width, height);
                    int totalNonBg = CountNonBackgroundPixels(pixels);

                    if (totalNonBg < width * height / 4)
                        throw new Exception(
                            $"Only {totalNonBg}/{width * height} non-background pixels. " +
                            "Hybrid render failed to draw either layer.");

                    // Classify pixels by hue signature:
                    // - Block (stone type 1) base color is neutral gray (0.5, 0.5, 0.5) -
                    //   after lighting R ~= G ~= B within ~15.
                    // - SDF shader base color is rocky tan (0.55 R, 0.50 G, 0.42 B) -
                    //   R exceeds B by ~25+ after lighting. The R>B gap is the tell.
                    int blockPixels = 0, sdfPixels = 0;
                    for (int i = 0; i < pixels.Length; i += 4)
                    {
                        int b = pixels[i + 0];
                        int g = pixels[i + 1];
                        int r = pixels[i + 2];
                        if (r < 15 && g < 15 && b < 15) continue; // cleared background

                        if (r - b >= 20)
                            sdfPixels++;
                        else if (Math.Abs(r - g) <= 15 && Math.Abs(g - b) <= 15)
                            blockPixels++;
                    }

                    if (blockPixels < 30)
                        throw new Exception(
                            $"Only {blockPixels} neutral-gray pixels (total non-bg {totalNonBg}, " +
                            $"classified-SDF {sdfPixels}). Block layer did not render in the hybrid pass.");
                    if (sdfPixels < 30)
                        throw new Exception(
                            $"Only {sdfPixels} rocky-tan pixels (total non-bg {totalNonBg}, " +
                            $"classified-block {blockPixels}). SDF layer did not render in the hybrid pass.");
                }
                finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
            }
            finally
            {
                idxBuf.Destroy(); idxBuf.Dispose();
                normBuf.Destroy(); normBuf.Dispose();
                posBuf.Destroy(); posBuf.Dispose();
            }
        });

        /// <summary>
        /// Depth-correctness check for hybrid render: a block elevated above an SDF
        /// ground plane must occlude the ground where they overlap. Reversed-Z depth
        /// (greater-than compare, 0-clear) must work consistently across both pipelines.
        /// </summary>
        [TestMethod]
        public async Task HybridRender_BlockOccludesSdfWhenCloser_ReversedZTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var blockPipeline = new VertexPullPipeline();
            blockPipeline.Init(device, queue, "bgra8unorm");
            using var sdfPipeline = new SdfRenderPipeline();
            sdfPipeline.Init(device, queue, "bgra8unorm");

            // Block directly below the camera, SDF at the same XZ but lower Y.
            // Top-down camera, so block must visibly cover SDF where they overlap.
            var blockQuads = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1) };
            using var blockBuffer = CreateQuadBuffer(device, queue, blockQuads);

            // SDF ground extends to +/-30 in X and Z at y=-2 (2 units below block top).
            // Quad must be wider than the top-down camera's view footprint at y=-2
            // (~40x40 world units for fov=PI/2 at height 22) so every non-block pixel
            // in the view has SDF coverage beneath it.
            var positions = new float[]
            {
                -30f, -2f,  30f,
                 30f, -2f,  30f,
                 30f, -2f, -30f,
                -30f, -2f, -30f,
            };
            var normals = new float[] { 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f, 0f };
            var quadIndices = new int[] { 0, 1, 2, 3 };
            var (posBuf, normBuf, idxBuf) = UploadSdfMesh(device, queue, positions, normals, quadIndices);

            try
            {
                int width = 32, height = 32;
                // Top-down view centered on the 16x16 block. With fov=PI/2 at y=20 the
                // view at y=1 covers ~38m x ~38m, so the 16x16 block fills the inner
                // ~42% of the view and the SDF ground visibly surrounds it.
                var camPos = new Vector3(8f, 20f, 8f);
                var view = Matrix4x4.CreateLookAt(camPos, new Vector3(8f, 0f, 8f), Vector3.UnitZ);
                var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
                var vp = view * proj;

                blockPipeline.UpdateUniforms(vp, Vector3.Zero, 1f, Vector3.Zero, 0f, new Vector3(0.5f), 0f, camPos);
                sdfPipeline.UpdateUniforms(vp, Vector3.Zero, 1f, Vector3.Zero, 0f, new Vector3(0.5f), 0f, camPos);

                var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, width, height);
                try
                {
                    using var encoder = device.CreateCommandEncoder();
                    using (var blockPass = blockPipeline.BeginRenderPass(
                        encoder, colorView, depthView,
                        clearColor: true, clearColorValue: Vector3.Zero))
                    {
                        blockPipeline.DrawSection(blockPass, blockBuffer, blockQuads.Length);
                        blockPass.End();
                    }
                    using (var sdfPass = sdfPipeline.BeginRenderPass(
                        encoder, colorView, depthView,
                        clearColor: false))
                    {
                        sdfPipeline.DrawSection(sdfPass, posBuf, normBuf, idxBuf, 1);
                        sdfPass.End();
                    }
                    using var cmd = encoder.Finish();
                    queue.Submit(new[] { cmd });
                    await queue.OnSubmittedWorkDone();

                    var pixels = await ReadBackPixels(device, queue, colorTex, width, height);

                    // Center pixel points at the middle of the block's top face -
                    // must be neutral block gray (R ~= G ~= B), NOT rocky tan.
                    var (r, g, b, _) = GetPixel(pixels, width, width / 2, height / 2);
                    if (r < 20 && g < 20 && b < 20)
                        throw new Exception(
                            $"Center pixel ({r},{g},{b}) is background - nothing rendered over the block's XZ.");
                    if (r - b >= 20)
                        throw new Exception(
                            $"Center pixel ({r},{g},{b}) is rocky-tan (R-B={r - b}) - SDF won the depth test " +
                            "over a closer block. Reversed-Z ordering is broken across the two pipelines.");

                    // A corner pixel falls outside the 16x16 block and inside the SDF ground -
                    // must be SDF tan (R > B) since the block does not cover it.
                    var (cr, cg, cb, _) = GetPixel(pixels, width, 1, 1);
                    if (cr < 20 && cg < 20 && cb < 20)
                        throw new Exception(
                            $"Corner pixel ({cr},{cg},{cb}) is background - SDF did not fill the block's uncovered surround.");
                    if (cr - cb < 10)
                        throw new Exception(
                            $"Corner pixel ({cr},{cg},{cb}) is not SDF-tinted (R-B={cr - cb}) - " +
                            "SDF ground did not render outside the block's XZ extent.");
                }
                finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
            }
            finally
            {
                idxBuf.Destroy(); idxBuf.Dispose();
                normBuf.Destroy(); normBuf.Dispose();
                posBuf.Destroy(); posBuf.Dispose();
            }
        });
    }
}
