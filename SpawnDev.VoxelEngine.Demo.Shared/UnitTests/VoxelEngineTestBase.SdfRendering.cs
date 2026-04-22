using System.Numerics;
using System.Runtime.InteropServices;
using ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // SDF render pipeline tests. Validates the WGSL shader + C# wrapper for
    // rendering Dual Marching Cubes meshes (positions, normals, quad indices).
    // Synthetic mesh data isolates the render path from the ILGPU SDF mesh
    // pipeline - the SDF pipeline itself is covered by 30+ tests elsewhere.
    public abstract partial class VoxelEngineTestBase
    {
        // ===================================================================
        // CPU-only tests (run on ALL backends)
        // ===================================================================

        [TestMethod]
        public void SdfRendering_UniformData_SizeIs128BytesTest()
        {
            int size = Marshal.SizeOf<SdfRenderPipeline.UniformData>();
            if (size != 128)
                throw new Exception(
                    $"SdfRenderPipeline.UniformData must be 128 bytes to match WGSL struct, got {size}. " +
                    "Layout must mirror VertexPullPipeline.UniformData for cross-pipeline uniform reuse.");
        }

        // ===================================================================
        // WebGPU rendering tests
        // ===================================================================

        [TestMethod]
        public async Task SdfRendering_PipelineInit_IsReadyTransitionTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new SdfRenderPipeline();

            if (pipeline.IsReady)
                throw new Exception("SdfRenderPipeline should NOT be ready before Init()");

            pipeline.Init(device, queue, "bgra8unorm");
            if (!pipeline.IsReady)
                throw new Exception("SdfRenderPipeline should be ready after Init()");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task SdfRendering_FlatQuad_RendersVisiblePixelsTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new SdfRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            // One horizontal quad at y=0, 8m x 8m centered on origin.
            // Sized to cover ~64% of the 32x32 view with 90deg FOV at 5u distance
            // (view frustum at y=0 is 10x10, quad covers 8/10 per axis, so ~656
            // of 1024 pixels expected). A smaller quad (eg 2x2) only produces ~36
            // pixels from pure rasterization math, which is correct rendering but
            // an ambiguous pass/fail signal. Larger quad proves pixels hit real
            // rasterization breadth, not just a single triangle sliver.
            var positions = new float[]
            {
                -4f, 0f,  4f,   // v0
                 4f, 0f,  4f,   // v1
                 4f, 0f, -4f,   // v2
                -4f, 0f, -4f,   // v3
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
                int width = 32, height = 32;
                var camPos = new Vector3(0, 5, 0);
                var view = Matrix4x4.CreateLookAt(camPos, Vector3.Zero, Vector3.UnitZ);
                var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
                var vp = view * proj;

                var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, width, height);
                try
                {
                    pipeline.UpdateUniforms(vp, Vector3.Zero, 1f, new Vector3(0), 0f, new Vector3(0.3f), 0f, camPos);

                    using var encoder = device.CreateCommandEncoder();
                    using var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, true, new Vector3(0));
                    pipeline.DrawSection(pass, posBuf, normBuf, idxBuf, 1);
                    pass.End();
                    using var cmd = encoder.Finish();
                    queue.Submit(new[] { cmd });
                    await queue.OnSubmittedWorkDone();

                    var pixels = await ReadBackPixels(device, queue, colorTex, width, height);
                    int rendered = CountNonBackgroundPixels(pixels);

                    // 8x8 quad at 5u distance with 90deg vertical FOV covers 80%
                    // of each axis (frustum is 10x10 at y=0). Expected pixel count
                    // is ~655/1024. Threshold 200 catches complete rendering
                    // failures and >70% partial losses while tolerating
                    // rasterization edge rounding.
                    if (rendered < 200)
                        throw new Exception(
                            $"Only {rendered}/{width * height} non-black pixels. " +
                            "Flat SDF quad looking straight down should cover a large portion of the view. " +
                            "Check shader vertex pulling and triangulation.");

                    // Center pixel should be lit rocky tan (0.55, 0.50, 0.42 base * lighting)
                    var (r, g, b, a) = GetPixel(pixels, width, width / 2, height / 2);
                    if (r < 20 && g < 20 && b < 20)
                        throw new Exception($"Center pixel ({r},{g},{b},{a}) is black - quad not rendered at center");

                    // Base color is warmer in red channel than blue (0.55 R > 0.42 B)
                    if (r <= b)
                        throw new Exception(
                            $"Center pixel ({r},{g},{b}) - red should exceed blue for rocky tan base color. " +
                            "Fragment shader baseColor or lighting may be wrong.");
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

        [TestMethod]
        public async Task SdfRendering_Fog_DensityAffectsBrightnessTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new SdfRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            // Same flat quad, but positioned 10 units from camera so fog has a clear distance to act on.
            var positions = new float[]
            {
                -5f, -10f,  5f,
                 5f, -10f,  5f,
                 5f, -10f, -5f,
                -5f, -10f, -5f,
            };
            var normals = new float[] { 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f, 0f };
            var quadIndices = new int[] { 0, 1, 2, 3 };

            var (posBuf, normBuf, idxBuf) = UploadSdfMesh(device, queue, positions, normals, quadIndices);
            try
            {
                int width = 32, height = 32;
                var camPos = new Vector3(0, 0, 0);
                var view = Matrix4x4.CreateLookAt(camPos, new Vector3(0, -10, 0), Vector3.UnitZ);
                var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
                var vp = view * proj;
                var fogColor = new Vector3(0.5f, 0.5f, 0.5f);

                // No fog
                var pxNoFog = await RenderSdfAndReadBack(device, queue, pipeline, posBuf, normBuf, idxBuf, 1,
                    vp, fogColor, 0f, new Vector3(0.3f), camPos, width, height);
                var (r0, g0, b0, _) = GetPixel(pxNoFog, width, width / 2, height / 2);

                // Heavy fog - distance ~10, density 0.5 -> fogFactor = exp(-10*0.5*0.5) = exp(-2.5) ~ 0.08
                var pxFog = await RenderSdfAndReadBack(device, queue, pipeline, posBuf, normBuf, idxBuf, 1,
                    vp, fogColor, 0.5f, new Vector3(0.3f), camPos, width, height);
                var (r1, g1, b1, _) = GetPixel(pxFog, width, width / 2, height / 2);

                int noFogBrightness = r0 + g0 + b0;
                int fogBrightness = r1 + g1 + b1;

                if (noFogBrightness < 30)
                    throw new Exception($"No-fog pixel ({r0},{g0},{b0}) too dark - quad not rendered");

                int brightnessDiff = Math.Abs(noFogBrightness - fogBrightness);
                if (brightnessDiff < 10)
                    throw new Exception(
                        $"Fog has no visible effect. No-fog=({r0},{g0},{b0}), heavy-fog=({r1},{g1},{b1}). " +
                        "fogDensity uniform is not reaching the shader or fog math is broken.");
            }
            finally
            {
                idxBuf.Destroy(); idxBuf.Dispose();
                normBuf.Destroy(); normBuf.Dispose();
                posBuf.Destroy(); posBuf.Dispose();
            }
        });

        [TestMethod]
        public async Task SdfRendering_NormalDirection_SunFacingBrighterTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new SdfRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            // Two identical quads in opposite orientations.
            // Shader sun = normalize(0.3, 1.0, 0.5), so +Y normal is bright, -Y normal is dark.
            var positions = new float[]
            {
                -1f, 0f,  1f,
                 1f, 0f,  1f,
                 1f, 0f, -1f,
                -1f, 0f, -1f,
            };
            var quadIndices = new int[] { 0, 1, 2, 3 };

            int width = 32, height = 32;
            var camPos = new Vector3(0, 5, 0);
            var view = Matrix4x4.CreateLookAt(camPos, Vector3.Zero, Vector3.UnitZ);
            var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
            var vp = view * proj;

            // Render with +Y normals (facing sun)
            var normalsUp = new float[] { 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 };
            var (posU, normU, idxU) = UploadSdfMesh(device, queue, positions, normalsUp, quadIndices);
            byte upR, upG, upB;
            try
            {
                var px = await RenderSdfAndReadBack(device, queue, pipeline, posU, normU, idxU, 1,
                    vp, new Vector3(0), 0f, new Vector3(0.3f), camPos, width, height);
                (upR, upG, upB, _) = GetPixel(px, width, width / 2, height / 2);
            }
            finally
            {
                idxU.Destroy(); idxU.Dispose();
                normU.Destroy(); normU.Dispose();
                posU.Destroy(); posU.Dispose();
            }

            // Render with -Y normals (facing away from sun)
            var normalsDown = new float[] { 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0 };
            var (posD, normD, idxD) = UploadSdfMesh(device, queue, positions, normalsDown, quadIndices);
            byte dnR, dnG, dnB;
            try
            {
                var px = await RenderSdfAndReadBack(device, queue, pipeline, posD, normD, idxD, 1,
                    vp, new Vector3(0), 0f, new Vector3(0.3f), camPos, width, height);
                (dnR, dnG, dnB, _) = GetPixel(px, width, width / 2, height / 2);
            }
            finally
            {
                idxD.Destroy(); idxD.Dispose();
                normD.Destroy(); normD.Dispose();
                posD.Destroy(); posD.Dispose();
            }

            int upBright = upR + upG + upB;
            int dnBright = dnR + dnG + dnB;

            if (upBright < 30)
                throw new Exception($"+Y normal pixel ({upR},{upG},{upB}) too dark - not rendered");
            if (upBright <= dnBright)
                throw new Exception(
                    $"+Y normal brightness {upBright} ({upR},{upG},{upB}) should be > " +
                    $"-Y normal brightness {dnBright} ({dnR},{dnG},{dnB}). " +
                    "Lambertian shading or normal vertex pulling is broken.");
        });

        [TestMethod]
        public async Task SdfRendering_MultipleQuads_IndexedByQuadBufferTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new SdfRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            // Two separate quads sharing no vertices. Tests that quadIndices indexing
            // works correctly for multiple quads (vertexID / 6 -> quadIdx).
            var positions = new float[]
            {
                // Quad A at x < 0
                -3f, 0f,  1f,
                -1f, 0f,  1f,
                -1f, 0f, -1f,
                -3f, 0f, -1f,
                // Quad B at x > 0
                 1f, 0f,  1f,
                 3f, 0f,  1f,
                 3f, 0f, -1f,
                 1f, 0f, -1f,
            };
            var normals = new float[24];
            for (int i = 0; i < 8; i++) { normals[i * 3 + 0] = 0; normals[i * 3 + 1] = 1; normals[i * 3 + 2] = 0; }
            var quadIndices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7 };

            var (posBuf, normBuf, idxBuf) = UploadSdfMesh(device, queue, positions, normals, quadIndices);
            try
            {
                int width = 64, height = 32;
                var camPos = new Vector3(0, 8, 0);
                var view = Matrix4x4.CreateLookAt(camPos, Vector3.Zero, Vector3.UnitZ);
                var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 2f, 0.1f);
                var vp = view * proj;

                var pixels = await RenderSdfAndReadBack(device, queue, pipeline, posBuf, normBuf, idxBuf, 2,
                    vp, new Vector3(0), 0f, new Vector3(0.3f), camPos, width, height);

                int leftNonBlack = 0, rightNonBlack = 0;
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        var (pr, pg, pb, _) = GetPixel(pixels, width, x, y);
                        if (pr > 10 || pg > 10 || pb > 10)
                        {
                            if (x < width / 2) leftNonBlack++;
                            else rightNonBlack++;
                        }
                    }

                if (leftNonBlack == 0)
                    throw new Exception($"Left half has 0 pixels. Quad A (x<0) did not render. Right={rightNonBlack}");
                if (rightNonBlack == 0)
                    throw new Exception($"Right half has 0 pixels. Quad B (x>0) did not render. Left={leftNonBlack}");
            }
            finally
            {
                idxBuf.Destroy(); idxBuf.Dispose();
                normBuf.Destroy(); normBuf.Dispose();
                posBuf.Destroy(); posBuf.Dispose();
            }
        });

        // -------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------

        /// <summary>Upload SDF mesh data to three GPUBuffers. Caller owns all three.</summary>
        private static (GPUBuffer positions, GPUBuffer normals, GPUBuffer quadIndices)
            UploadSdfMesh(GPUDevice device, GPUQueue queue, float[] positions, float[] normals, int[] quadIndices)
        {
            var posBytes = new byte[positions.Length * 4];
            Buffer.BlockCopy(positions, 0, posBytes, 0, posBytes.Length);
            var posBuf = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)Math.Max(posBytes.Length, 16),
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });
            queue.WriteBuffer(posBuf, 0, posBytes);

            var normBytes = new byte[normals.Length * 4];
            Buffer.BlockCopy(normals, 0, normBytes, 0, normBytes.Length);
            var normBuf = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)Math.Max(normBytes.Length, 16),
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });
            queue.WriteBuffer(normBuf, 0, normBytes);

            var idxBytes = new byte[quadIndices.Length * 4];
            Buffer.BlockCopy(quadIndices, 0, idxBytes, 0, idxBytes.Length);
            var idxBuf = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)Math.Max(idxBytes.Length, 16),
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });
            queue.WriteBuffer(idxBuf, 0, idxBytes);

            return (posBuf, normBuf, idxBuf);
        }

        /// <summary>Render an SDF mesh and read back the color target pixels.</summary>
        private static async Task<byte[]> RenderSdfAndReadBack(
            GPUDevice device, GPUQueue queue, SdfRenderPipeline pipeline,
            GPUBuffer positions, GPUBuffer normals, GPUBuffer quadIndices, int quadCount,
            Matrix4x4 vp, Vector3 fogColor, float fogDensity, Vector3 ambientColor, Vector3 camPos,
            int width, int height)
        {
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, width, height);
            try
            {
                pipeline.UpdateUniforms(vp, Vector3.Zero, 1f, fogColor, fogDensity, ambientColor, 0f, camPos);

                using var encoder = device.CreateCommandEncoder();
                using var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, true, new Vector3(0));
                pipeline.DrawSection(pass, positions, normals, quadIndices, quadCount);
                pass.End();
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();

                return await ReadBackPixels(device, queue, colorTex, width, height);
            }
            finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
        }
    }
}
