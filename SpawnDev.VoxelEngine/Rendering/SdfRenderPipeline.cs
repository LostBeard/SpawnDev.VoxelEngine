using SpawnDev.BlazorJS.JSObjects;
using System.Numerics;
using System.Runtime.InteropServices;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WebGPU render pipeline for Dual Marching Cubes (SDF) smooth-terrain meshes.
    ///
    /// Pulls vertices from three storage buffers produced by SdfMeshPipeline:
    ///   positions  : array&lt;f32&gt; (3 floats per vertex, world-space)
    ///   normals    : array&lt;f32&gt; (3 floats per vertex, world-space)
    ///   quadIndices: array&lt;i32&gt; (4 ints per quad, vertex indices)
    ///
    /// The DMC kernel bakes chunk world offset into positions, so SectionOffset in
    /// uniforms is unused for SDF rendering (pass Vector3.Zero).
    ///
    /// Uniform layout is identical to VertexPullPipeline's 128-byte UniformData so
    /// callers can share camera/fog/time state across blocky and smooth pipelines.
    /// </summary>
    public class SdfRenderPipeline : IDisposable
    {
        private GPUDevice? _device;
        private GPUQueue? _queue;

        private GPURenderPipeline? _pipeline;
        private GPUBuffer? _uniformBuffer;
        // Bind groups keyed by (positions, normals, quadIndices) triple identity.
        // DMC mesh buffers are per-section and may be recreated on re-mesh, so the
        // cache is invalidated on every UpdateUniforms call (same rule as the blocky path).
        private readonly Dictionary<long, GPUBindGroup> _bindGroupCache = new();

        /// <summary>Whether the pipeline is initialized and ready to render.</summary>
        public bool IsReady => _pipeline != null;

        /// <summary>
        /// Uniform data layout matching the WGSL Uniforms struct (128 bytes).
        /// Kept byte-for-byte compatible with VertexPullPipeline.UniformData so
        /// a single per-frame struct can feed both pipelines.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct UniformData
        {
            public Matrix4x4 MVP;
            public Vector3 SectionOffset;
            public float VoxelSize;
            public Vector3 FogColor;
            public float FogDensity;
            public Vector3 AmbientColor;
            public float Time;
            public Vector3 CameraWorldPos;
            public float _pad0;
        }

        /// <summary>
        /// Initialize the SDF mesh render pipeline. Call after the WebGPU device is available.
        /// </summary>
        public void Init(GPUDevice device, GPUQueue queue, string colorFormat)
        {
            _device = device;
            _queue = queue;

            _uniformBuffer = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = 128,
                Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
            });

            using var shaderModule = device.CreateShaderModule(new GPUShaderModuleDescriptor
            {
                Code = SdfRenderShaders.SdfMeshShader,
            });

            _pipeline = device.CreateRenderPipeline(new GPURenderPipelineDescriptor
            {
                Layout = "auto",
                Vertex = new GPUVertexState
                {
                    Module = shaderModule,
                    EntryPoint = "vs_main",
                },
                Fragment = new GPUFragmentState
                {
                    Module = shaderModule,
                    EntryPoint = "fs_main",
                    Targets = new[]
                    {
                        new GPUColorTargetState
                        {
                            Format = colorFormat,
                            Blend = new GPUBlendState
                            {
                                Color = new GPUBlendComponent
                                {
                                    SrcFactor = GPUBlendFactor.SrcAlpha,
                                    DstFactor = GPUBlendFactor.OneMinusSrcAlpha,
                                    Operation = GPUBlendOperation.Add,
                                },
                                Alpha = new GPUBlendComponent
                                {
                                    SrcFactor = GPUBlendFactor.One,
                                    DstFactor = GPUBlendFactor.OneMinusSrcAlpha,
                                    Operation = GPUBlendOperation.Add,
                                },
                            },
                        },
                    },
                },
                Primitive = new GPUPrimitiveState
                {
                    Topology = GPUPrimitiveTopology.TriangleList,
                    CullMode = GPUCullMode.None,
                    FrontFace = GPUFrontFace.CCW,
                },
                DepthStencil = new GPUDepthStencilState
                {
                    Format = ReversedZHelper.DepthFormat,
                    DepthWriteEnabled = true,
                    DepthCompare = ReversedZHelper.DepthCompare,
                },
            });
        }

        /// <summary>
        /// Upload uniform data for the current frame. Also invalidates the per-mesh bind group
        /// cache because SDF mesh buffers are per-section and may be destroyed on re-mesh.
        /// MVP is a raw .NET Matrix4x4 (row-major); WGSL reads as column-major, producing the
        /// implicit transpose needed to convert v*M row-vector math to M*v column-vector math.
        /// </summary>
        public void UpdateUniforms(
            Matrix4x4 mvp,
            Vector3 sectionOffset,
            float voxelSize,
            Vector3 fogColor,
            float fogDensity,
            Vector3 ambientColor,
            float time,
            Vector3 cameraWorldPos)
        {
            if (_uniformBuffer == null || _queue == null) return;

            foreach (var bg in _bindGroupCache.Values)
                bg.Dispose();
            _bindGroupCache.Clear();

            var data = new UniformData
            {
                MVP = mvp,
                SectionOffset = sectionOffset,
                VoxelSize = voxelSize,
                FogColor = fogColor,
                FogDensity = fogDensity,
                AmbientColor = ambientColor,
                Time = time,
                CameraWorldPos = cameraWorldPos,
            };

            var bytes = new byte[128];
            unsafe
            {
                fixed (byte* ptr = bytes)
                {
                    *(UniformData*)ptr = data;
                }
            }
            _queue.WriteBuffer(_uniformBuffer, 0, bytes);
        }

        /// <summary>
        /// Begin a render pass compatible with SDF mesh rendering.
        /// Matches VertexPullPipeline's pass configuration (reversed-Z depth, sky-blue clear).
        /// </summary>
        public GPURenderPassEncoder BeginRenderPass(
            GPUCommandEncoder encoder,
            GPUTextureView colorTarget,
            GPUTextureView depthTarget,
            bool clearColor = true,
            Vector3? clearColorValue = null)
        {
            var cc = clearColorValue ?? new Vector3(0.5f, 0.7f, 0.9f);

            return encoder.BeginRenderPass(new GPURenderPassDescriptor
            {
                ColorAttachments = new[]
                {
                    new GPURenderPassColorAttachment
                    {
                        View = colorTarget,
                        LoadOp = clearColor ? GPULoadOp.Clear : GPULoadOp.Load,
                        StoreOp = GPUStoreOp.Store,
                        ClearValue = new double[] { cc.X, cc.Y, cc.Z, 1.0 },
                    },
                },
                DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
                {
                    View = depthTarget,
                    DepthLoadOp = clearColor ? "clear" : "load",
                    DepthStoreOp = "store",
                    DepthClearValue = ReversedZHelper.DepthClearValue,
                },
            });
        }

        /// <summary>
        /// Draw one SDF mesh section. Issues 6 vertices per quad (two CCW triangles).
        /// </summary>
        /// <param name="pass">Active render pass.</param>
        /// <param name="positions">f32 vertex positions (stride 3).</param>
        /// <param name="normals">f32 vertex normals (stride 3).</param>
        /// <param name="quadIndices">i32 quad indices (stride 4).</param>
        /// <param name="quadCount">Number of quads to draw.</param>
        public void DrawSection(
            GPURenderPassEncoder pass,
            GPUBuffer positions,
            GPUBuffer normals,
            GPUBuffer quadIndices,
            int quadCount)
        {
            if (_pipeline == null || quadCount <= 0) return;

            var bindGroup = GetOrCreateBindGroup(positions, normals, quadIndices);
            if (bindGroup == null) return;

            pass.SetPipeline(_pipeline);
            pass.SetBindGroup(0, bindGroup);
            pass.Draw((uint)(quadCount * 6), 1, 0, 0);
        }

        private GPUBindGroup? GetOrCreateBindGroup(GPUBuffer positions, GPUBuffer normals, GPUBuffer quadIndices)
        {
            // Composite key from the three buffer identities. Collision risk is negligible
            // in practice; the cache is also invalidated every frame via UpdateUniforms.
            long key = unchecked(
                ((long)positions.GetHashCode() * 73856093L) ^
                ((long)normals.GetHashCode() * 19349663L) ^
                ((long)quadIndices.GetHashCode() * 83492791L));

            if (_bindGroupCache.TryGetValue(key, out var existing))
                return existing;

            if (_device == null || _pipeline == null || _uniformBuffer == null)
                return null;

            var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _pipeline.GetBindGroupLayout(0),
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniformBuffer } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = positions } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = normals } },
                    new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = quadIndices } },
                },
            });

            _bindGroupCache[key] = bg;
            return bg;
        }

        /// <summary>Clear the bind group cache (call when mesh buffers are reallocated).</summary>
        public void InvalidateBindGroups()
        {
            foreach (var bg in _bindGroupCache.Values)
                bg.Dispose();
            _bindGroupCache.Clear();
        }

        public void Dispose()
        {
            InvalidateBindGroups();
            _uniformBuffer?.Destroy();
            _uniformBuffer?.Dispose();
            _pipeline?.Dispose();
        }
    }
}
