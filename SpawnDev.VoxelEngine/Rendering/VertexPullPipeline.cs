using SpawnDev.BlazorJS.JSObjects;
using System.Numerics;
using System.Runtime.InteropServices;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WebGPU render pipeline for vertex-pulled voxel rendering.
    ///
    /// Manages the GPU render pipeline, bind groups, and per-frame rendering.
    /// The vertex shader reads PackedQuad data from a storage buffer and generates
    /// vertices procedurally - no vertex buffer needed.
    ///
    /// Usage per frame:
    ///   pipeline.UpdateUniforms(mvp, sectionOffset, voxelSize, ...);
    ///   pipeline.DrawSection(encoder, targetView, depthView, quadBuffer, quadCount);
    ///
    /// Two rendering modes:
    /// 1. Solid color: block type -> color lookup (debug, AubsCraft simple mode)
    /// 2. Textured: block type -> texture array layer (production, PBR)
    /// </summary>
    public class VertexPullPipeline : IDisposable
    {
        private GPUDevice? _device;
        private GPUQueue? _queue;

        // GPU resources
        private GPURenderPipeline? _pipeline;
        private GPUBuffer? _uniformBuffer;
        private GPUBuffer? _colorBuffer;
        // Per-section bind groups cached by quad buffer identity
        private readonly Dictionary<nint, GPUBindGroup> _bindGroupCache = new();

        /// <summary>Whether the pipeline is initialized and ready to render.</summary>
        public bool IsReady => _pipeline != null;

        /// <summary>Uniform data layout matching the WGSL Uniforms struct.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct UniformData
        {
            public Matrix4x4 MVP;           // 64 bytes
            public Vector3 SectionOffset;   // 12 bytes
            public float VoxelSize;         // 4 bytes
            public Vector3 FogColor;        // 12 bytes
            public float FogDensity;        // 4 bytes
            public Vector3 AmbientColor;    // 12 bytes
            public float Time;              // 4 bytes
            // Total: 112 bytes (aligned to 16)
        }

        /// <summary>
        /// Initialize the vertex pull render pipeline.
        /// Call after WebGPU device is available.
        /// </summary>
        /// <param name="device">WebGPU device.</param>
        /// <param name="queue">WebGPU queue.</param>
        /// <param name="colorFormat">Canvas texture format (e.g., "bgra8unorm").</param>
        /// <param name="blockColors">Per-block-type RGBA colors (4096 entries max).</param>
        public void Init(GPUDevice device, GPUQueue queue, string colorFormat, Vector4[]? blockColors = null)
        {
            _device = device;
            _queue = queue;

            // Default block colors if not provided
            if (blockColors == null)
            {
                blockColors = new Vector4[PackedBlock.MaxBlockTypes];
                blockColors[0] = new Vector4(0, 0, 0, 0);           // air = transparent
                blockColors[1] = new Vector4(0.5f, 0.5f, 0.5f, 1);  // stone = gray
                blockColors[2] = new Vector4(0.45f, 0.3f, 0.15f, 1); // dirt = brown
                blockColors[3] = new Vector4(0.3f, 0.6f, 0.2f, 1);  // grass = green
                blockColors[4] = new Vector4(0.85f, 0.8f, 0.6f, 1);  // sand = tan
                blockColors[5] = new Vector4(0.4f, 0.4f, 0.4f, 1);  // cobblestone = dark gray
                blockColors[6] = new Vector4(0.55f, 0.35f, 0.15f, 1); // wood = brown
                blockColors[7] = new Vector4(0.7f, 0.85f, 0.9f, 0.5f); // glass = light blue translucent
                blockColors[8] = new Vector4(0.2f, 0.3f, 0.8f, 0.6f); // water = blue translucent
                blockColors[9] = new Vector4(0.15f, 0.5f, 0.1f, 0.8f); // tall grass = green
                blockColors[10] = new Vector4(0.2f, 0.45f, 0.1f, 0.7f); // leaves = dark green
            }

            // Uniform buffer (112 bytes, 16-byte aligned -> 128 bytes)
            _uniformBuffer = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = 128,
                Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
            });

            // Color lookup buffer (4096 * 16 bytes = 64KB)
            var colorBytes = new byte[blockColors.Length * 16];
            for (int i = 0; i < blockColors.Length; i++)
            {
                var c = blockColors[i];
                Buffer.BlockCopy(BitConverter.GetBytes(c.X), 0, colorBytes, i * 16, 4);
                Buffer.BlockCopy(BitConverter.GetBytes(c.Y), 0, colorBytes, i * 16 + 4, 4);
                Buffer.BlockCopy(BitConverter.GetBytes(c.Z), 0, colorBytes, i * 16 + 8, 4);
                Buffer.BlockCopy(BitConverter.GetBytes(c.W), 0, colorBytes, i * 16 + 12, 4);
            }

            _colorBuffer = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)colorBytes.Length,
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });
            queue.WriteBuffer(_colorBuffer, 0, colorBytes);

            // Shader module
            using var shaderModule = device.CreateShaderModule(new GPUShaderModuleDescriptor
            {
                Code = VertexPullShaders.SolidColorShader,
            });

            // Use auto layout - simpler and avoids BlazorJS enum type issues
            // The shader defines the bind group layout via WGSL @group/@binding annotations

            // Render pipeline with reversed-Z depth
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
                    CullMode = GPUCullMode.Back,
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
        /// Upload uniform data for the current frame/section.
        /// </summary>
        public void UpdateUniforms(
            Matrix4x4 mvp,
            Vector3 sectionOffset,
            float voxelSize,
            Vector3 fogColor,
            float fogDensity,
            Vector3 ambientColor,
            float time)
        {
            if (_uniformBuffer == null || _queue == null) return;

            var data = new UniformData
            {
                MVP = mvp,
                SectionOffset = sectionOffset,
                VoxelSize = voxelSize,
                FogColor = fogColor,
                FogDensity = fogDensity,
                AmbientColor = ambientColor,
                Time = time,
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
        /// Draw a section's quads using vertex pulling.
        /// The quad buffer is a GPUBuffer containing PackedQuad data (i64 as vec2&lt;u32&gt;).
        /// </summary>
        /// <param name="pass">Active render pass.</param>
        /// <param name="quadBuffer">GPU buffer containing packed quads.</param>
        /// <param name="quadCount">Number of quads to draw.</param>
        /// <param name="quadOffset">Byte offset into the quad buffer (for shared buffers).</param>
        public void DrawSection(GPURenderPassEncoder pass, GPUBuffer quadBuffer, int quadCount, ulong quadOffset = 0)
        {
            if (_pipeline == null) return;

            var bindGroup = GetOrCreateBindGroup(quadBuffer, quadOffset);
            if (bindGroup == null) return;

            pass.SetPipeline(_pipeline);
            pass.SetBindGroup(0, bindGroup);
            pass.Draw((uint)(quadCount * 6), 1, 0, 0); // 6 vertices per quad, vertex pulling
        }

        /// <summary>
        /// Begin a render pass for voxel rendering.
        /// Uses reversed-Z depth, LoadOp.Clear for first pass or LoadOp.Load for subsequent.
        /// </summary>
        public GPURenderPassEncoder BeginRenderPass(
            GPUCommandEncoder encoder,
            GPUTextureView colorTarget,
            GPUTextureView depthTarget,
            bool clearColor = true,
            Vector3? clearColorValue = null)
        {
            var cc = clearColorValue ?? new Vector3(0.5f, 0.7f, 0.9f); // sky blue default

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

        private GPUBindGroup? GetOrCreateBindGroup(GPUBuffer quadBuffer, ulong offset)
        {
            // Simple cache key using buffer handle
            // In production, use a more robust key that includes offset
            var key = quadBuffer.GetHashCode() + (nint)offset;
            if (_bindGroupCache.TryGetValue(key, out var existing))
                return existing;

            if (_device == null || _pipeline == null || _uniformBuffer == null || _colorBuffer == null)
                return null;

            var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _pipeline.GetBindGroupLayout(0),
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniformBuffer } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = quadBuffer, Offset = offset } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _colorBuffer } },
                },
            });

            _bindGroupCache[key] = bg;
            return bg;
        }

        /// <summary>Clear the bind group cache (call when quad buffers are reallocated).</summary>
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
            _colorBuffer?.Destroy();
            _colorBuffer?.Dispose();
            _pipeline?.Dispose();
        }
    }
}
