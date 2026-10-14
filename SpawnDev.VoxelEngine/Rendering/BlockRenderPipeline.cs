using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Textured, lit, fogged voxel rendering from <see cref="Meshing.PackedQuad"/> buffers. Where <see cref="VertexPullPipeline"/> draws a
    /// quad in one flat colour per block type, this draws what a Minecraft-style game needs:
    ///
    /// - a texture per block FACE from a texture array (<see cref="BlockTextureArray"/>), tiled across greedy-merged quads
    /// - a per-block tint (grass, foliage, water), alpha cutout (leaf gaps, plant outlines), per-face brightness
    /// - three passes: <see cref="Pass.Opaque"/> (cull back faces), <see cref="Pass.Cross"/> (plants: two crossed planes, double-sided),
    ///   <see cref="Pass.Water"/> (alpha blended, depth test but no depth write: draw it after the others, farthest section first)
    /// - sun + ambient + fixed fill light and start/end fog from <see cref="BlockFrameUniforms"/>; the caller owns the lighting model
    /// - section placement through a storage buffer of origins indexed by the draw's firstInstance, so one quad buffer holds many
    ///   sections and drawing one writes no uniform
    ///
    /// Reversed-Z depth (<see cref="ReversedZHelper"/>): build the projection with it and clear depth to 0.
    ///
    /// Per frame: <see cref="BeginFrame"/>; <see cref="BeginRenderPass"/>; for each pass <see cref="BindPass"/> once per quad buffer, then
    /// <see cref="Draw"/> per section range.
    /// </summary>
    public sealed class BlockRenderPipeline : IDisposable
    {
        public enum Pass { Opaque, Cross, Water }

        private GPUDevice? _device;
        private GPUQueue? _queue;
        private GPUBuffer? _frameBuffer, _materialBuffer, _originBuffer;
        private GPUSampler? _sampler;
        private GPUBindGroupLayout? _frameLayout, _quadLayout;
        private GPUPipelineLayout? _pipelineLayout;
        private GPUBindGroup? _frameBindGroup;
        private GPURenderPipeline? _opaque, _cross, _water;
        private BlockTextureArray? _ownedPlaceholder;
        private GPUTextureView? _textureView;
        private readonly Dictionary<GPUBuffer, GPUBindGroup> _quadBindGroups = new();
        private int _maxSections;

        public bool IsReady => _opaque != null;
        public int MaxSections => _maxSections;

        /// <param name="colorFormat">The colour target's format (the canvas format, or "bgra8unorm" for an offscreen target).</param>
        /// <param name="maxSections">How many section origins the origin buffer holds (the draw's firstInstance must be below it).</param>
        public void Init(GPUDevice device, GPUQueue queue, string colorFormat, int maxSections = 1 << 16)
        {
            _device = device;
            _queue = queue;
            _maxSections = maxSections;

            _frameBuffer = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = BlockFrameUniforms.SizeInBytes,
                Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
            });
            _materialBuffer = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)(PackedBlock.MaxBlockTypes * Marshal.SizeOf<BlockMaterial>()),
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });
            _originBuffer = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)maxSections * 16,
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });

            // Every block type starts as "missing" (untextured magenta), so an unregistered type is visible, not black
            var missing = new BlockMaterial[PackedBlock.MaxBlockTypes];
            System.Array.Fill(missing, BlockMaterial.Missing);
            SetMaterials(missing);

            // Nearest (the pixelated look) and repeating: a merged w x h quad tiles its texture w x h times
            _sampler = device.CreateSampler(new GPUSamplerDescriptor
            {
                MagFilter = GPUFilterMode.Nearest,
                MinFilter = GPUFilterMode.Nearest,
                AddressModeU = "repeat",
                AddressModeV = "repeat",
            });

            _frameLayout = device.CreateBindGroupLayout(new GPUBindGroupLayoutDescriptor
            {
                Entries = new GPUBindGroupLayoutEntry[]
                {
                    new() { Binding = 0, Visibility = GPUShaderStageFlags.VERTEX | GPUShaderStageFlags.FRAGMENT, Buffer = new GPUBufferBindingLayout { Type = "uniform" } },
                    new() { Binding = 1, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 2, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 3, Visibility = GPUShaderStageFlags.FRAGMENT, Texture = new GPUTextureBindingLayout { SampleType = "float", ViewDimension = "2d-array" } },
                    new() { Binding = 4, Visibility = GPUShaderStageFlags.FRAGMENT, Sampler = new GPUSamplerBindingLayout { Type = "filtering" } },
                },
            });
            _quadLayout = device.CreateBindGroupLayout(new GPUBindGroupLayoutDescriptor
            {
                Entries = new GPUBindGroupLayoutEntry[]
                {
                    new() { Binding = 0, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                },
            });
            _pipelineLayout = device.CreatePipelineLayout(new GPUPipelineLayoutDescriptor
            {
                BindGroupLayouts = new[] { _frameLayout, _quadLayout },
            });

            using var shader = device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = BlockRenderShaders.Source });
            _opaque = CreatePipeline(shader, "fs_main", colorFormat, GPUCullMode.Back, blend: false, depthWrite: true);
            _cross = CreatePipeline(shader, "fs_main", colorFormat, GPUCullMode.None, blend: false, depthWrite: true);
            _water = CreatePipeline(shader, "fs_water", colorFormat, GPUCullMode.Back, blend: true, depthWrite: false);

            // A valid texture binding before any real textures are set
            _ownedPlaceholder = BlockTextureArray.White(device, queue);
            SetTextures(_ownedPlaceholder);
        }

        private GPURenderPipeline CreatePipeline(GPUShaderModule shader, string fragmentEntry, string colorFormat,
            GPUCullMode cull, bool blend, bool depthWrite)
        {
            var target = new GPUColorTargetState { Format = colorFormat };
            if (blend)
            {
                target.Blend = new GPUBlendState
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
                };
            }
            return _device!.CreateRenderPipeline(new GPURenderPipelineDescriptor
            {
                Layout = _pipelineLayout!,
                Vertex = new GPUVertexState { Module = shader, EntryPoint = "vs_main" },
                Fragment = new GPUFragmentState { Module = shader, EntryPoint = fragmentEntry, Targets = new[] { target } },
                Primitive = new GPUPrimitiveState
                {
                    Topology = GPUPrimitiveTopology.TriangleList,
                    CullMode = cull,
                    FrontFace = GPUFrontFace.CCW,
                },
                DepthStencil = new GPUDepthStencilState
                {
                    Format = ReversedZHelper.DepthFormat,
                    DepthWriteEnabled = depthWrite,
                    DepthCompare = ReversedZHelper.DepthCompare,
                },
            });
        }

        /// <summary>Upload the material table: entry i is block type i (up to <see cref="PackedBlock.MaxBlockTypes"/>).</summary>
        public void SetMaterials(ReadOnlySpan<BlockMaterial> materials)
        {
            if (_materialBuffer == null || _queue == null) throw new InvalidOperationException("Call Init first");
            if (materials.Length > PackedBlock.MaxBlockTypes)
                throw new ArgumentOutOfRangeException(nameof(materials), $"{materials.Length} materials; at most {PackedBlock.MaxBlockTypes} block types");
            _queue.WriteBuffer(_materialBuffer, 0, MemoryMarshal.AsBytes(materials).ToArray());
        }

        /// <summary>Use these textures. The caller keeps ownership and must keep them alive while the pipeline can draw.</summary>
        public void SetTextures(BlockTextureArray textures)
        {
            if (_device == null) throw new InvalidOperationException("Call Init first");
            _textureView = textures.View;
            RebuildFrameBindGroup();
        }

        private void RebuildFrameBindGroup()
        {
            _frameBindGroup?.Dispose();
            _frameBindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _frameLayout!,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _frameBuffer! } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = _materialBuffer! } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _originBuffer! } },
                    new() { Binding = 3, Resource = _textureView! },
                    new() { Binding = 4, Resource = _sampler! },
                },
            });
        }

        /// <summary>Set the world-space origin (the corner of section-local 0,0,0) of sections firstIndex, firstIndex + 1, ...</summary>
        public void SetSectionOrigins(int firstIndex, ReadOnlySpan<Vector3> origins)
        {
            if (_originBuffer == null || _queue == null) throw new InvalidOperationException("Call Init first");
            if (firstIndex < 0 || firstIndex + origins.Length > _maxSections)
                throw new ArgumentOutOfRangeException(nameof(firstIndex), $"Sections {firstIndex}..{firstIndex + origins.Length} exceed the {_maxSections} the pipeline was created for");
            var bytes = new byte[origins.Length * 16];
            for (int i = 0; i < origins.Length; i++)
            {
                BitConverter.TryWriteBytes(new Span<byte>(bytes, i * 16, 4), origins[i].X);
                BitConverter.TryWriteBytes(new Span<byte>(bytes, i * 16 + 4, 4), origins[i].Y);
                BitConverter.TryWriteBytes(new Span<byte>(bytes, i * 16 + 8, 4), origins[i].Z);
            }
            _queue.WriteBuffer(_originBuffer, (ulong)firstIndex * 16, bytes);
        }

        /// <summary>One section's origin.</summary>
        public void SetSectionOrigin(int index, Vector3 origin) => SetSectionOrigins(index, stackalloc Vector3[] { origin });

        /// <summary>Upload this frame's camera, light and fog.</summary>
        public void BeginFrame(in BlockFrameUniforms frame)
        {
            if (_frameBuffer == null || _queue == null) throw new InvalidOperationException("Call Init first");
            var bytes = new byte[BlockFrameUniforms.SizeInBytes];
            unsafe
            {
                fixed (byte* p = bytes) *(BlockFrameUniforms*)p = frame;
            }
            _queue.WriteBuffer(_frameBuffer, 0, bytes);
        }

        /// <summary>Begin the frame's render pass: clears colour and depth (reversed-Z: 0 is the far plane).</summary>
        public GPURenderPassEncoder BeginRenderPass(GPUCommandEncoder encoder, GPUTextureView colorTarget, GPUTextureView depthTarget,
            Vector3 clearColor)
        {
            return encoder.BeginRenderPass(new GPURenderPassDescriptor
            {
                ColorAttachments = new[]
                {
                    new GPURenderPassColorAttachment
                    {
                        View = colorTarget,
                        LoadOp = GPULoadOp.Clear,
                        StoreOp = GPUStoreOp.Store,
                        ClearValue = new double[] { clearColor.X, clearColor.Y, clearColor.Z, 1.0 },
                    },
                },
                DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
                {
                    View = depthTarget,
                    DepthLoadOp = "clear",
                    DepthStoreOp = "store",
                    DepthClearValue = ReversedZHelper.DepthClearValue,
                },
            });
        }

        /// <summary>Select <paramref name="kind"/>'s pipeline and bind <paramref name="quadBuffer"/> for the draws that follow.</summary>
        public void BindPass(GPURenderPassEncoder pass, Pass kind, GPUBuffer quadBuffer)
        {
            if (_frameBindGroup == null) throw new InvalidOperationException("Call Init first");
            pass.SetPipeline(kind switch { Pass.Opaque => _opaque!, Pass.Cross => _cross!, _ => _water! });
            pass.SetBindGroup(0, _frameBindGroup);
            pass.SetBindGroup(1, GetQuadBindGroup(quadBuffer));
        }

        /// <summary>
        /// Draw <paramref name="quadCount"/> quads starting at quad <paramref name="firstQuad"/> of the bound buffer, placed at section
        /// <paramref name="sectionIndex"/>'s origin.
        /// </summary>
        public void Draw(GPURenderPassEncoder pass, int firstQuad, int quadCount, int sectionIndex)
            => pass.Draw((uint)(quadCount * 6), 1, (uint)(firstQuad * 6), (uint)sectionIndex);

        private GPUBindGroup GetQuadBindGroup(GPUBuffer quadBuffer)
        {
            if (_quadBindGroups.TryGetValue(quadBuffer, out var existing)) return existing;
            var group = _device!.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _quadLayout!,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = quadBuffer } },
                },
            });
            _quadBindGroups[quadBuffer] = group;
            return group;
        }

        /// <summary>Forget the cached bind groups of quad buffers: call when a buffer was destroyed or replaced.</summary>
        public void InvalidateQuadBuffers()
        {
            foreach (var g in _quadBindGroups.Values) g.Dispose();
            _quadBindGroups.Clear();
        }

        public void Dispose()
        {
            InvalidateQuadBuffers();
            _frameBindGroup?.Dispose();
            _opaque?.Dispose();
            _cross?.Dispose();
            _water?.Dispose();
            _pipelineLayout?.Dispose();
            _frameLayout?.Dispose();
            _quadLayout?.Dispose();
            _sampler?.Dispose();
            _frameBuffer?.Destroy(); _frameBuffer?.Dispose();
            _materialBuffer?.Destroy(); _materialBuffer?.Dispose();
            _originBuffer?.Destroy(); _originBuffer?.Dispose();
            _ownedPlaceholder?.Dispose();
        }
    }
}
