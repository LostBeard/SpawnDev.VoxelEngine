using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Textured, lit, fogged voxel rendering from <see cref="Meshing.PackedQuad"/> buffers. Where <see cref="VertexPullPipeline"/> draws a
    /// quad in one flat colour per block type, this draws what a Minecraft-style game needs:
    ///
    /// - a texture per block FACE from a 2D atlas (<see cref="BlockTextureAtlas"/>), tiled across greedy-merged quads, with animated tiles (<see cref="SetAnimations"/>)
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
        private GPUBuffer? _frameBuffer, _materialBuffer, _originBuffer, _animTableBuffer, _timelineBuffer, _shapeBuffer;
        private GPUBindGroupLayout? _frameLayout, _quadLayout;
        private GPUPipelineLayout? _pipelineLayout;
        private GPUBindGroup? _frameBindGroup;
        private GPURenderPipeline? _opaque, _cross, _water;
        private BlockTextureAtlas? _ownedPlaceholder;
        private BiomeTintMap? _biomeTints, _ownedBiomeTints;
        private LightVolume? _light, _ownedLight;
        private GPUTextureView? _textureView;
        private int _atlasTilesPerRow = 1, _atlasTileSize = 1, _atlasTileCount = 1;
        private readonly Dictionary<GPUBuffer, GPUBindGroup> _quadBindGroups = new();
        private int _maxSections;

        public bool IsReady => _opaque != null;
        /// <summary>How many quad buffers have a cached bind group (grows with each buffer bound, shrinks with <see cref="ReleaseQuadBuffer"/>).</summary>
        public int CachedQuadBindGroups => _quadBindGroups.Count;

        /// <summary>
        /// Quad-buffer binds <see cref="BindPass"/> has actually issued, and pipeline switches (each also sets the frame group): running totals.
        /// <see cref="BindPass"/> skips a repeat, so these are the calls into the browser, not the number of times it was called; a renderer
        /// subtracts two readings to report a frame.
        /// </summary>
        public long QuadBufferBinds { get; private set; }
        public long PipelineSwitches { get; private set; }
        public int MaxSections => _maxSections;
        /// <summary>The frame uniform buffer (<see cref="BlockFrameUniforms"/>, written by <see cref="BeginFrame"/>): other pipelines drawn in the same pass share it, so they are lit and fogged alike.</summary>
        public GPUBuffer FrameBuffer => _frameBuffer ?? throw new InvalidOperationException("Call Init first");

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
            // The shaped blocks' boxes (all absent until SetShapes)
            _shapeBuffer = device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)ShapeTable.MaxSlots * ShapeTable.WordsPerSlot * 4,
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });

            // Every block type starts as "missing" (untextured magenta), so an unregistered type is visible, not black
            var missing = new BlockMaterial[PackedBlock.MaxBlockTypes];
            System.Array.Fill(missing, BlockMaterial.Missing);
            SetMaterials(missing);

            _frameLayout = device.CreateBindGroupLayout(new GPUBindGroupLayoutDescriptor
            {
                Entries = new GPUBindGroupLayoutEntry[]
                {
                    new() { Binding = 0, Visibility = GPUShaderStageFlags.VERTEX | GPUShaderStageFlags.FRAGMENT, Buffer = new GPUBufferBindingLayout { Type = "uniform" } },
                    new() { Binding = 1, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 2, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 3, Visibility = GPUShaderStageFlags.FRAGMENT, Texture = new GPUTextureBindingLayout { SampleType = "float", ViewDimension = "2d" } },
                    new() { Binding = 4, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 5, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 6, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 7, Visibility = GPUShaderStageFlags.FRAGMENT, Texture = new GPUTextureBindingLayout { SampleType = "float", ViewDimension = "2d" } },
                    new() { Binding = 8, Visibility = GPUShaderStageFlags.FRAGMENT, Texture = new GPUTextureBindingLayout { SampleType = "float", ViewDimension = "2d" } },
                    new() { Binding = 9, Visibility = GPUShaderStageFlags.FRAGMENT, Texture = new GPUTextureBindingLayout { SampleType = "float", ViewDimension = "2d" } },
                    new() { Binding = 10, Visibility = GPUShaderStageFlags.FRAGMENT, Sampler = new GPUSamplerBindingLayout { Type = "filtering" } },
                    // The light of the world near the camera (LightVolume): its settings, the slot of each column of the window, the bytes
                    new() { Binding = 11, Visibility = GPUShaderStageFlags.FRAGMENT, Buffer = new GPUBufferBindingLayout { Type = "uniform" } },
                    new() { Binding = 12, Visibility = GPUShaderStageFlags.FRAGMENT, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 13, Visibility = GPUShaderStageFlags.FRAGMENT, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
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
            // Water is seen from both sides: from below, a swimmer looks up at the underside of the surface
            _water = CreatePipeline(shader, "fs_water", colorFormat, GPUCullMode.None, blend: true, depthWrite: false);

            // No light until a world gives some: a one-slot volume, switched off
            _ownedLight = new LightVolume();
            _ownedLight.Init(device, queue, 1, 1);
            _ownedLight.Enabled = false;
            _ownedLight.Update();
            _light = _ownedLight;

            // Plains colours everywhere until a world says otherwise
            _ownedBiomeTints = BiomeTintMap.Create(device, queue);
            _biomeTints = _ownedBiomeTints;

            // A valid texture binding before any real textures are set
            _ownedPlaceholder = BlockTextureAtlas.White(device, queue);
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

        /// <summary>
        /// Use this atlas. The caller keeps ownership and must keep it alive while the pipeline can draw. Animations set before are dropped
        /// (their tile numbers belonged to the old atlas): call <see cref="SetAnimations"/> again.
        /// </summary>
        public void SetTextures(BlockTextureAtlas textures)
        {
            if (_device == null) throw new InvalidOperationException("Call Init first");
            _textureView = textures.View;
            _atlasTilesPerRow = textures.TilesPerRow;
            _atlasTileSize = textures.TileSize;
            _atlasTileCount = textures.TileCount;
            SetAnimations(System.Array.Empty<TextureAnimation>());
        }

        /// <summary>
        /// Which tiles animate: wherever a face names an animation's <see cref="TextureAnimation.BaseTile"/>, the tile the timeline gives for the
        /// current time is drawn (20 ticks a second, from <see cref="BlockFrameUniforms.Time"/>). Replaces every earlier animation.
        /// </summary>
        public void SetAnimations(IReadOnlyList<TextureAnimation> animations)
        {
            if (_device == null || _queue == null) throw new InvalidOperationException("Call Init first");
            var table = new uint[Math.Max(_atlasTileCount, 1) * 2];       // per tile: offset, length
            var timeline = new List<uint>();
            foreach (var a in animations)
            {
                if (a.BaseTile < 0 || a.BaseTile >= _atlasTileCount)
                    throw new ArgumentOutOfRangeException(nameof(animations), $"Animated tile {a.BaseTile} is not in the {_atlasTileCount}-tile atlas");
                if (a.Timeline.Length == 0)
                    throw new ArgumentException($"Animated tile {a.BaseTile} has an empty timeline", nameof(animations));
                if (table[a.BaseTile * 2 + 1] != 0)
                    throw new ArgumentException($"Tile {a.BaseTile} is animated twice", nameof(animations));
                table[a.BaseTile * 2] = (uint)timeline.Count;
                table[a.BaseTile * 2 + 1] = (uint)a.Timeline.Length;
                foreach (int t in a.Timeline)
                {
                    if (t < 0 || t >= _atlasTileCount)
                        throw new ArgumentOutOfRangeException(nameof(animations), $"Tile {a.BaseTile} timeline names tile {t}, outside the {_atlasTileCount}-tile atlas");
                    timeline.Add((uint)t);
                }
            }
            if (timeline.Count == 0) timeline.Add(0);                      // a storage binding cannot be empty

            _animTableBuffer?.Destroy(); _animTableBuffer?.Dispose();
            _timelineBuffer?.Destroy(); _timelineBuffer?.Dispose();
            _animTableBuffer = _device.CreateBuffer(new GPUBufferDescriptor { Size = (ulong)table.Length * 4, Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst });
            _timelineBuffer = _device.CreateBuffer(new GPUBufferDescriptor { Size = (ulong)timeline.Count * 4, Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst });
            _queue.WriteBuffer(_animTableBuffer, 0, MemoryMarshal.AsBytes(table.AsSpan()).ToArray());
            _queue.WriteBuffer(_timelineBuffer, 0, MemoryMarshal.AsBytes(timeline.ToArray().AsSpan()).ToArray());
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
                    new() { Binding = 4, Resource = new GPUBufferBinding { Buffer = _animTableBuffer! } },
                    new() { Binding = 5, Resource = new GPUBufferBinding { Buffer = _timelineBuffer! } },
                    new() { Binding = 6, Resource = new GPUBufferBinding { Buffer = _shapeBuffer! } },
                    new() { Binding = 7, Resource = _biomeTints!.Grass.View },
                    new() { Binding = 8, Resource = _biomeTints.Foliage.View },
                    new() { Binding = 9, Resource = _biomeTints.Water.View },
                    new() { Binding = 10, Resource = _biomeTints.Sampler },
                    new() { Binding = 11, Resource = new GPUBufferBinding { Buffer = _light!.Params! } },
                    new() { Binding = 12, Resource = new GPUBufferBinding { Buffer = _light.Slots! } },
                    new() { Binding = 13, Resource = new GPUBufferBinding { Buffer = _light.Data! } },
                },
            });
        }

        /// <summary>
        /// The light the pixels are lit by (sky light and block light near the camera). The caller keeps ownership and must keep the volume alive while the pipeline can draw; call
        /// <see cref="LightVolume.Update"/> each frame before drawing. Without one the world is lit as it was before there was any light data.
        /// </summary>
        public void SetLight(LightVolume volume)
        {
            if (_device == null) throw new InvalidOperationException("Call Init first");
            if (!volume.IsReady) throw new ArgumentException("The light volume has not been initialised", nameof(volume));
            _light = volume;
            RebuildFrameBindGroup();
        }

        /// <summary>
        /// The shapes of the shaped blocks (<see cref="BlockMaterial.Shape"/>): their boxes, tiles and windows. Give the same table to <see cref="Meshing.ColumnMesher.SetShapes"/>, which makes the quads.
        /// Replaces the table; the tiles are atlas tiles, so set it after <see cref="SetTextures"/> (whose animations it does not touch).
        /// </summary>
        public void SetShapes(ShapeTable shapes)
        {
            if (_shapeBuffer == null || _queue == null) throw new InvalidOperationException("Call Init first");
            var words = new uint[ShapeTable.MaxSlots * ShapeTable.WordsPerSlot];
            shapes.Words().CopyTo(words, 0);
            _queue.WriteBuffer(_shapeBuffer, 0, MemoryMarshal.AsBytes(words.AsSpan()).ToArray());
        }

        /// <summary>
        /// The biome colour maps that blocks tinted with <see cref="BlockMaterial.TintGrass"/> / <see cref="BlockMaterial.TintFoliage"/> / <see cref="BlockMaterial.TintWater"/> read. The caller keeps ownership and
        /// must keep the map alive while the pipeline can draw; the pipeline starts with plains colours everywhere.
        /// </summary>
        public void SetBiomeTints(BiomeTintMap map)
        {
            if (_device == null) throw new InvalidOperationException("Call Init first");
            _biomeTints = map;
            RebuildFrameBindGroup();
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
            _light?.Update();
            var bytes = new byte[BlockFrameUniforms.SizeInBytes];
            var f = frame;
            f.AtlasTilesPerRow = _atlasTilesPerRow;   // the atlas in use, not something a caller should have to repeat
            f.AtlasTileSize = _atlasTileSize;
            unsafe
            {
                fixed (byte* p = bytes) *(BlockFrameUniforms*)p = f;
            }
            _queue.WriteBuffer(_frameBuffer, 0, bytes);
        }

        /// <summary>
        /// Begin the frame's render pass: clears colour and depth (reversed-Z: 0 is the far plane). <paramref name="clearAlpha"/> below 1 clears to a premultiplied see-through colour
        /// (augmented reality: what is not drawn shows the camera feed).
        /// </summary>
        public GPURenderPassEncoder BeginRenderPass(GPUCommandEncoder encoder, GPUTextureView colorTarget, GPUTextureView depthTarget,
            Vector3 clearColor, float clearAlpha = 1f)
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
                        ClearValue = new double[] { clearColor.X * clearAlpha, clearColor.Y * clearAlpha, clearColor.Z * clearAlpha, clearAlpha },
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
            // A call into the browser costs about as much as the GPU work of a small draw: set only what changed since the last bind on
            // this pass (the pipeline and the frame group when the kind changes, the quad group when the buffer does)
            if (!ReferenceEquals(pass, _lastPass) || kind != _lastKind)
            {
                pass.SetPipeline(kind switch { Pass.Opaque => _opaque!, Pass.Cross => _cross!, _ => _water! });
                pass.SetBindGroup(0, _frameBindGroup);
                PipelineSwitches++;
                _lastPass = pass;
                _lastKind = kind;
                _lastQuadBuffer = null;
            }
            if (!ReferenceEquals(quadBuffer, _lastQuadBuffer))
            {
                pass.SetBindGroup(1, GetQuadBindGroup(quadBuffer));
                QuadBufferBinds++;
                _lastQuadBuffer = quadBuffer;
            }
        }

        // What BindPass last set on which pass (so it can skip repeats)
        private GPURenderPassEncoder? _lastPass;
        private Pass _lastKind;
        private GPUBuffer? _lastQuadBuffer;

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

        /// <summary>
        /// Forget one quad buffer's cached bind group: call when that buffer is about to be destroyed (a section re-meshed, a column
        /// evicted). Without it the cache keeps a bind group per buffer ever drawn. Harmless for a buffer never bound.
        /// </summary>
        public void ReleaseQuadBuffer(GPUBuffer quadBuffer)
        {
            if (ReferenceEquals(quadBuffer, _lastQuadBuffer)) _lastQuadBuffer = null;
            if (_quadBindGroups.Remove(quadBuffer, out var group)) group.Dispose();
        }

        /// <summary>Forget the cached bind groups of quad buffers: call when a buffer was destroyed or replaced.</summary>
        public void InvalidateQuadBuffers()
        {
            foreach (var g in _quadBindGroups.Values) g.Dispose();
            _quadBindGroups.Clear();
            _lastQuadBuffer = null;
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
            _animTableBuffer?.Destroy(); _animTableBuffer?.Dispose();
            _timelineBuffer?.Destroy(); _timelineBuffer?.Dispose();
            _frameBuffer?.Destroy(); _frameBuffer?.Dispose();
            _materialBuffer?.Destroy(); _materialBuffer?.Dispose();
            _originBuffer?.Destroy(); _originBuffer?.Dispose();
            _shapeBuffer?.Destroy(); _shapeBuffer?.Dispose();
            _ownedPlaceholder?.Dispose();
            _ownedBiomeTints?.Dispose();
            _ownedLight?.Dispose();
        }
    }
}
