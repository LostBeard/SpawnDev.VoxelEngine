using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// What one kind of particle does (smoke, a flame, a spark), 112 bytes: every particle of an emitter is born, rises and dies by these numbers, computed in the vertex shader from the
    /// frame's clock alone - nothing is simulated and nothing is stored per particle.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ParticleKind
    {
        /// <summary>x: particles alive at once per emitter (1..<see cref="ParticleRenderPipeline.MaxPerEmitter"/>); y: a particle's life in seconds; z: <see cref="FlagEmissive"/> or 0; w: frames of its animation.</summary>
        public Vector4 Counts;
        /// <summary>The first frame's place in the particle atlas in pixels (xy) and a frame's size in pixels (zw). The frames follow to the right, side by side.</summary>
        public Vector4 Sprite;
        /// <summary>Velocity at birth in blocks a second (xyz) and gravity along y in blocks a second squared (w; negative falls, positive is buoyancy).</summary>
        public Vector4 Motion;
        /// <summary>x: how far from <see cref="Origin"/> a particle is born, sideways (blocks); y: random sideways speed (blocks a second); z: wobble that grows over the life (blocks); w: wobble speed (radians a second).</summary>
        public Vector4 Scatter;
        /// <summary>x: size at birth, y: at death (blocks, the sprite's side); z: opacity at birth, w: at death.</summary>
        public Vector4 Fade;
        /// <summary>A colour multiplied into the sprite (rgb).</summary>
        public Vector4 Tint;
        /// <summary>Where in its block a particle is born (xyz, blocks from the block's minimum corner); w: how far up or down from there it may start (blocks).</summary>
        public Vector4 Origin;

        public const int SizeInBytes = 112;
        /// <summary>The particle glows: not darkened by night, whatever the light.</summary>
        public const float FlagEmissive = 1f;
    }

    /// <summary>A place that gives off particles of one kind: the block's minimum corner (whole numbers) and the kind's index. 16 bytes.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ParticleEmitterInstance
    {
        public Vector4 PositionKind;
        public ParticleEmitterInstance(int x, int y, int z, int kind) => PositionKind = new Vector4(x, y, z, kind);
        public const int SizeInBytes = 16;
    }

    /// <summary>
    /// Draws the world's particles - smoke, flames, sparks - in ONE instanced draw: <see cref="MaxPerEmitter"/> camera-facing quads for every emitter, each placed and faded by the vertex shader from
    /// its kind (<see cref="ParticleKind"/>) and the frame's clock (the Time of the shared <see cref="BlockFrameUniforms"/>). A particle that is not alive in its kind (its number is past the kind's count)
    /// collapses to nothing. The CPU's whole job is the list of emitters near the camera.
    ///
    /// The sprites come from an atlas of RGBA pixels, fetched exactly (pixel art stays pixel art). Alpha blended, depth tested against the terrain, never written. Lit and fogged by the same
    /// frame uniforms as the blocks. Draw it inside the block pipeline's render pass, after the opaque and plant passes and the creatures, before the water.
    /// </summary>
    public sealed class ParticleRenderPipeline : IDisposable
    {
        /// <summary>The most particles one emitter can have alive at once.</summary>
        public const int MaxPerEmitter = 16;
        /// <summary>The most kinds.</summary>
        public const int MaxKinds = 64;
        /// <summary>The most emitters one frame holds (<see cref="SetEmitters"/> drops the rest and says so).</summary>
        public const int MaxEmitters = 1 << 14;

        private GPUDevice? _device;
        private GPUQueue? _queue;
        private GPUBuffer? _frameBuffer, _emitterBuffer, _kindBuffer;
        private GPUBindGroupLayout? _layout;
        private GPUPipelineLayout? _pipelineLayout;
        private GPUBindGroup? _bindGroup;
        private GPURenderPipeline? _pipeline;
        private EntityTextureAtlas? _atlas;
        private EntityTextureAtlas? _ownedPlaceholder;
        private byte[] _upload = System.Array.Empty<byte>();
        private int _capacity;

        /// <summary>Emitters the last <see cref="SetEmitters"/> uploaded (what <see cref="Draw"/> draws).</summary>
        public int EmitterCount { get; private set; }
        /// <summary>Emitters the last <see cref="SetEmitters"/> had to leave out because of <see cref="MaxEmitters"/>.</summary>
        public int DroppedEmitters { get; private set; }
        public bool IsReady => _pipeline != null;

        /// <param name="frameUniforms">The block pipeline's frame buffer (<see cref="BlockRenderPipeline.FrameBuffer"/>).</param>
        public void Init(GPUDevice device, GPUQueue queue, string colorFormat, GPUBuffer frameUniforms, int initialEmitters = 256)
        {
            _device = device;
            _queue = queue;
            _frameBuffer = frameUniforms;

            _layout = device.CreateBindGroupLayout(new GPUBindGroupLayoutDescriptor
            {
                Entries = new GPUBindGroupLayoutEntry[]
                {
                    new() { Binding = 0, Visibility = GPUShaderStageFlags.VERTEX | GPUShaderStageFlags.FRAGMENT, Buffer = new GPUBufferBindingLayout { Type = "uniform" } },
                    new() { Binding = 1, Visibility = GPUShaderStageFlags.FRAGMENT, Texture = new GPUTextureBindingLayout { SampleType = "float", ViewDimension = "2d" } },
                    new() { Binding = 2, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                    new() { Binding = 3, Visibility = GPUShaderStageFlags.VERTEX, Buffer = new GPUBufferBindingLayout { Type = "read-only-storage" } },
                },
            });
            _pipelineLayout = device.CreatePipelineLayout(new GPUPipelineLayoutDescriptor { BindGroupLayouts = new[] { _layout } });
            using var shader = device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = ParticleRenderShaders.Source });
            _pipeline = device.CreateRenderPipeline(new GPURenderPipelineDescriptor
            {
                Layout = _pipelineLayout,
                Vertex = new GPUVertexState { Module = shader, EntryPoint = "vs_main" },
                Fragment = new GPUFragmentState
                {
                    Module = shader,
                    EntryPoint = "fs_main",
                    Targets = new[]
                    {
                        new GPUColorTargetState
                        {
                            Format = colorFormat,
                            Blend = new GPUBlendState
                            {
                                Color = new GPUBlendComponent { SrcFactor = GPUBlendFactor.SrcAlpha, DstFactor = GPUBlendFactor.OneMinusSrcAlpha, Operation = GPUBlendOperation.Add },
                                Alpha = new GPUBlendComponent { SrcFactor = GPUBlendFactor.One, DstFactor = GPUBlendFactor.OneMinusSrcAlpha, Operation = GPUBlendOperation.Add },
                            },
                        },
                    },
                },
                Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList, CullMode = GPUCullMode.None, FrontFace = GPUFrontFace.CCW },
                DepthStencil = new GPUDepthStencilState
                {
                    Format = ReversedZHelper.DepthFormat,
                    DepthWriteEnabled = false,
                    DepthCompare = ReversedZHelper.DepthCompare,
                },
            });

            _kindBuffer = device.CreateBuffer(new GPUBufferDescriptor { Size = (ulong)(MaxKinds * ParticleKind.SizeInBytes), Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst });
            EnsureCapacity(Math.Max(initialEmitters, 1));
            _ownedPlaceholder = EntityTextureAtlas.FromRgba(device, queue, new byte[] { 255, 255, 255, 255 }, 1, 1);
            SetTextures(_ownedPlaceholder);
        }

        /// <summary>Use this atlas. The caller keeps ownership and must keep it alive while the pipeline can draw.</summary>
        public void SetTextures(EntityTextureAtlas atlas)
        {
            if (_device == null) throw new InvalidOperationException("Call Init first");
            _atlas = atlas;
            RebuildBindGroup();
        }

        /// <summary>The kinds emitters refer to by index (replacing any before). At most <see cref="MaxKinds"/>.</summary>
        public void SetKinds(ReadOnlySpan<ParticleKind> kinds)
        {
            if (_queue == null || _kindBuffer == null) throw new InvalidOperationException("Call Init first");
            if (kinds.Length > MaxKinds) throw new ArgumentException($"{kinds.Length} kinds: at most {MaxKinds}");
            if (kinds.Length == 0) return;
            for (int i = 0; i < kinds.Length; i++)
            {
                int n = (int)kinds[i].Counts.X;
                if (n < 1 || n > MaxPerEmitter) throw new ArgumentException($"Kind {i} keeps {n} particles alive: 1..{MaxPerEmitter}");
                if (!(kinds[i].Counts.Y > 0f)) throw new ArgumentException($"Kind {i} lives {kinds[i].Counts.Y} seconds: it must be positive");
                if ((int)kinds[i].Counts.W < 1) throw new ArgumentException($"Kind {i} has {(int)kinds[i].Counts.W} frames: at least 1");
            }
            var bytes = MemoryMarshal.AsBytes(kinds).ToArray();
            _queue.WriteBuffer(_kindBuffer, 0, bytes, 0, bytes.Length);
        }

        private void EnsureCapacity(int emitters)
        {
            if (emitters <= _capacity) return;
            int capacity = Math.Min(Math.Max(emitters, _capacity * 2), MaxEmitters);
            _emitterBuffer?.Destroy(); _emitterBuffer?.Dispose();
            _emitterBuffer = _device!.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)capacity * ParticleEmitterInstance.SizeInBytes,
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });
            _capacity = capacity;
            _upload = new byte[capacity * ParticleEmitterInstance.SizeInBytes];
            if (_atlas != null) RebuildBindGroup();
        }

        private void RebuildBindGroup()
        {
            _bindGroup?.Dispose();
            _bindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _layout!,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _frameBuffer! } },
                    new() { Binding = 1, Resource = _atlas!.View },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _emitterBuffer! } },
                    new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = _kindBuffer! } },
                },
            });
        }

        /// <summary>Upload this frame's emitters (replacing the last frame's). Returns how many are drawn: the rest beyond <see cref="MaxEmitters"/> are dropped.</summary>
        public int SetEmitters(ReadOnlySpan<ParticleEmitterInstance> emitters)
        {
            if (_queue == null || _emitterBuffer == null) throw new InvalidOperationException("Call Init first");
            int count = Math.Min(emitters.Length, MaxEmitters);
            DroppedEmitters = emitters.Length - count;
            EnsureCapacity(count);
            EmitterCount = count;
            if (count == 0) return 0;
            MemoryMarshal.AsBytes(emitters[..count]).CopyTo(_upload);
            _queue.WriteBuffer(_emitterBuffer, 0, _upload, 0, count * ParticleEmitterInstance.SizeInBytes);
            return count;
        }

        /// <summary>Draw every emitter uploaded by the last <see cref="SetEmitters"/>. The block pipeline's <see cref="BlockRenderPipeline.BeginFrame"/> must have run this frame.</summary>
        public void Draw(GPURenderPassEncoder pass)
        {
            if (_pipeline == null || _bindGroup == null) throw new InvalidOperationException("Call Init first");
            if (EmitterCount == 0) return;
            pass.SetPipeline(_pipeline);
            pass.SetBindGroup(0, _bindGroup);
            pass.Draw(6, (uint)(EmitterCount * MaxPerEmitter), 0, 0);
        }

        public void Dispose()
        {
            _bindGroup?.Dispose();
            _pipeline?.Dispose();
            _pipelineLayout?.Dispose();
            _layout?.Dispose();
            _emitterBuffer?.Destroy(); _emitterBuffer?.Dispose();
            _kindBuffer?.Destroy(); _kindBuffer?.Dispose();
            _ownedPlaceholder?.Dispose();
        }
    }
}
