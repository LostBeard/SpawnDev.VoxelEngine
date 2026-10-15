using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// One textured box of a model, placed: what <see cref="EntityRenderPipeline"/> draws, 128 bytes. The box is in MODEL pixels in
    /// Minecraft's model space (y points DOWN, a block is 16 px): <see cref="Origin"/> is its minimum corner, <see cref="Size"/> its width,
    /// height and depth. <see cref="Model"/> takes model pixels to world space (a .NET matrix, row-vector convention: <c>v * Model</c>), with
    /// the part's pivot, its animation rotation, the entity's placement and the 1/16 scale all folded in.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct EntityPartInstance
    {
        public Matrix4x4 Model;
        /// <summary>The box's minimum corner (xyz, model px) and how far every face is pushed outward (w, px: an overlay layer).</summary>
        public Vector4 OriginInflate;
        /// <summary>The box's width, height, depth (xyz, px) and the flags (w): <see cref="FlagMirror"/>.</summary>
        public Vector4 SizeFlags;
        /// <summary>Where the box's texture layout starts in the atlas, in pixels (xy); zw unused.</summary>
        public Vector4 Uv;
        /// <summary>Colour multiplied into the texture (rgb): a dye, a hurt flash. White = unchanged. w unused.</summary>
        public Vector4 Tint;

        public const int SizeInBytes = 128;
        /// <summary>The box's texture is the mirror image (the left limbs of a model reuse the right limbs' texture).</summary>
        public const float FlagMirror = 1f;
    }

    /// <summary>
    /// Draws the creatures of a world: every part of every model, each a textured box, in ONE instanced draw. The box is expanded in the
    /// vertex shader (36 vertices an instance), so the CPU's whole job is one <see cref="EntityPartInstance"/> a box and one buffer upload.
    ///
    /// The texture of a box is laid out as Minecraft lays out a model's boxes (the cross unfolding: top and bottom over the four sides, from the
    /// box's texture origin), so a model's original texture offsets work unchanged. Texels are fetched exactly (<see cref="EntityTextureAtlas"/>),
    /// alpha under 0.1 is discarded (overlay layers: a hat, a coat, wool).
    ///
    /// Lighting and fog come from the SAME <see cref="BlockFrameUniforms"/> buffer as the blocks (<see cref="BlockRenderPipeline.FrameBuffer"/>),
    /// so a creature is lit and fogged like the ground it stands on. Draw it inside the block pipeline's render pass, after the opaque and plant
    /// passes (reversed-Z, depth tested and written) and before the water.
    /// </summary>
    public sealed class EntityRenderPipeline : IDisposable
    {
        private GPUDevice? _device;
        private GPUQueue? _queue;
        private GPUBuffer? _frameBuffer, _partBuffer;
        private GPUBindGroupLayout? _layout;
        private GPUPipelineLayout? _pipelineLayout;
        private GPUBindGroup? _bindGroup;
        private GPURenderPipeline? _pipeline;
        private EntityTextureAtlas? _atlas;
        private EntityTextureAtlas? _ownedPlaceholder;
        private byte[] _upload = System.Array.Empty<byte>();
        private int _capacity;

        /// <summary>Parts the last <see cref="SetParts"/> uploaded (what <see cref="Draw"/> draws).</summary>
        public int PartCount { get; private set; }
        /// <summary>The most parts one frame can hold (<see cref="SetParts"/> drops the rest and says so).</summary>
        public const int MaxParts = 1 << 17;
        /// <summary>Parts the last <see cref="SetParts"/> had to leave out because of <see cref="MaxParts"/>.</summary>
        public int DroppedParts { get; private set; }
        public bool IsReady => _pipeline != null;

        /// <param name="frameUniforms">The block pipeline's frame buffer (<see cref="BlockRenderPipeline.FrameBuffer"/>).</param>
        public void Init(GPUDevice device, GPUQueue queue, string colorFormat, GPUBuffer frameUniforms, int initialParts = 1024)
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
                },
            });
            _pipelineLayout = device.CreatePipelineLayout(new GPUPipelineLayoutDescriptor { BindGroupLayouts = new[] { _layout } });
            using var shader = device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = EntityRenderShaders.Source });
            _pipeline = device.CreateRenderPipeline(new GPURenderPipelineDescriptor
            {
                Layout = _pipelineLayout,
                Vertex = new GPUVertexState { Module = shader, EntryPoint = "vs_main" },
                Fragment = new GPUFragmentState { Module = shader, EntryPoint = "fs_main", Targets = new[] { new GPUColorTargetState { Format = colorFormat } } },
                // No culling: thin parts (wings, ears) are seen from both sides, and a model's winding is not worth a bug
                Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList, CullMode = GPUCullMode.None, FrontFace = GPUFrontFace.CCW },
                DepthStencil = new GPUDepthStencilState
                {
                    Format = ReversedZHelper.DepthFormat,
                    DepthWriteEnabled = true,
                    DepthCompare = ReversedZHelper.DepthCompare,
                },
            });

            EnsureCapacity(Math.Max(initialParts, 1));
            _ownedPlaceholder = EntityTextureAtlas.FromRgba(device, queue, new byte[] { 255, 0, 255, 255 }, 1, 1);
            SetTextures(_ownedPlaceholder);
        }

        /// <summary>Use this atlas. The caller keeps ownership and must keep it alive while the pipeline can draw.</summary>
        public void SetTextures(EntityTextureAtlas atlas)
        {
            if (_device == null) throw new InvalidOperationException("Call Init first");
            _atlas = atlas;
            RebuildBindGroup();
        }

        private void EnsureCapacity(int parts)
        {
            if (parts <= _capacity) return;
            int capacity = Math.Max(parts, _capacity * 2);
            _partBuffer?.Destroy(); _partBuffer?.Dispose();
            _partBuffer = _device!.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)capacity * EntityPartInstance.SizeInBytes,
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
            });
            _capacity = capacity;
            _upload = new byte[capacity * EntityPartInstance.SizeInBytes];
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
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _partBuffer! } },
                },
            });
        }

        /// <summary>Upload this frame's parts (replacing the last frame's). Returns how many are drawn: the rest beyond <see cref="MaxParts"/> are dropped.</summary>
        public int SetParts(ReadOnlySpan<EntityPartInstance> parts)
        {
            if (_queue == null || _partBuffer == null) throw new InvalidOperationException("Call Init first");
            int count = Math.Min(parts.Length, MaxParts);
            DroppedParts = parts.Length - count;
            EnsureCapacity(count);
            PartCount = count;
            if (count == 0) return 0;
            MemoryMarshal.AsBytes(parts[..count]).CopyTo(_upload);
            _queue.WriteBuffer(_partBuffer, 0, _upload, 0, count * EntityPartInstance.SizeInBytes);
            return count;
        }

        /// <summary>Draw every part uploaded by the last <see cref="SetParts"/>. The block pipeline's <see cref="BlockRenderPipeline.BeginFrame"/> must have run this frame.</summary>
        public void Draw(GPURenderPassEncoder pass)
        {
            if (_pipeline == null || _bindGroup == null) throw new InvalidOperationException("Call Init first");
            if (PartCount == 0) return;
            pass.SetPipeline(_pipeline);
            pass.SetBindGroup(0, _bindGroup);
            pass.Draw(36, (uint)PartCount, 0, 0);
        }

        public void Dispose()
        {
            _bindGroup?.Dispose();
            _pipeline?.Dispose();
            _pipelineLayout?.Dispose();
            _layout?.Dispose();
            _partBuffer?.Destroy(); _partBuffer?.Dispose();
            _ownedPlaceholder?.Dispose();
        }
    }
}
