using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>What the sky looks like this frame, 208 bytes, matching the WGSL <c>Sky</c> struct.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SkyUniforms
    {
        /// <summary>Inverse of the view-projection matrix of the frame (the same as <see cref="BlockFrameUniforms.ViewProjection"/>): turns a pixel into a direction.</summary>
        public Matrix4x4 InverseViewProjection;
        public Vector4 Camera;
        /// <summary>Unit vector toward the sun (xyz) and how much daylight there is (w: 0 night .. 1 day).</summary>
        public Vector4 SunDirection;
        /// <summary>The colour overhead (rgb) and the moon's phase (w: 0 full .. 7, as the game numbers them).</summary>
        public Vector4 Zenith;
        /// <summary>The colour at the horizon (rgb: the fog's) and how night it is (w: 0 day .. 1 night, stars fade in with it).</summary>
        public Vector4 Horizon;
        /// <summary>Cloud layer: x its height, y how far it has slid along x (blocks), z the weather (0 clear .. 1 rain: clouds everywhere, no sun or stars), w the stars' turn (radians).</summary>
        public Vector4 Cloud;
        /// <summary>A cloud's colour (rgb) and opacity (w).</summary>
        public Vector4 CloudColor;
        /// <summary>Where the sun is in the atlas: x, y, width, height in pixels.</summary>
        public Vector4 SunRect;
        /// <summary>Where the FIRST moon phase is in the atlas (x, y, a frame's width, height): the eight phases follow, four to a row.</summary>
        public Vector4 MoonRect;
        /// <summary>Where the cloud map is in the atlas (x, y, width, height); one pixel of it is 12 blocks.</summary>
        public Vector4 CloudRect;

        public const int SizeInBytes = 208;
    }

    /// <summary>
    /// The sky behind the world: its gradient, a glow round the sun, the sun and the moon with its phases, stars, and a layer of clouds - one full-screen triangle drawn FIRST in the world's render
    /// pass (no depth test, no depth write), so terrain, creatures and water draw over it. The sun, moon and clouds are images from an atlas (<see cref="EntityTextureAtlas"/>) the caller makes from the game's
    /// own textures; <see cref="SkyUniforms"/> says where each is.
    /// </summary>
    public sealed class SkyPipeline : IDisposable
    {
        private GPUDevice? _device;
        private GPUQueue? _queue;
        private GPUBuffer? _uniforms;
        private GPUBindGroupLayout? _layout;
        private GPUPipelineLayout? _pipelineLayout;
        private GPUBindGroup? _bindGroup;
        private GPURenderPipeline? _pipeline;
        private EntityTextureAtlas? _atlas, _ownedPlaceholder;
        public bool IsReady => _pipeline != null;

        public void Init(GPUDevice device, GPUQueue queue, string colorFormat)
        {
            _device = device;
            _queue = queue;
            _uniforms = device.CreateBuffer(new GPUBufferDescriptor { Size = SkyUniforms.SizeInBytes, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
            _layout = device.CreateBindGroupLayout(new GPUBindGroupLayoutDescriptor
            {
                Entries = new GPUBindGroupLayoutEntry[]
                {
                    new() { Binding = 0, Visibility = GPUShaderStageFlags.VERTEX | GPUShaderStageFlags.FRAGMENT, Buffer = new GPUBufferBindingLayout { Type = "uniform" } },
                    new() { Binding = 1, Visibility = GPUShaderStageFlags.FRAGMENT, Texture = new GPUTextureBindingLayout { SampleType = "float", ViewDimension = "2d" } },
                },
            });
            _pipelineLayout = device.CreatePipelineLayout(new GPUPipelineLayoutDescriptor { BindGroupLayouts = new[] { _layout } });
            using var shader = device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = SkyShaders.Source });
            _pipeline = device.CreateRenderPipeline(new GPURenderPipelineDescriptor
            {
                Layout = _pipelineLayout,
                Vertex = new GPUVertexState { Module = shader, EntryPoint = "vs_main" },
                Fragment = new GPUFragmentState { Module = shader, EntryPoint = "fs_main", Targets = new[] { new GPUColorTargetState { Format = colorFormat } } },
                Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList, CullMode = GPUCullMode.None },
                // The pass has a depth attachment: the sky neither tests nor writes it
                DepthStencil = new GPUDepthStencilState { Format = ReversedZHelper.DepthFormat, DepthWriteEnabled = false, DepthCompare = "always" },
            });
            _ownedPlaceholder = EntityTextureAtlas.FromRgba(device, queue, new byte[] { 0, 0, 0, 0 }, 1, 1);
            SetTextures(_ownedPlaceholder);
        }

        /// <summary>Use this atlas (the sun, the moon's phases, the clouds). The caller keeps ownership and must keep it alive while the pipeline can draw.</summary>
        public void SetTextures(EntityTextureAtlas atlas)
        {
            if (_device == null) throw new InvalidOperationException("Call Init first");
            _atlas = atlas;
            _bindGroup?.Dispose();
            _bindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _layout!,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniforms! } },
                    new() { Binding = 1, Resource = atlas.View },
                },
            });
        }

        /// <summary>Upload this frame's sky and draw it. Call right after the pass begins, before anything else is drawn.</summary>
        public void Draw(GPURenderPassEncoder pass, in SkyUniforms sky)
        {
            if (_pipeline == null || _bindGroup == null || _queue == null) throw new InvalidOperationException("Call Init first");
            var bytes = new byte[SkyUniforms.SizeInBytes];
            var copy = sky;
            unsafe { fixed (byte* p = bytes) *(SkyUniforms*)p = copy; }
            _queue.WriteBuffer(_uniforms!, 0, bytes);
            pass.SetPipeline(_pipeline);
            pass.SetBindGroup(0, _bindGroup);
            pass.Draw(3, 1, 0, 0);
        }

        public void Dispose()
        {
            _bindGroup?.Dispose();
            _pipeline?.Dispose();
            _pipelineLayout?.Dispose();
            _layout?.Dispose();
            _uniforms?.Destroy(); _uniforms?.Dispose();
            _ownedPlaceholder?.Dispose();
        }
    }
}
