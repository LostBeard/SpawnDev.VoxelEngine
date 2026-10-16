using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>What the weather looks like this frame, 144 bytes, matching the WGSL <c>Weather</c> struct.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct WeatherUniforms
    {
        /// <summary>The frame's view-projection (reversed-Z, as <see cref="BlockFrameUniforms.ViewProjection"/>).</summary>
        public Matrix4x4 ViewProjection;
        /// <summary>The camera's place (xyz) and the clock in ticks (w; the streaks fall with it, keep it small, a few thousand a time).</summary>
        public Vector4 Camera;
        /// <summary>x how hard it falls (0..1: the streaks' opacity), y how bright the light is (0..1: a night's rain is dark), z the window's side in columns, w how far above and below the camera a streak reaches (blocks).</summary>
        public Vector4 Params;
        /// <summary>The world x and z of the window's first column (xy; whole numbers).</summary>
        public Vector4 Window;
        /// <summary>Where the rain streaks are in the atlas (x, y, width, height pixels).</summary>
        public Vector4 RainRect;
        /// <summary>Where the snow flakes are in the atlas (x, y, width, height pixels).</summary>
        public Vector4 SnowRect;

        public const int SizeInBytes = 144;
    }

    /// <summary>
    /// Rain and snow, as the game draws them: a streak for each column of a window round the camera, a vertical quad that turns to face the camera and reaches from the ground (or from a little below the
    /// camera) to a little above it, showing a texture of falling streaks (or flakes) that slides down it with the clock. A column with no sky over it (a roof, a cave) has its ground above the
    /// quad's top, so nothing falls there. The caller says, for each column of the window, how high its ground is and what falls on it (nothing, rain, snow): <see cref="SetCells"/>, one 32-bit
    /// word a column, <c>(kind &lt;&lt; 16) | (ground y + 64)</c>. Drawn last in the world's pass, blended, depth tested and not written.
    /// </summary>
    public sealed class WeatherPipeline : IDisposable
    {
        /// <summary>The widest window, in columns.</summary>
        public const int MaxWindow = 64;
        public const uint KindNone = 0, KindRain = 1, KindSnow = 2;

        private GPUDevice? _device;
        private GPUQueue? _queue;
        private GPUBuffer? _uniforms, _cells;
        private GPUBindGroupLayout? _layout;
        private GPUPipelineLayout? _pipelineLayout;
        private GPUBindGroup? _bindGroup;
        private GPURenderPipeline? _pipeline;
        private EntityTextureAtlas? _atlas, _ownedPlaceholder;
        private int _size;
        public bool IsReady => _pipeline != null;

        public void Init(GPUDevice device, GPUQueue queue, string colorFormat)
        {
            _device = device;
            _queue = queue;
            _uniforms = device.CreateBuffer(new GPUBufferDescriptor { Size = WeatherUniforms.SizeInBytes, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
            _cells = device.CreateBuffer(new GPUBufferDescriptor { Size = (ulong)(MaxWindow * MaxWindow * 4), Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst });
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
            using var shader = device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = WeatherShaders.Source });
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
                                Color = new GPUBlendComponent { SrcFactor = "src-alpha", DstFactor = "one-minus-src-alpha", Operation = "add" },
                                Alpha = new GPUBlendComponent { SrcFactor = "one", DstFactor = "one-minus-src-alpha", Operation = "add" },
                            },
                        },
                    },
                },
                Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList, CullMode = GPUCullMode.None },
                DepthStencil = new GPUDepthStencilState { Format = ReversedZHelper.DepthFormat, DepthWriteEnabled = false, DepthCompare = ReversedZHelper.DepthCompare },
            });
            _ownedPlaceholder = EntityTextureAtlas.FromRgba(device, queue, new byte[] { 0, 0, 0, 0 }, 1, 1);
            SetTextures(_ownedPlaceholder);
        }

        /// <summary>Use this atlas (the rain streaks and snow flakes). The caller keeps ownership and must keep it alive while the pipeline can draw.</summary>
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
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _cells! } },
                },
            });
        }

        /// <summary>The window's columns: <paramref name="cells"/> holds <paramref name="size"/> squared words, row after row (z), each column along x.</summary>
        public void SetCells(ReadOnlySpan<uint> cells, int size)
        {
            if (_queue == null) throw new InvalidOperationException("Call Init first");
            if (size < 1 || size > MaxWindow || cells.Length < size * size) throw new ArgumentOutOfRangeException(nameof(size));
            _size = size;
            var bytes = MemoryMarshal.AsBytes(cells[..(size * size)]).ToArray();
            _queue.WriteBuffer(_cells!, 0, bytes);
        }

        /// <summary>Upload this frame's weather and draw it. Call last in the pass (after the water): the streaks are blended over everything.</summary>
        public void Draw(GPURenderPassEncoder pass, in WeatherUniforms weather)
        {
            if (_pipeline == null || _bindGroup == null || _queue == null) throw new InvalidOperationException("Call Init first");
            if (_size == 0) return;
            var bytes = new byte[WeatherUniforms.SizeInBytes];
            var copy = weather;
            copy.Params.Z = _size;
            unsafe { fixed (byte* p = bytes) *(WeatherUniforms*)p = copy; }
            _queue.WriteBuffer(_uniforms!, 0, bytes);
            pass.SetPipeline(_pipeline);
            pass.SetBindGroup(0, _bindGroup);
            pass.Draw(6, (uint)(_size * _size), 0, 0);
        }

        public void Dispose()
        {
            _bindGroup?.Dispose();
            _pipeline?.Dispose();
            _pipelineLayout?.Dispose();
            _layout?.Dispose();
            _uniforms?.Destroy(); _uniforms?.Dispose();
            _cells?.Destroy(); _cells?.Dispose();
            _ownedPlaceholder?.Dispose();
        }
    }
}
