using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Fills the smaller levels of a block atlas: each level is the one above it with every 2 x 2 texels averaged (colours weighted by their alpha, so the see-through
    /// texels of leaves and litter do not darken what is left), done on the GPU in one small pass per level - the pixels never come into .NET. Tiles are a power of two
    /// across and sit on even coordinates, so a level is exactly each tile shrunk on its own: nothing of one tile bleeds into the next, down to one texel a tile.
    /// </summary>
    internal static class AtlasMipGenerator
    {
        private const string Source = """
            @group(0) @binding(0) var src: texture_2d<f32>;
            @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
                return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
            }
            @fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                let at = vec2<i32>(i32(pos.x), i32(pos.y)) * 2;
                let a = textureLoad(src, at, 0);
                let b = textureLoad(src, at + vec2<i32>(1, 0), 0);
                let c = textureLoad(src, at + vec2<i32>(0, 1), 0);
                let d = textureLoad(src, at + vec2<i32>(1, 1), 0);
                let weight = a.a + b.a + c.a + d.a;
                var rgb = (a.rgb + b.rgb + c.rgb + d.rgb) * 0.25;
                if (weight > 0.0001) { rgb = (a.rgb * a.a + b.rgb * b.a + c.rgb * c.a + d.rgb * d.a) / weight; }
                return vec4<f32>(rgb, weight * 0.25);
            }
            """;

        /// <summary>How many levels an atlas of tiles <paramref name="tileSize"/> px across has: the tile halves while it is still even, down to one texel at most.</summary>
        public static int LevelsFor(int tileSize)
        {
            int levels = 1;
            while (tileSize > 1 && tileSize % 2 == 0) { tileSize /= 2; levels++; }
            return levels;
        }

        /// <summary>Fill levels 1.. of <paramref name="texture"/> from level 0 (<paramref name="levels"/> levels in all).</summary>
        public static void Generate(GPUDevice device, GPUQueue queue, GPUTexture texture, int levels)
        {
            if (levels <= 1) return;
            using var shader = device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = Source });
            using var layout = device.CreateBindGroupLayout(new GPUBindGroupLayoutDescriptor
            {
                Entries = new GPUBindGroupLayoutEntry[]
                {
                    new() { Binding = 0, Visibility = GPUShaderStageFlags.FRAGMENT, Texture = new GPUTextureBindingLayout { SampleType = "float", ViewDimension = "2d" } },
                },
            });
            using var pipelineLayout = device.CreatePipelineLayout(new GPUPipelineLayoutDescriptor { BindGroupLayouts = new[] { layout } });
            using var pipeline = device.CreateRenderPipeline(new GPURenderPipelineDescriptor
            {
                Layout = pipelineLayout,
                Vertex = new GPUVertexState { Module = shader, EntryPoint = "vs_main" },
                Fragment = new GPUFragmentState { Module = shader, EntryPoint = "fs_main", Targets = new[] { new GPUColorTargetState { Format = "rgba8unorm" } } },
                Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList, CullMode = GPUCullMode.None },
            });
            using var encoder = device.CreateCommandEncoder();
            for (int level = 1; level < levels; level++)
            {
                using var from = texture.CreateView(new GPUTextureViewDescriptor { BaseMipLevel = level - 1, MipLevelCount = 1 });
                using var to = texture.CreateView(new GPUTextureViewDescriptor { BaseMipLevel = level, MipLevelCount = 1 });
                using var bind = device.CreateBindGroup(new GPUBindGroupDescriptor
                {
                    Layout = layout,
                    Entries = new GPUBindGroupEntry[] { new() { Binding = 0, Resource = from } },
                });
                using var pass = encoder.BeginRenderPass(new GPURenderPassDescriptor
                {
                    ColorAttachments = new[] { new GPURenderPassColorAttachment { View = to, LoadOp = GPULoadOp.Clear, StoreOp = GPUStoreOp.Store, ClearValue = new double[] { 0, 0, 0, 0 } } },
                });
                pass.SetPipeline(pipeline);
                pass.SetBindGroup(0, bind);
                pass.Draw(3, 1, 0, 0);
                pass.End();
            }
            using var commands = encoder.Finish();
            queue.Submit(new[] { commands });
        }
    }
}
