using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// The block textures as one GPU 2D texture array (one square tile per layer), the form <see cref="BlockRenderPipeline"/> samples.
    ///
    /// A texture array with a repeating sampler tiles a texture across a greedy-merged quad; an atlas cannot. Most games keep their
    /// textures as an atlas (a grid of tiles in one image), so <see cref="FromAtlas(GPUDevice, GPUQueue, Uint8Array, int, int, int)"/>
    /// slices one into layers ON THE GPU: the atlas is uploaded once and each tile is copied into its layer with
    /// copyTextureToTexture. Layer = tileY * tilesX + tileX (row-major from the top-left), so an atlas tile index is the layer index.
    /// </summary>
    public sealed class BlockTextureArray : IDisposable
    {
        public GPUTexture Texture { get; private set; }
        /// <summary>A "2d-array" view of every layer.</summary>
        public GPUTextureView View { get; private set; }
        public int TileSize { get; }
        public int LayerCount { get; }

        private BlockTextureArray(GPUTexture texture, GPUTextureView view, int tileSize, int layerCount)
        {
            Texture = texture;
            View = view;
            TileSize = tileSize;
            LayerCount = layerCount;
        }

        /// <summary>
        /// Slice an RGBA8 atlas (<paramref name="atlasWidth"/> x <paramref name="atlasHeight"/>, a grid of <paramref name="tileSize"/> square
        /// tiles) into a texture array. The pixels stay in the JS typed array all the way to the GPU: no copy into the .NET heap.
        /// </summary>
        public static BlockTextureArray FromAtlas(GPUDevice device, GPUQueue queue, Uint8Array atlasRgba,
            int atlasWidth, int atlasHeight, int tileSize)
        {
            ValidateAtlas(atlasRgba.Length, atlasWidth, atlasHeight, tileSize);

            int tilesX = atlasWidth / tileSize, tilesY = atlasHeight / tileSize;
            int layers = tilesX * tilesY;

            var atlas = device.CreateTexture(new GPUTextureDescriptor
            {
                Size = new[] { atlasWidth, atlasHeight },
                Format = "rgba8unorm",
                Usage = GPUTextureUsage.TextureBinding | GPUTextureUsage.CopyDst | GPUTextureUsage.CopySrc,
            });
            var array = device.CreateTexture(new GPUTextureDescriptor
            {
                Size = new[] { tileSize, tileSize, layers },
                Format = "rgba8unorm",
                Usage = GPUTextureUsage.TextureBinding | GPUTextureUsage.CopyDst,
            });
            try
            {
                queue.WriteTexture(
                    new GPUTexelCopyTextureInfo { Texture = atlas },
                    atlasRgba,
                    new GPUTexelCopyBufferLayout { BytesPerRow = (uint)(atlasWidth * 4), RowsPerImage = (uint)atlasHeight },
                    new GPUExtent3DDict { Width = (uint)atlasWidth, Height = (uint)atlasHeight });

                using var encoder = device.CreateCommandEncoder();
                for (int ty = 0; ty < tilesY; ty++)
                    for (int tx = 0; tx < tilesX; tx++)
                        encoder.CopyTextureToTexture(
                            new GPUTexelCopyTextureInfo { Texture = atlas, Origin = new GPUOrigin3DDict { X = (uint)(tx * tileSize), Y = (uint)(ty * tileSize), Z = 0 } },
                            new GPUTexelCopyTextureInfo { Texture = array, Origin = new GPUOrigin3DDict { X = 0, Y = 0, Z = (uint)(ty * tilesX + tx) } },
                            new GPUExtent3DDict { Width = (uint)tileSize, Height = (uint)tileSize, DepthOrArrayLayers = 1 });
                using var commands = encoder.Finish();
                queue.Submit(new[] { commands });
            }
            finally
            {
                // The submitted copies keep the atlas alive until they run; destroy only releases it after that.
                atlas.Destroy();
                atlas.Dispose();
            }

            return new BlockTextureArray(array, array.CreateView(new GPUTextureViewDescriptor { Dimension = "2d-array" }), tileSize, layers);
        }

        /// <summary>The same as <see cref="FromAtlas(GPUDevice, GPUQueue, Uint8Array, int, int, int)"/> for pixels that are already in .NET memory (tests, tools).</summary>
        public static BlockTextureArray FromAtlas(GPUDevice device, GPUQueue queue, byte[] atlasRgba,
            int atlasWidth, int atlasHeight, int tileSize)
        {
            // Checked before anything crosses to JavaScript, so a bad atlas fails the same way on every host
            ValidateAtlas(atlasRgba.Length, atlasWidth, atlasHeight, tileSize);
            using var js = new Uint8Array(atlasRgba);
            return FromAtlas(device, queue, js, atlasWidth, atlasHeight, tileSize);
        }

        private static void ValidateAtlas(long byteLength, int atlasWidth, int atlasHeight, int tileSize)
        {
            if (tileSize <= 0 || atlasWidth <= 0 || atlasHeight <= 0 || atlasWidth % tileSize != 0 || atlasHeight % tileSize != 0)
                throw new ArgumentException($"The {atlasWidth}x{atlasHeight} atlas is not a whole number of {tileSize}px tiles");
            if ((long)atlasWidth * atlasHeight * 4 != byteLength)
                throw new ArgumentException($"{byteLength} bytes is not a {atlasWidth}x{atlasHeight} RGBA image");
        }

        /// <summary>A 1 x 1 white single-layer array: a valid binding before any real textures are loaded.</summary>
        public static BlockTextureArray White(GPUDevice device, GPUQueue queue)
            => FromAtlas(device, queue, new byte[] { 255, 255, 255, 255 }, 1, 1, 1);

        public void Dispose()
        {
            View?.Dispose();
            Texture?.Destroy();
            Texture?.Dispose();
        }
    }
}
