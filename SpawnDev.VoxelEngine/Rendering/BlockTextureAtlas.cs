using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// The block textures as ONE 2D texture: a grid of equal square tiles, the form <see cref="BlockRenderPipeline"/> reads.
    /// A tile's number is its place in the grid, row-major from the top-left (tileY * <see cref="TilesPerRow"/> + tileX), and that
    /// number is what a <see cref="BlockMaterial"/> face names.
    ///
    /// The shader fetches texels exactly (no sampler) and tiles a tile across a greedy-merged quad itself, so the atlas has none of a
    /// texture array's limits: WebGPU guarantees only 256 array layers, a game's block textures (with every frame of every animated
    /// one) are several thousand tiles, and a 2D texture holds as many as its side allows (8192 px = 512 tiles of 16 px a side).
    /// </summary>
    public sealed class BlockTextureAtlas : IDisposable
    {
        public GPUTexture Texture { get; private set; }
        /// <summary>A 2D view of the whole atlas.</summary>
        public GPUTextureView View { get; private set; }
        /// <summary>Side of one tile in pixels.</summary>
        public int TileSize { get; }
        public int TilesPerRow { get; }
        public int TileRows { get; }
        public int TileCount => TilesPerRow * TileRows;

        private BlockTextureAtlas(GPUTexture texture, GPUTextureView view, int tileSize, int tilesPerRow, int tileRows)
        {
            Texture = texture;
            View = view;
            TileSize = tileSize;
            TilesPerRow = tilesPerRow;
            TileRows = tileRows;
        }

        /// <summary>
        /// Upload an RGBA8 image (<paramref name="width"/> x <paramref name="height"/>, a grid of <paramref name="tileSize"/> square tiles) as the
        /// atlas. The pixels stay in the JS typed array all the way to the GPU: no copy into the .NET heap.
        /// </summary>
        public static BlockTextureAtlas FromRgba(GPUDevice device, GPUQueue queue, Uint8Array rgba, int width, int height, int tileSize)
        {
            Validate(rgba.Length, width, height, tileSize);
            var texture = device.CreateTexture(new GPUTextureDescriptor
            {
                Size = new[] { width, height },
                Format = "rgba8unorm",
                Usage = GPUTextureUsage.TextureBinding | GPUTextureUsage.CopyDst,
            });
            queue.WriteTexture(
                new GPUTexelCopyTextureInfo { Texture = texture },
                rgba,
                new GPUTexelCopyBufferLayout { BytesPerRow = (uint)(width * 4), RowsPerImage = (uint)height },
                new GPUExtent3DDict { Width = (uint)width, Height = (uint)height });
            return new BlockTextureAtlas(texture, texture.CreateView(), tileSize, width / tileSize, height / tileSize);
        }

        /// <summary>The same for pixels that are already in .NET memory (tests, tools).</summary>
        public static BlockTextureAtlas FromRgba(GPUDevice device, GPUQueue queue, byte[] rgba, int width, int height, int tileSize)
        {
            // Checked before anything crosses to JavaScript, so a bad atlas fails the same way on every host
            Validate(rgba.Length, width, height, tileSize);
            using var js = new Uint8Array(rgba);
            return FromRgba(device, queue, js, width, height, tileSize);
        }

        private static void Validate(long byteLength, int width, int height, int tileSize)
        {
            if (tileSize <= 0 || width <= 0 || height <= 0 || width % tileSize != 0 || height % tileSize != 0)
                throw new ArgumentException($"The {width}x{height} atlas is not a whole number of {tileSize}px tiles");
            if ((long)width * height * 4 != byteLength)
                throw new ArgumentException($"{byteLength} bytes is not a {width}x{height} RGBA image");
        }

        /// <summary>A 1 x 1 white one-tile atlas: a valid binding before any real textures are loaded.</summary>
        public static BlockTextureAtlas White(GPUDevice device, GPUQueue queue)
            => FromRgba(device, queue, new byte[] { 255, 255, 255, 255 }, 1, 1, 1);

        public void Dispose()
        {
            View?.Dispose();
            Texture?.Destroy();
            Texture?.Dispose();
        }
    }

    /// <summary>
    /// A tile that plays through other tiles: wherever a face names <see cref="BaseTile"/>, the pipeline draws
    /// <c>Timeline[tick % Timeline.Length]</c> instead, with <c>tick = floor(seconds * 20)</c> (Minecraft's tick). A timeline lists one tile per
    /// TICK, so frames that last several ticks, repeat, or play out of order are all just a longer list. Set with
    /// <see cref="BlockRenderPipeline.SetAnimations"/>; the time comes from <see cref="BlockFrameUniforms.Time"/>.
    /// </summary>
    public sealed record TextureAnimation(int BaseTile, int[] Timeline);
}
