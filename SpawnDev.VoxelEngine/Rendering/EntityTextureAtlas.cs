using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// The textures of everything that is not a block (creatures, players) as ONE 2D texture: any mix of image sizes packed by the caller, read
    /// by <see cref="EntityRenderPipeline"/> with exact texel fetches in atlas PIXELS. A part names the pixel where its texture's box layout
    /// starts, so no table of sizes is needed on the GPU. Regions can be rewritten while the atlas is in use (a player's skin arriving).
    /// </summary>
    public sealed class EntityTextureAtlas : IDisposable
    {
        public GPUTexture Texture { get; private set; }
        public GPUTextureView View { get; private set; }
        public int Width { get; }
        public int Height { get; }
        private readonly GPUQueue _queue;

        private EntityTextureAtlas(GPUQueue queue, GPUTexture texture, int width, int height)
        {
            _queue = queue;
            Texture = texture;
            View = texture.CreateView();
            Width = width;
            Height = height;
        }

        /// <summary>An atlas of this size, every texel transparent black (nothing is drawn where nothing was written).</summary>
        public static EntityTextureAtlas Create(GPUDevice device, GPUQueue queue, int width, int height)
        {
            if (width <= 0 || height <= 0) throw new ArgumentException($"{width}x{height} is not an atlas size");
            var texture = device.CreateTexture(new GPUTextureDescriptor
            {
                Size = new[] { width, height },
                Format = "rgba8unorm",
                Usage = GPUTextureUsage.TextureBinding | GPUTextureUsage.CopyDst,
            });
            return new EntityTextureAtlas(queue, texture, width, height);
        }

        /// <summary>An atlas holding this RGBA8 image. The pixels stay in the JS typed array all the way to the GPU.</summary>
        public static EntityTextureAtlas FromRgba(GPUDevice device, GPUQueue queue, Uint8Array rgba, int width, int height)
        {
            if ((long)width * height * 4 != rgba.Length) throw new ArgumentException($"{rgba.Length} bytes is not a {width}x{height} RGBA image");
            var atlas = Create(device, queue, width, height);
            atlas.WriteRegion(0, 0, width, height, rgba);
            return atlas;
        }

        /// <summary>The same for pixels already in .NET memory (tests, tools).</summary>
        public static EntityTextureAtlas FromRgba(GPUDevice device, GPUQueue queue, byte[] rgba, int width, int height)
        {
            if ((long)width * height * 4 != rgba.Length) throw new ArgumentException($"{rgba.Length} bytes is not a {width}x{height} RGBA image");
            using var js = new Uint8Array(rgba);
            return FromRgba(device, queue, js, width, height);
        }

        /// <summary>Overwrite a rectangle of the atlas with RGBA8 pixels (rows tightly packed, <paramref name="width"/> x <paramref name="height"/>).</summary>
        public void WriteRegion(int x, int y, int width, int height, Uint8Array rgba)
        {
            CheckRegion(x, y, width, height);
            if ((long)width * height * 4 != rgba.Length) throw new ArgumentException($"{rgba.Length} bytes is not a {width}x{height} RGBA image");
            _queue.WriteTexture(
                new GPUTexelCopyTextureInfo { Texture = Texture, Origin = new GPUOrigin3DDict { X = (uint)x, Y = (uint)y } },
                rgba,
                new GPUTexelCopyBufferLayout { BytesPerRow = (uint)(width * 4), RowsPerImage = (uint)height },
                new GPUExtent3DDict { Width = (uint)width, Height = (uint)height });
        }

        /// <summary>The same for pixels already in .NET memory.</summary>
        public void WriteRegion(int x, int y, int width, int height, byte[] rgba)
        {
            CheckRegion(x, y, width, height);
            if ((long)width * height * 4 != rgba.Length) throw new ArgumentException($"{rgba.Length} bytes is not a {width}x{height} RGBA image");
            using var js = new Uint8Array(rgba);
            WriteRegion(x, y, width, height, js);
        }

        private void CheckRegion(int x, int y, int width, int height)
        {
            if (x < 0 || y < 0 || width <= 0 || height <= 0 || x + width > Width || y + height > Height)
                throw new ArgumentOutOfRangeException(nameof(x), $"The {width}x{height} region at ({x},{y}) is not inside the {Width}x{Height} atlas");
        }

        public void Dispose()
        {
            View?.Dispose();
            Texture?.Destroy();
            Texture?.Dispose();
        }
    }
}
