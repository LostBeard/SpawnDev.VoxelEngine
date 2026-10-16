using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// The colour of grass, leaves and water across the world, for <see cref="BlockRenderPipeline"/>: three 256 x 256 maps, one texel for every 4 x 4 blocks (a biome is as fine as that), so the maps
    /// cover 1,024 x 1,024 blocks and WRAP: a column 64 columns away lands on the same texels, which the draw distance never shows at once. A block whose material says it takes a biome tint
    /// (<see cref="BlockMaterial.TintGrass"/> and friends) is multiplied, per pixel, by the map's colour at the pixel's place, sampled with linear filtering - so the border of two biomes is a soft
    /// blend, as it is in the game, and not a staircase of 4-block steps. Unwritten texels hold the colours the map was created with (plains), so ground not heard of yet still looks like grass.
    /// </summary>
    public sealed class BiomeTintMap : IDisposable
    {
        /// <summary>Texels across a map: 256 cells of <see cref="CellBlocks"/> blocks.</summary>
        public const int CellsAcross = 256;
        public const int CellBlocks = 4;
        /// <summary>A column is 16 blocks: 4 x 4 cells.</summary>
        public const int ColumnCells = 4;
        /// <summary>Blocks the maps cover before wrapping (and the shader's divisor).</summary>
        public const int BlocksAcross = CellsAcross * CellBlocks;

        public EntityTextureAtlas Grass { get; }
        public EntityTextureAtlas Foliage { get; }
        public EntityTextureAtlas Water { get; }
        public GPUSampler Sampler { get; }
        private bool _disposed;

        private BiomeTintMap(EntityTextureAtlas grass, EntityTextureAtlas foliage, EntityTextureAtlas water, GPUSampler sampler)
        {
            Grass = grass; Foliage = foliage; Water = water; Sampler = sampler;
        }

        /// <summary>The vanilla plains colours: grass, foliage, water.</summary>
        public static readonly (uint grass, uint foliage, uint water) PlainsColours = (0x91BD59, 0x77AB2F, 0x3F76E4);

        /// <summary>Maps filled with these colours (0xRRGGBB) everywhere.</summary>
        public static BiomeTintMap Create(GPUDevice device, GPUQueue queue, uint grass = 0x91BD59, uint foliage = 0x77AB2F, uint water = 0x3F76E4)
        {
            EntityTextureAtlas Filled(uint rgb)
            {
                var atlas = EntityTextureAtlas.Create(device, queue, CellsAcross, CellsAcross);
                var pixels = new byte[CellsAcross * CellsAcross * 4];
                for (int i = 0; i < pixels.Length; i += 4) { pixels[i] = (byte)(rgb >> 16); pixels[i + 1] = (byte)(rgb >> 8); pixels[i + 2] = (byte)rgb; pixels[i + 3] = 255; }
                atlas.WriteRegion(0, 0, CellsAcross, CellsAcross, pixels);
                return atlas;
            }
            var sampler = device.CreateSampler(new GPUSamplerDescriptor
            {
                MinFilter = "linear",
                MagFilter = "linear",
                AddressModeU = "repeat",
                AddressModeV = "repeat",
            });
            return new BiomeTintMap(Filled(grass), Filled(foliage), Filled(water), sampler);
        }

        /// <summary>Texel of a block coordinate: the cell it falls in, wrapped into the map.</summary>
        public static int CellOf(int block) => (((block >> 2) % CellsAcross) + CellsAcross) % CellsAcross;

        /// <summary>
        /// Set the colours of the 4 x 4 cells of column (<paramref name="cx"/>, <paramref name="cz"/>) (block x = cx * 16): each span holds 16 colours (0xRRGGBB), cell (i, j) at j * 4 + i with i along x and j along z.
        /// </summary>
        public void SetColumn(int cx, int cz, ReadOnlySpan<uint> grass, ReadOnlySpan<uint> foliage, ReadOnlySpan<uint> water)
        {
            if (grass.Length != 16 || foliage.Length != 16 || water.Length != 16) throw new ArgumentException("A column has 16 cells");
            int x0 = CellOf(cx * 16), z0 = CellOf(cz * 16);
            Write(Grass, x0, z0, grass);
            Write(Foliage, x0, z0, foliage);
            Write(Water, x0, z0, water);
        }

        private static void Write(EntityTextureAtlas atlas, int x0, int z0, ReadOnlySpan<uint> rgb)
        {
            var bytes = new byte[16 * 4];
            for (int i = 0; i < 16; i++)
            {
                bytes[i * 4] = (byte)(rgb[i] >> 16);
                bytes[i * 4 + 1] = (byte)(rgb[i] >> 8);
                bytes[i * 4 + 2] = (byte)rgb[i];
                bytes[i * 4 + 3] = 255;
            }
            atlas.WriteRegion(x0, z0, ColumnCells, ColumnCells, bytes);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Grass.Dispose(); Foliage.Dispose(); Water.Dispose();
            Sampler.Dispose();
        }
    }
}
