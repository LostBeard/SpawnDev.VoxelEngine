using System.Numerics;
using System.Runtime.InteropServices;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// How one block type is drawn by <see cref="BlockRenderPipeline"/>: a texture-atlas tile for each of its six faces, a tint,
    /// and a few flags. One 32-byte entry per block type (indexed by the 12-bit type in <see cref="PackedQuad"/>), uploaded as a
    /// storage buffer. Matches the WGSL <c>BlockMaterial</c> struct byte for byte.
    ///
    /// Faces are numbered as in <see cref="Meshing.VoxelMeshConstants"/>: 0 +X, 1 -X, 2 +Z, 3 -Z, 4 +Y, 5 -Y. Plants (the
    /// crossed-plane "faces" 6 and 7) use face 0's layer.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BlockMaterial
    {
        /// <summary>Tiles of faces 0 and 1 (low and high 16 bits).</summary>
        public uint Layers01;
        /// <summary>Tiles of faces 2 and 3.</summary>
        public uint Layers23;
        /// <summary>Tiles of faces 4 and 5.</summary>
        public uint Layers45;
        /// <summary>A combination of the <c>Flag*</c> constants.</summary>
        public uint Flags;
        /// <summary>Tint colour (rgb), multiplied into the texture when <see cref="FlagTinted"/> is set, and the colour itself when a face has no texture.</summary>
        public float TintR, TintG, TintB;
        /// <summary>Opacity in the water pass (0..1). Ignored by the opaque passes.</summary>
        public float Alpha;

        /// <summary>The face has no texture: it is drawn in the tint colour.</summary>
        public const ushort NoTexture = 0xFFFF;

        /// <summary>The tint multiplies the texture (grass, foliage, water). Without it a textured block draws its texture as is.</summary>
        public const uint FlagTinted = 1;
        /// <summary>The top face sits 0.1 below the block's top edge (a water surface).</summary>
        public const uint FlagLowerTop = 2;
        /// <summary>No per-face brightness steps: side and top faces draw at full brightness, only the bottom face is darkened (water).</summary>
        public const uint FlagFlatShading = 4;
        /// <summary>Every face is drawn at 0.7 of its colour: what lies under water (the same block, darker).</summary>
        public const uint FlagDarken = 8;

        /// <summary>
        /// A plant's crossed planes (quad faces 6 and 7) take their texture and tint from FACE 5 (-Y) instead of face 0: for a block whose cube faces
        /// are something else, such as seagrass, which is a plant standing in a cell of water (faces 0 to 4 the water, face 5 the plant).
        /// </summary>
        public const uint FlagPlantFromBottom = 1u << 8;
        private const int ThicknessShift = 9;

        /// <summary>
        /// A thin block (a carpet, a rail, a pressure plate, a snow layer): the cube is squashed to <paramref name="sixteenths"/> / 16 of its height
        /// from its floor, tops, sides and all, the sides showing the top strip of their texture. 1..15; 0 or 16 = a full block. Used with a class that
        /// does not hide its neighbours (<see cref="Meshing.BlockClass.Cutout"/>), so what is beside and below it still shows. OR the result into <see cref="Flags"/>.
        /// </summary>
        public static uint Thickness(int sixteenths) => sixteenths is <= 0 or >= 16 ? 0u : (uint)sixteenths << ThicknessShift;

        private const int SlopeShift = 13;
        /// <summary>The thin block rises along +X (it is a ramp: its low end at -X, its high end one block up at +X).</summary>
        public const int SlopeEast = 1;
        /// <summary>Rises toward -X.</summary>
        public const int SlopeWest = 2;
        /// <summary>Rises toward +Z.</summary>
        public const int SlopeSouth = 3;
        /// <summary>Rises toward -Z.</summary>
        public const int SlopeNorth = 4;

        /// <summary>
        /// The block is a RAMP: the whole thin block is sheared so its far end is a block higher (a sloping rail). Sections of a ramp that lie side by side merge across the slope,
        /// never along it (the next block of a ramp is a layer higher), so each quad is one block along the slope. 0 = level. OR the result into <see cref="Flags"/>; pair with <see cref="Thickness"/>.
        /// </summary>
        public static uint Slope(int direction) => direction is < 1 or > 4 ? 0u : (uint)direction << SlopeShift;

        /// <summary>
        /// The tint multiplies only the faces named by <paramref name="faceMask"/> (bit f = face f: 0 +X, 1 -X, 2 +Z, 3 -Z, 4 +Y, 5 -Y): the top of a
        /// grass block takes the biome's colour and its sides do not. OR the result into <see cref="Flags"/>. <see cref="FlagTinted"/> tints them all.
        /// </summary>
        public static uint TintFaces(int faceMask) => (uint)(faceMask & 63) << 16;

        private const int ShapeShift = 22;

        /// <summary>
        /// The block is a SHAPE: its quads are faces of the boxes of slot <paramref name="slot"/> of the <see cref="ShapeTable"/> (the vertex shader places each by its box and takes the
        /// tile, window and turn from the table, not from this material's faces). 0..1023. OR the result into <see cref="Flags"/>; the block's class says <see cref="Meshing.BlockClass.Shaped"/>.
        /// </summary>
        public static uint Shape(int slot) => (uint)(slot & 1023) << ShapeShift;

        /// <summary>A material that draws every face with the same texture layer.</summary>
        public static BlockMaterial Uniform(ushort layer, Vector3? tint = null, uint flags = 0, float alpha = 1f)
            => Create(layer, layer, layer, layer, layer, layer, tint, flags, alpha);

        /// <summary>A material with one texture layer per face (<see cref="NoTexture"/> for none).</summary>
        public static BlockMaterial Create(ushort posX, ushort negX, ushort posZ, ushort negZ, ushort posY, ushort negY,
            Vector3? tint = null, uint flags = 0, float alpha = 1f)
        {
            var t = tint ?? Vector3.One;
            return new BlockMaterial
            {
                Layers01 = (uint)posX | ((uint)negX << 16),
                Layers23 = (uint)posZ | ((uint)negZ << 16),
                Layers45 = (uint)posY | ((uint)negY << 16),
                Flags = flags,
                TintR = t.X, TintG = t.Y, TintB = t.Z,
                Alpha = alpha,
            };
        }

        /// <summary>The material of a block type nobody registered: untextured magenta, so a missing entry is visible.</summary>
        public static BlockMaterial Missing => Uniform(NoTexture, new Vector3(1f, 0f, 1f));
    }

    /// <summary>
    /// Everything that changes per frame for <see cref="BlockRenderPipeline"/>: the camera, the sun, ambient light and fog.
    /// 176 bytes, matches the WGSL <c>Frame</c> struct. The caller owns the lighting model (a day/night cycle, a dungeon): the
    /// pipeline only applies the numbers.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BlockFrameUniforms
    {
        /// <summary>View * projection as a raw .NET Matrix4x4 (row-major: WGSL reads it as column-major, which is the transpose it needs). Use <see cref="ReversedZHelper"/>; the pipeline's depth test is reversed-Z.</summary>
        public Matrix4x4 ViewProjection;
        /// <summary>Camera position in world space (xyz); fog distance is measured from it.</summary>
        public Vector4 CameraPosition;
        /// <summary>Unit vector pointing TOWARD the sun (xyz).</summary>
        public Vector4 SunDirection;
        /// <summary>Sun colour (rgb) and, in w, a multiplier on its contribution (night: near 0).</summary>
        public Vector4 SunColor;
        /// <summary>Ambient light colour (rgb) and, in w, the strength of the fixed cool fill light from the -x/-z side (0 = none).</summary>
        public Vector4 Ambient;
        /// <summary>Fog colour (rgb).</summary>
        public Vector4 FogColor;
        /// <summary>Fog starts at x and is complete at y (world units from the camera); the blend is quadratic.</summary>
        public Vector4 FogRange;
        /// <summary>Seconds since some fixed moment (the viewer's start): animated textures play from it (20 ticks a second). Monotonic; precision is fine for days.</summary>
        public float Time;
        /// <summary>Filled in by <see cref="BlockRenderPipeline.BeginFrame"/> from the atlas in use: tiles per row of the atlas, and a tile's side in pixels. Callers leave them 0.</summary>
        public float AtlasTilesPerRow;
        public float AtlasTileSize;
        private float _reserved;

        public const int SizeInBytes = 176;
    }
}
