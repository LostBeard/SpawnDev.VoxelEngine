namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Packed quad format - 8 bytes per merged quad.
    /// Encodes position, dimensions, face direction, and block type.
    ///
    /// Layout (64 bits):
    ///   [0:5]   x position within chunk (6 bits, 0-63)
    ///   [6:11]  y position within chunk (6 bits, 0-63)
    ///   [12:17] z position within chunk (6 bits, 0-63)
    ///   [18:23] width of merged quad (6 bits, 1-64)
    ///   [24:29] height of merged quad (6 bits, 1-64)
    ///   [30:32] face direction (3 bits, 0-5)
    ///   [33:47] block type (15 bits, 0-32767)
    ///   [48:63] reserved
    /// </summary>
    public static class PackedQuad
    {
        public static long Pack(int x, int y, int z, int width, int height, int face, int blockType)
        {
            return (long)(x & 0x3F)
                | ((long)(y & 0x3F) << 6)
                | ((long)(z & 0x3F) << 12)
                | ((long)(width & 0x3F) << 18)
                | ((long)(height & 0x3F) << 24)
                | ((long)(face & 0x7) << 30)
                | ((long)(blockType & 0x7FFF) << 33);
        }

        public static void Unpack(long packed, out int x, out int y, out int z,
            out int width, out int height, out int face, out int blockType)
        {
            x = (int)(packed & 0x3F);
            y = (int)((packed >> 6) & 0x3F);
            z = (int)((packed >> 12) & 0x3F);
            width = (int)((packed >> 18) & 0x3F);
            height = (int)((packed >> 24) & 0x3F);
            face = (int)((packed >> 30) & 0x7);
            blockType = (int)((packed >> 33) & 0x7FFF);
        }
    }
}
