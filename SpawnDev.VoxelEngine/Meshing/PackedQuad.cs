namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Packed quad format - 8 bytes per merged quad.
    /// Encodes section-local position, dimensions, face direction, and block info.
    ///
    /// Section-local coordinates: x,z = 0-15, y = 0-15 (16x16x16 sections).
    /// Quad dimensions: width/height 1-16 (stored as 0-15, add 1 on unpack).
    ///
    /// Layout (64 bits):
    ///   [0:3]   x position within section (4 bits, 0-15)
    ///   [4:7]   y position within section (4 bits, 0-15)
    ///   [8:11]  z position within section (4 bits, 0-15)
    ///   [12:15] width of merged quad - 1 (4 bits, 0-15 = width 1-16)
    ///   [16:19] height of merged quad - 1 (4 bits, 0-15 = height 1-16)
    ///   [20:22] face direction (3 bits, 0-5)
    ///   [23:34] block type (12 bits, 0-4095, matches PackedBlock.TypeMask)
    ///   [35:38] damage level (4 bits, 0-15, matches PackedBlock.DamageMask)
    ///   [39:46] per-vertex AO (8 bits: 4 corners x 2 bits each, 0-3)
    ///   [47:51] section index (5 bits, 0-31): which 16-high section of its column the quad sits in. 0 unless a mesher was told to embed it
    ///           (<c>EmbedSectionIndex</c>); the renderer adds sectionIndex * 16 to the quad's Y, so ONE draw can cover a whole column
    ///           instead of one per section
    ///   [52:63] reserved (12 bits)
    /// </summary>
    public static class PackedQuad
    {
        /// <summary>Bit position of the embedded section index.</summary>
        public const int SectionIndexShift = 47;
        public const int SectionIndexMask = 0x1F;

        public static long Pack(int x, int y, int z, int width, int height, int face, int blockType, int damage = 0, int ao = 0, int sectionIndex = 0)
        {
            return (long)(x & 0xF)
                | ((long)(y & 0xF) << 4)
                | ((long)(z & 0xF) << 8)
                | ((long)((width - 1) & 0xF) << 12)  // store width-1
                | ((long)((height - 1) & 0xF) << 16) // store height-1
                | ((long)(face & 0x7) << 20)
                | ((long)(blockType & 0xFFF) << 23)
                | ((long)(damage & 0xF) << 35)
                | ((long)(ao & 0xFF) << 39)
                | ((long)(sectionIndex & SectionIndexMask) << SectionIndexShift);
        }

        /// <summary>The section index embedded in a quad (0 when none was).</summary>
        public static int UnpackSectionIndex(long packed) => (int)((packed >> SectionIndexShift) & SectionIndexMask);

        /// <summary>The same quad with its section index set (replacing any embedded one).</summary>
        public static long WithSectionIndex(long packed, int sectionIndex)
            => (packed & ~((long)SectionIndexMask << SectionIndexShift)) | ((long)(sectionIndex & SectionIndexMask) << SectionIndexShift);

        public static void Unpack(long packed, out int x, out int y, out int z,
            out int width, out int height, out int face, out int blockType)
        {
            x = (int)(packed & 0xF);
            y = (int)((packed >> 4) & 0xF);
            z = (int)((packed >> 8) & 0xF);
            width = (int)((packed >> 12) & 0xF) + 1;  // restore +1
            height = (int)((packed >> 16) & 0xF) + 1;  // restore +1
            face = (int)((packed >> 20) & 0x7);
            blockType = (int)((packed >> 23) & 0xFFF);
        }

        public static void UnpackFull(long packed, out int x, out int y, out int z,
            out int width, out int height, out int face, out int blockType,
            out int damage, out int ao)
        {
            Unpack(packed, out x, out y, out z, out width, out height, out face, out blockType);
            damage = (int)((packed >> 35) & 0xF);
            ao = (int)((packed >> 39) & 0xFF);
        }
    }
}
