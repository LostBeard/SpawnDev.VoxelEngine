using ILGPU;

namespace SpawnDev.VoxelEngine.Lighting
{
    /// <summary>
    /// The kernels of <see cref="LightPropagator"/>. A column is lit inside a region of 3 x 3 columns (the column in the middle, its four neighbours on the sides, the corners not
    /// loaded: <see cref="Region"/> cells across), because light travels at most 15 cells and nothing farther can reach the middle column. A cell of the region is an int:
    /// bits 0-3 block light, bits 4-7 sky light, bit 8 set when light passes through the block, bits 9-12 how much the block dims light that passes through it (see <see cref="TableDimsShift"/>).
    /// </summary>
    internal static class LightKernels
    {
        /// <summary>Cells across the region: three columns.</summary>
        public const int Region = 48;
        /// <summary>Cells from the region's edge to the middle column.</summary>
        public const int Margin = 16;
        /// <summary>How far light travels: a cell 15 away from a source of 15 is 0.</summary>
        public const int Reach = 15;
        public const int PassesBit = 0x100;
        public const int OpenSky = PassesBit | 0xF0;

        /// <summary>The table's bit for a block that light passes through; bits 4-7 are the light the block gives off.</summary>
        public const int TablePasses = 1;
        /// <summary>The table's bits 8-11: levels of light a block takes off whatever passes through it, over the one level every step costs (leaves: 1; sky light falling straight down loses exactly this).</summary>
        public const int TableDimsShift = 8;
        /// <summary>Where a cell keeps its block's dimming.</summary>
        public const int CellDimsShift = 9;

        private static int TypeAt(ArrayView<ushort> column, ArrayView<ushort> negX, ArrayView<ushort> posX, ArrayView<ushort> negZ, ArrayView<ushort> posZ, int source, int at)
        {
            if (source == 0) return column[at] & 0xFFF;
            if (source == 1) return negX[at] & 0xFFF;
            if (source == 2) return posX[at] & 0xFFF;
            if (source == 3) return negZ[at] & 0xFFF;
            return posZ[at] & 0xFFF;
        }

        /// <summary>One thread for each (x, z) of the region: the column of cells from the top down, with the sky light 15 where nothing is above and each block's own light.</summary>
        public static void InitKernel(
            Index1D index, ArrayView<ushort> column, ArrayView<ushort> negX, ArrayView<ushort> posX, ArrayView<ushort> negZ, ArrayView<ushort> posZ,
            ArrayView<int> table, ArrayView<int> cells, int height, int present)
        {
            if (index >= Region * Region) return;
            int rz = index / Region;
            int rx = index - rz * Region;
            int x = rx - Margin, z = rz - Margin;
            int source = -1, sx = 0, sz = 0;
            bool inX = x >= 0 && x < 16, inZ = z >= 0 && z < 16;
            if (inX && inZ) { source = 0; sx = x; sz = z; }
            else if (x < 0 && inZ && (present & 1) != 0) { source = 1; sx = x + 16; sz = z; }
            else if (x >= 16 && inZ && (present & 2) != 0) { source = 2; sx = x - 16; sz = z; }
            else if (z < 0 && inX && (present & 4) != 0) { source = 3; sx = x; sz = z + 16; }
            else if (z >= 16 && inX && (present & 8) != 0) { source = 4; sx = x; sz = z - 16; }

            bool blocked = false, dimmed = false;
            for (int y = height - 1; y >= 0; y--)
            {
                int cell = OpenSky;   // not loaded: lit as open sky, as a column without a slot is in the LightVolume
                if (source >= 0)
                {
                    int info = table[TypeAt(column, negX, posX, negZ, posZ, source, sx + sz * 16 + y * 256)];
                    bool passes = (info & TablePasses) != 0;
                    int dims = (info >> TableDimsShift) & 15;
                    if (!passes) blocked = true;
                    // Sky light 15 falls straight down through open blocks; the first block that dims it leaves what it lets through (the cells below are lit by the steps from there)
                    int sky = 0;
                    if (!blocked && !dimmed)
                    {
                        sky = 15 - dims;
                        if (dims > 0) dimmed = true;
                    }
                    cell = (passes ? PassesBit : 0) | (dims << CellDimsShift) | (sky << 4) | ((info >> 4) & 15);
                }
                cells[(y * Region + rz) * Region + rx] = cell;
            }
        }

        // What a neighbour gives a cell that dims light by dims: one level less of each light (or the dimming, if more); sky light 15 coming from above loses only the dimming
        private static int Spread(int neighbour, bool fromAbove, int dims)
        {
            int sky = (neighbour >> 4) & 15, block = neighbour & 15;
            int cost = dims > 1 ? dims : 1;
            int s = fromAbove && sky == 15 ? 15 - dims : sky - cost;
            int b = block - cost;
            if (s < 0) s = 0;
            if (b < 0) b = 0;
            return (s << 4) | b;
        }

        private static int MaxLevels(int a, int b)
        {
            int sky = Math.Max((a >> 4) & 15, (b >> 4) & 15);
            int block = Math.Max(a & 15, b & 15);
            return (sky << 4) | block;
        }

        /// <summary>One step of the light: every cell that light passes takes what its six neighbours give. Light only ever grows, so after <see cref="Reach"/> steps nothing changes.</summary>
        public static void RelaxKernel(Index1D index, ArrayView<int> src, ArrayView<int> dst, int height)
        {
            int layer = Region * Region;
            if (index >= layer * height) return;
            int v = src[index];
            if ((v & PassesBit) == 0) { dst[index] = v; return; }
            int dims = (v >> CellDimsShift) & 15;
            int y = index / layer;
            int r = index - y * layer;
            int rz = r / Region;
            int rx = r - rz * Region;
            int best = v & 0xFF;
            if (rx > 0) best = MaxLevels(best, Spread(src[index - 1], false, dims));
            if (rx < Region - 1) best = MaxLevels(best, Spread(src[index + 1], false, dims));
            if (rz > 0) best = MaxLevels(best, Spread(src[index - Region], false, dims));
            if (rz < Region - 1) best = MaxLevels(best, Spread(src[index + Region], false, dims));
            if (y > 0) best = MaxLevels(best, Spread(src[index - layer], false, dims));
            if (y < height - 1) best = MaxLevels(best, Spread(src[index + layer], true, dims));
            dst[index] = (v & ~0xFF) | best;
        }

        /// <summary>One thread for each emitter: a cell of the region (index) and a level, made at least that bright in block light before the light spreads. The cells are all different (the caller keeps the brightest of two in one cell).</summary>
        public static void EmitKernel(Index1D index, ArrayView<int> emitters, ArrayView<int> cells)
        {
            int cell = emitters[2 * index];
            int level = emitters[2 * index + 1];
            int v = cells[cell];
            if (level > (v & 15)) cells[cell] = (v & ~15) | level;
        }

        /// <summary>The middle column's light as bytes (sky in the high four bits, block light in the low four), four to a word, in the order of the block ids: x + z * 16 + y * 256.</summary>
        public static void PackKernel(Index1D index, ArrayView<int> cells, ArrayView<uint> output, int height)
        {
            if (index >= height * 64) return;
            int first = index * 4;
            int x0 = first & 15, z = (first >> 4) & 15, y = first >> 8;
            uint word = 0;
            for (int i = 0; i < 4; i++)
            {
                int v = cells[(y * Region + (z + Margin)) * Region + (x0 + i + Margin)];
                word |= (uint)(v & 0xFF) << (8 * i);
            }
            output[index] = word;
        }

        /// <summary>
        /// One thread for each (side, y): a fingerprint of the 15 cells deep and 16 along the side of the middle column, of its blocks and of the light just worked out. The side's fingerprints from
        /// the last time are what tells whether the neighbour on that side has to be lit again.
        /// </summary>
        public static void SignatureKernel(Index1D index, ArrayView<ushort> column, ArrayView<int> cells, ArrayView<int> signatures, int height)
        {
            if (index >= 4 * height) return;
            int side = index / height;
            int y = index - side * height;
            int h = 17;
            for (int d = 0; d < Reach; d++)
                for (int k = 0; k < 16; k++)
                {
                    int x, z;
                    if (side == 0) { x = d; z = k; }
                    else if (side == 1) { x = 15 - d; z = k; }
                    else if (side == 2) { x = k; z = d; }
                    else { x = k; z = 15 - d; }
                    int light = cells[(y * Region + (z + Margin)) * Region + (x + Margin)] & 0xFF;
                    int type = column[x + z * 16 + y * 256] & 0xFFF;
                    h = (h * 16777619) ^ (light | (type << 8));
                }
            signatures[index] = h;
        }
    }
}
