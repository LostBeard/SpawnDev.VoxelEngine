using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// GPU kernels that mesh a whole 16 x H x 16 block column - all its 16-high sections in one dispatch per stage - into
    /// <see cref="PackedQuad"/> lists, one list per section. They are the batched, class-aware form of
    /// <see cref="FaceCullKernels"/> + <see cref="GreedyMergeKernels"/> (same bit-column face culling, same greedy merge), driven by
    /// <see cref="ColumnMesher"/>. Dispatching 24 sections one by one would cost 24 x the dispatches; here a column costs a handful.
    ///
    /// Column layout: blocks[x + z * 16 + y * 256], y from the column's bottom, as ushort block types (12 bits used, 0 = air).
    /// Section s holds y in [16 s, 16 s + 16). Occupancy arrays are indexed [s * 324 + (z + 1) * 18 + (x + 1)] - an 18 x 18 grid per
    /// section whose outer ring holds the four neighbouring columns' edge cells (diagonals are never read).
    ///
    /// Every kernel is a flat Index1D with the work decoded inside, so one launch covers every section.
    /// </summary>
    public static class ColumnMeshKernels
    {
        public const int SectionSize = 16;
        public const int PaddedXZ = 18;
        public const int PaddedCellsPerSection = PaddedXZ * PaddedXZ;          // 324
        public const int ColumnsCellsPerLayer = SectionSize * SectionSize;      // 256
        public const int CellsPerSection = SectionSize * SectionSize * SectionSize; // 4096
        /// <summary>
        /// Scratch quads reserved per section. A 16^3 section has 4096 cells and a cell has at most 6 faces, so 24576 quads is the
        /// proven upper bound (a greedy merge never makes more quads than faces): nothing is ever dropped for lack of room.
        /// </summary>
        public const int ScratchQuadsPerSection = CellsPerSection * 6;

        // ------------------------------------------------------------------------------------------------------------
        // Stage 1: occupancy bit-columns, with the neighbouring columns' edges in the ring
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        /// For every (section, x, z) in the padded 18 x 18 grid: bit y of <paramref name="emit"/> / <paramref name="occlude"/> is set when
        /// the block at that height emits / occludes in the pass (<paramref name="passShift"/> selects the bits of the class table).
        /// A missing neighbour column (bit set in <paramref name="missingMask"/>: 1 -X, 2 +X, 4 -Z, 8 +Z) occludes everything when
        /// <paramref name="missingSolid"/> is 1 (no faces toward an unloaded column) and nothing when it is 0.
        /// </summary>
        public static void OccupancyKernel(
            Index1D index,
            ArrayView<ushort> column,
            ArrayView<ushort> negX,
            ArrayView<ushort> posX,
            ArrayView<ushort> negZ,
            ArrayView<ushort> posZ,
            ArrayView<int> typeFlags,
            ArrayView<long> emit,
            ArrayView<long> occlude,
            int passShift,
            int missingMask,
            int missingSolid,
            int sectionCount)
        {
            int total = sectionCount * PaddedCellsPerSection;
            if (index >= total) return;

            int s = index / PaddedCellsPerSection;
            int r = index - s * PaddedCellsPerSection;
            int pz = r / PaddedXZ;
            int px = r - pz * PaddedXZ;
            int x = px - 1;
            int z = pz - 1;

            bool xIn = x >= 0 && x < SectionSize;
            bool zIn = z >= 0 && z < SectionSize;
            long emitBits = 0;
            long occBits = 0;
            long solid = missingSolid != 0 ? 0xFFFFL : 0L;

            if (xIn && zIn)
            {
                emitBits = ColumnBits(column, typeFlags, x, z, s, passShift, 0);
                occBits = ColumnBits(column, typeFlags, x, z, s, passShift, 1);
            }
            else if (!xIn && zIn)
            {
                // The ring only occludes (its faces belong to the neighbour's own mesh): the column to the -X
                // (x == -1: its x = 15) or +X (x == 16: its x = 0)
                if (x < 0)
                    occBits = (missingMask & 1) != 0 ? solid : ColumnBits(negX, typeFlags, SectionSize - 1, z, s, passShift, 1);
                else
                    occBits = (missingMask & 2) != 0 ? solid : ColumnBits(posX, typeFlags, 0, z, s, passShift, 1);
            }
            else if (xIn && !zIn)
            {
                if (z < 0)
                    occBits = (missingMask & 4) != 0 ? solid : ColumnBits(negZ, typeFlags, x, SectionSize - 1, s, passShift, 1);
                else
                    occBits = (missingMask & 8) != 0 ? solid : ColumnBits(posZ, typeFlags, x, 0, s, passShift, 1);
            }
            // corners of the ring (both out) are never read by the face cull

            emit[index] = emitBits;
            occlude[index] = occBits;
        }

        /// <summary>
        /// Bit y of the result: the block at (x, y, z) of section <paramref name="section"/> has its emit (<paramref name="which"/> 0) or
        /// occlude (1) flag for the pass whose flags start at <paramref name="passShift"/>.
        /// </summary>
        private static long ColumnBits(ArrayView<ushort> blocks, ArrayView<int> typeFlags, int x, int z, int section, int passShift, int which)
        {
            int baseIdx = x + z * SectionSize + (section * SectionSize) * ColumnsCellsPerLayer;
            long bits = 0;
            for (int y = 0; y < SectionSize; y++)
            {
                int type = blocks[baseIdx + y * ColumnsCellsPerLayer] & 0xFFF;
                int flag = (typeFlags[type] >> (passShift + which)) & 1;
                bits |= (long)flag << y;
            }
            return bits;
        }

        // ------------------------------------------------------------------------------------------------------------
        // Stage 2: face masks (a face is drawn where the block emits and the neighbour does not occlude)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        /// One thread per (section, x, z). Writes faceMasks[((s * 6) + face) * 256 + z * 16 + x] for the six faces, every entry (a
        /// reused buffer holds the last column's masks): bit y = that cell's face is visible. Faces: 0 +X, 1 -X, 2 +Z, 3 -Z, 4 +Y, 5 -Y.
        /// The cell above the top of a section is the next section's bottom cell; above the column's top and below its bottom is air.
        /// </summary>
        public static void FaceCullKernel(
            Index1D index,
            ArrayView<long> emit,
            ArrayView<long> occlude,
            ArrayView<long> faceMasks,
            int sectionCount)
        {
            int total = sectionCount * ColumnsCellsPerLayer;
            if (index >= total) return;

            int s = index / ColumnsCellsPerLayer;
            int r = index - s * ColumnsCellsPerLayer;
            int z = r / SectionSize;
            int x = r - z * SectionSize;

            int p = s * PaddedCellsPerSection + (z + 1) * PaddedXZ + (x + 1);
            long e = emit[p] & 0xFFFFL;
            long self = occlude[p] & 0xFFFFL;

            long aboveSolid = (self >> 1) | (s < sectionCount - 1 ? (occlude[p + PaddedCellsPerSection] & 1L) << 15 : 0L);
            long belowSolid = ((self << 1) & 0xFFFFL) | (s > 0 ? (occlude[p - PaddedCellsPerSection] >> 15) & 1L : 0L);

            int faceBase = s * 6 * ColumnsCellsPerLayer + r;
            faceMasks[faceBase + 0 * ColumnsCellsPerLayer] = e & ~occlude[p + 1];
            faceMasks[faceBase + 1 * ColumnsCellsPerLayer] = e & ~occlude[p - 1];
            faceMasks[faceBase + 2 * ColumnsCellsPerLayer] = e & ~occlude[p + PaddedXZ];
            faceMasks[faceBase + 3 * ColumnsCellsPerLayer] = e & ~occlude[p - PaddedXZ];
            faceMasks[faceBase + 4 * ColumnsCellsPerLayer] = e & ~aboveSolid & 0xFFFFL;
            faceMasks[faceBase + 5 * ColumnsCellsPerLayer] = e & ~belowSolid & 0xFFFFL;
        }

        // ------------------------------------------------------------------------------------------------------------
        // Stage 3: greedy merge (the logic of GreedyMergeKernels, per section, into per-section slices of a scratch buffer)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        /// One thread per (section, face, layer). <paramref name="faceStart"/> 0 launches faces 0-3 (+X -X +Z -Z) with the layer along
        /// the face normal: sections * 4 * 16 threads; 4 launches faces 4-5 (+Y -Y) with the layer = the Y level: sections * 2 * 16
        /// threads. The two are separate launches because +Y/-Y threads of different layers share mask words.
        /// Quads go to output[s * ScratchQuadsPerSection + slot]; counters[s] counts them, counters[sectionCount] counts overflows.
        /// </summary>
        public static void GreedyMergeKernel(
            Index1D index,
            ArrayView<long> faceMasks,
            ArrayView<ushort> column,
            ArrayView<long> output,
            ArrayView<int> counters,
            int faceStart,
            int sectionCount)
        {
            int faceCount = faceStart == 0 ? 4 : 2;
            int perSection = faceCount * SectionSize;
            if (index >= sectionCount * perSection) return;

            int s = index / perSection;
            int r = index - s * perSection;
            int faceInGroup = r / SectionSize;
            int layer = r - faceInGroup * SectionSize;
            int face = faceStart + faceInGroup;

            if (face >= 4)
                MergeXZPlane(faceMasks, column, output, counters, s, layer, face, sectionCount);
            else
                MergePerpendicularPlane(faceMasks, column, output, counters, s, layer, face, sectionCount);
        }

        private static void EmitQuad(ArrayView<long> output, ArrayView<int> counters, int s, int sectionCount, long quad)
        {
            int slot = Atomic.Add(ref counters[s], 1);
            if (slot >= ScratchQuadsPerSection)
            {
                Atomic.Add(ref counters[s], -1);
                Atomic.Add(ref counters[sectionCount], 1); // overflow: reported, never silent
            }
            else
            {
                output[s * ScratchQuadsPerSection + slot] = quad;
            }
        }

        private static int TypeAt(ArrayView<ushort> column, int x, int z, int y, int s)
            => column[x + z * SectionSize + (s * SectionSize + y) * ColumnsCellsPerLayer] & 0xFFF;

        private static void MergeXZPlane(
            ArrayView<long> faceMasks, ArrayView<ushort> column, ArrayView<long> output, ArrayView<int> counters,
            int s, int yLayer, int face, int sectionCount)
        {
            int faceOffset = (s * 6 + face) * ColumnsCellsPerLayer;

            for (int z = 0; z < SectionSize; z++)
            {
                int x = 0;
                while (x < SectionSize)
                {
                    long mask = faceMasks[faceOffset + x + z * SectionSize];
                    if ((mask & (1L << yLayer)) == 0)
                    {
                        x++;
                        continue;
                    }

                    int blockType = TypeAt(column, x, z, yLayer, s);

                    int w = 1;
                    while (x + w < SectionSize)
                    {
                        long nextMask = faceMasks[faceOffset + (x + w) + z * SectionSize];
                        if ((nextMask & (1L << yLayer)) == 0) break;
                        if (TypeAt(column, x + w, z, yLayer, s) != blockType) break;
                        w++;
                    }

                    int h = 1;
                    bool canExtend = true;
                    while (z + h < SectionSize && canExtend)
                    {
                        for (int dx = 0; dx < w; dx++)
                        {
                            long checkMask = faceMasks[faceOffset + (x + dx) + (z + h) * SectionSize];
                            if ((checkMask & (1L << yLayer)) == 0) { canExtend = false; break; }
                            if (TypeAt(column, x + dx, z + h, yLayer, s) != blockType) { canExtend = false; break; }
                        }
                        if (canExtend) h++;
                    }

                    // Layers of the same section share these mask words (one bit each): clear atomically
                    for (int dz = 0; dz < h; dz++)
                        for (int dx = 0; dx < w; dx++)
                            Atomic.And(ref faceMasks[faceOffset + (x + dx) + (z + dz) * SectionSize], ~(1L << yLayer));

                    EmitQuad(output, counters, s, sectionCount, PackedQuad.Pack(x, yLayer, z, w, h, face, blockType));
                    x += w;
                }
            }
        }

        private static void MergePerpendicularPlane(
            ArrayView<long> faceMasks, ArrayView<ushort> column, ArrayView<long> output, ArrayView<int> counters,
            int s, int layer, int face, int sectionCount)
        {
            int faceOffset = (s * 6 + face) * ColumnsCellsPerLayer;
            bool isXFace = face <= 1;

            for (int outer = 0; outer < SectionSize; outer++)
            {
                int innerX = isXFace ? layer : outer;
                int innerZ = isXFace ? outer : layer;
                int innerIdx = innerX + innerZ * SectionSize;
                long mask = faceMasks[faceOffset + innerIdx];
                if (mask == 0) continue;

                while (mask != 0)
                {
                    int y = TrailingZeros(mask);
                    if (y >= SectionSize) break;

                    int px = isXFace ? layer : outer;
                    int pz = isXFace ? outer : layer;
                    int blockType = TypeAt(column, px, pz, y, s);

                    int h = 1;
                    while (y + h < SectionSize && (mask & (1L << (y + h))) != 0)
                    {
                        if (TypeAt(column, px, pz, y + h, s) != blockType) break;
                        h++;
                    }

                    for (int dy = 0; dy < h; dy++)
                        mask &= ~(1L << (y + dy));
                    faceMasks[faceOffset + innerIdx] = mask;

                    int w = 1;
                    while (outer + w < SectionSize)
                    {
                        int nx = isXFace ? layer : outer + w;
                        int nz = isXFace ? outer + w : layer;
                        int nextInner = nx + nz * SectionSize;
                        long nextMask = faceMasks[faceOffset + nextInner];

                        bool allMatch = true;
                        for (int dy = 0; dy < h; dy++)
                        {
                            if ((nextMask & (1L << (y + dy))) == 0) { allMatch = false; break; }
                            if (TypeAt(column, nx, nz, y + dy, s) != blockType) { allMatch = false; break; }
                        }
                        if (!allMatch) break;

                        for (int dy = 0; dy < h; dy++)
                            nextMask &= ~(1L << (y + dy));
                        faceMasks[faceOffset + nextInner] = nextMask;
                        w++;
                    }

                    int qx = isXFace ? layer : outer;
                    int qz = isXFace ? outer : layer;
                    EmitQuad(output, counters, s, sectionCount, PackedQuad.Pack(qx, y, qz, w, h, face, blockType));
                }
            }
        }

        private static int TrailingZeros(long value)
        {
            if (value == 0) return 64;
            int count = 0;
            long v = value;
            while ((v & 1) == 0)
            {
                count++;
                v >>= 1;
            }
            return count;
        }

        // ------------------------------------------------------------------------------------------------------------
        // Plants: two crossed planes per plant block
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>One thread per cell. A block whose class has <see cref="BlockClass.Plant"/> emits the two diagonal quads (faces 6 and 7).</summary>
        public static void PlantKernel(
            Index1D index,
            ArrayView<ushort> column,
            ArrayView<int> typeFlags,
            ArrayView<long> output,
            ArrayView<int> counters,
            int sectionCount)
        {
            int total = sectionCount * CellsPerSection;
            if (index >= total) return;

            int s = index / CellsPerSection;
            int r = index - s * CellsPerSection;
            int y = r / ColumnsCellsPerLayer;
            int rem = r - y * ColumnsCellsPerLayer;
            int z = rem / SectionSize;
            int x = rem - z * SectionSize;

            int type = TypeAt(column, x, z, y, s);
            if (type == 0) return;
            if ((typeFlags[type] & BlockClass.Plant) == 0) return;

            EmitQuad(output, counters, s, sectionCount, PackedQuad.Pack(x, y, z, 1, 1, 6, type));
            EmitQuad(output, counters, s, sectionCount, PackedQuad.Pack(x, y, z, 1, 1, 7, type));
        }

        // ------------------------------------------------------------------------------------------------------------
        // Shapes: the boxes of the blocks that are not cubes (stairs, slabs, fences, doors...)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        /// One thread per cell. A block whose class has <see cref="BlockClass.Shaped"/> emits, for every box of its slot in the shape table and every face of the box that has a
        /// texture, a one-by-one quad flagged as a box face (<see cref="PackedQuad.PackShapeFace"/>). A face lying on the cell boundary is left out when the neighbour there is a block that
        /// occludes (<paramref name="occlude"/> is the opaque pass occupancy, the ring of neighbouring columns included): nothing of it could be seen. Faces inside the cell stay, even where
        /// two boxes touch.
        /// </summary>
        public static void ShapeKernel(
            Index1D index,
            ArrayView<ushort> column,
            ArrayView<int> typeFlags,
            ArrayView<uint> shapeWords,
            ArrayView<long> occlude,
            ArrayView<long> output,
            ArrayView<int> counters,
            int sectionCount)
        {
            int total = sectionCount * CellsPerSection;
            if (index >= total) return;

            int s = index / CellsPerSection;
            int r = index - s * CellsPerSection;
            int y = r / ColumnsCellsPerLayer;
            int rem = r - y * ColumnsCellsPerLayer;
            int z = rem / SectionSize;
            int x = rem - z * SectionSize;

            int type = TypeAt(column, x, z, y, s);
            if (type == 0) return;
            int flags = typeFlags[type];
            if ((flags & BlockClass.Shaped) == 0) return;
            int slot = (flags >> BlockClass.ShapeSlotShift) & BlockClass.ShapeSlotMask;

            // What the neighbours of this cell do: the bits of the occupancy columns, as the face cull reads them
            int p = s * PaddedCellsPerSection + (z + 1) * PaddedXZ + (x + 1);
            long self = occlude[p] & 0xFFFFL;
            long posX = occlude[p + 1], negX = occlude[p - 1], posZ = occlude[p + PaddedXZ], negZ = occlude[p - PaddedXZ];
            long above = (self >> 1) | (s < sectionCount - 1 ? (occlude[p + PaddedCellsPerSection] & 1L) << 15 : 0L);
            long below = ((self << 1) & 0xFFFFL) | (s > 0 ? (occlude[p - PaddedCellsPerSection] >> 15) & 1L : 0L);
            long bit = 1L << y;
            bool hidePosX = (posX & bit) != 0, hideNegX = (negX & bit) != 0, hidePosZ = (posZ & bit) != 0, hideNegZ = (negZ & bit) != 0;
            bool hidePosY = (above & bit) != 0, hideNegY = (below & bit) != 0;

            int slotBase = slot * (8 * 16);
            for (int b = 0; b < 8; b++)
            {
                int boxBase = slotBase + b * 16;
                uint w0 = shapeWords[boxBase];
                if ((w0 >> 31) == 0) break;   // the boxes of a shape are the first ones of its slot
                int x0 = (int)(w0 & 31u), y0 = (int)((w0 >> 5) & 31u), z0 = (int)((w0 >> 10) & 31u);
                int x1 = (int)((w0 >> 15) & 31u), y1 = (int)((w0 >> 20) & 31u), z1 = (int)((w0 >> 25) & 31u);

                for (int face = 0; face < 6; face++)
                {
                    if ((shapeWords[boxBase + 1 + 2 * face] & 0xFFFFu) == 0xFFFFu) continue;   // no texture: this face is not there
                    bool hidden;
                    if (face == 0) hidden = x1 == 16 && hidePosX;
                    else if (face == 1) hidden = x0 == 0 && hideNegX;
                    else if (face == 2) hidden = z1 == 16 && hidePosZ;
                    else if (face == 3) hidden = z0 == 0 && hideNegZ;
                    else if (face == 4) hidden = y1 == 16 && hidePosY;
                    else hidden = y0 == 0 && hideNegY;
                    if (hidden) continue;
                    EmitQuad(output, counters, s, sectionCount, PackedQuad.PackShapeFace(x, y, z, face, type, b));
                }
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // Compaction: per-section scratch slices -> one exact-size buffer
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        /// One thread per (section, slot). Copies scratch[s * ScratchQuadsPerSection + i] to output[sectionStart[s] + i] for i below
        /// the section's count; sectionStart is the running sum of the counts (the host computes it from the counter readback).
        /// </summary>
        public static void CompactKernel(
            Index1D index,
            ArrayView<long> scratch,
            ArrayView<int> counters,
            ArrayView<int> sectionStart,
            ArrayView<long> output,
            int sectionCount,
            int embedSectionIndex)
        {
            int total = sectionCount * ScratchQuadsPerSection;
            if (index >= total) return;

            int s = index / ScratchQuadsPerSection;
            int i = index - s * ScratchQuadsPerSection;
            if (i >= counters[s]) return;
            long quad = scratch[index];
            if (embedSectionIndex != 0) quad = PackedQuad.WithSectionIndex(quad, s);
            output[sectionStart[s] + i] = quad;
        }
    }
}
