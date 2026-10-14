using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// The layout of one chunk's heightmap in a flat int buffer (<see cref="Ints"/> ints = 3072 bytes, little-endian, so a fetched
    /// byte array can be copied in as it arrives):
    /// <list type="bullet">
    /// <item>[0, 256): the top block's world Y per column (cell = x + z * 16)</item>
    /// <item>[256, 384): the top block's type per column, two ushorts per int (cell c: int 256 + c/2, low half for even c)</item>
    /// <item>[384, 640): for a column topped by water, the world Y of the first solid block below it (otherwise -64)</item>
    /// <item>[640, 768): that block's type, two ushorts per int; 0 = none</item>
    /// </list>
    /// World Y is Minecraft's (-64 .. 319 here: the column grid is 16 x 384 x 16 with its bottom at <see cref="WorldMinY"/>).
    /// </summary>
    public static class HeightmapLayout
    {
        public const int Cells = 256;
        public const int Ints = 768;
        public const int Bytes = Ints * 4;
        public const int HeightsOffset = 0;
        public const int TypesOffset = 256;
        public const int SeabedHeightsOffset = 384;
        public const int SeabedTypesOffset = 640;
        public const int WorldMinY = -64;
    }

    /// <summary>
    /// GPU kernel that turns a chunk's heightmap into <see cref="PackedQuad"/> lists, one list per 16-high section (the sections the
    /// quads sit in, so a draw can place each by section origin exactly like a column's). The far tier of a world viewer: no
    /// caves, no interior, one top face per column plus the walls where a column stands higher than its neighbour.
    ///
    /// What is drawn, per column with a top block of class C:
    /// <list type="bullet">
    /// <item>solid (opaque, leaves, glass): its top face and walls down to the neighbouring column's solid top</item>
    /// <item>water: a water-pass top face; then the seabed below it (the first solid block) as a solid column of its own, with the
    /// type + <c>seabedTypeOffset</c> so a viewer can darken what lies under water</item>
    /// <item>plants and unknown types: nothing</item>
    /// </list>
    /// A column's "solid top" is its top block's Y, or its seabed's Y under water. A wall spans the difference to the neighbour's solid
    /// top, so a shore wall is complete down to the seabed. A neighbour that is not loaded counts as bottomless (the wall goes to the
    /// world floor); a neighbour column with no blocks too.
    /// </summary>
    public static class HeightmapMeshKernels
    {
        /// <summary>Scratch quads per section. Bound: a column adds at most a top and four walls to a section, so 5 x 256.</summary>
        public const int ScratchQuadsPerSection = 5 * HeightmapLayout.Cells;

        /// <summary>The solid top of a missing or empty column: below the lowest block, so a wall reaches the world floor.</summary>
        private const int Floor = HeightmapLayout.WorldMinY - 1;

        /// <summary>
        /// One thread per column cell (256). hood = 5 heightmaps back to back (slot 0 this chunk, 1 -X, 2 +X, 3 -Z, 4 +Z; a slot whose
        /// bit (slot - 1) is set in <paramref name="missing"/> is not read). Opaque quads go to scratchOpaque[s * ScratchQuadsPerSection + slot]
        /// with counters[s], water quads to scratchWater with counters[sections + 1 + s]; counters[sections] and counters[2 * sections + 1]
        /// count quads that did not fit (never, by the bound above).
        /// </summary>
        public static void HeightmapKernel(
            Index1D index,
            ArrayView<int> hood,
            ArrayView<int> typeFlags,
            ArrayView<long> scratchOpaque,
            ArrayView<long> scratchWater,
            ArrayView<int> counters,
            int missing,
            int seabedTypeOffset,
            int sections)
        {
            if (index >= HeightmapLayout.Cells) return;
            int cell = index;
            int x = cell & 15;
            int z = cell >> 4;

            int t = TypeOf(hood, 0, HeightmapLayout.TypesOffset, cell);
            if (t == 0) return;
            int flags = typeFlags[t];
            if (flags == 0 || (flags & BlockClass.Plant) != 0) return;

            if (IsWater(flags))
            {
                EmitQuad(scratchWater, counters, sections + 1, sections, hood[HeightmapLayout.HeightsOffset + cell], x, z, 1, 1, 4, t);

                int st = TypeOf(hood, 0, HeightmapLayout.SeabedTypesOffset, cell);
                if (st == 0) return;
                int sflags = typeFlags[st];
                if (sflags == 0 || (sflags & BlockClass.Plant) != 0) return;
                EmitSolidColumn(hood, typeFlags, scratchOpaque, counters, x, z, hood[HeightmapLayout.SeabedHeightsOffset + cell], st + seabedTypeOffset,
                    missing, sections);
            }
            else
            {
                EmitSolidColumn(hood, typeFlags, scratchOpaque, counters, x, z, hood[HeightmapLayout.HeightsOffset + cell], t, missing, sections);
            }
        }

        /// <summary>One thread per (section, slot): copies scratch[s * ScratchQuadsPerSection + i] to output[sectionStart[s] + i] for i below the section's count.</summary>
        public static void CompactKernel(
            Index1D index,
            ArrayView<long> scratch,
            ArrayView<int> counters,
            ArrayView<int> sectionStart,
            ArrayView<long> output,
            int sectionCount)
        {
            int total = sectionCount * ScratchQuadsPerSection;
            if (index >= total) return;
            int s = index / ScratchQuadsPerSection;
            int i = index - s * ScratchQuadsPerSection;
            if (i >= counters[s]) return;
            output[sectionStart[s] + i] = scratch[index];
        }

        private static bool IsWater(int flags) => (flags & BlockClass.EmitWater) != 0 && (flags & BlockClass.EmitOpaque) == 0;

        private static int TypeOf(ArrayView<int> hood, int slotBase, int typesOffset, int cell)
            => (hood[slotBase + typesOffset + (cell >> 1)] >> ((cell & 1) * 16)) & 0xFFF;

        /// <summary>The Y of the highest solid block of a column (its top, or its seabed under water), or <see cref="Floor"/> when it has none.</summary>
        private static int SolidTop(ArrayView<int> hood, ArrayView<int> typeFlags, int slotBase, int cell)
        {
            int t = TypeOf(hood, slotBase, HeightmapLayout.TypesOffset, cell);
            if (t == 0) return Floor;
            int flags = typeFlags[t];
            if (flags == 0 || (flags & BlockClass.Plant) != 0) return Floor;
            if (IsWater(flags))
            {
                int st = TypeOf(hood, slotBase, HeightmapLayout.SeabedTypesOffset, cell);
                if (st == 0) return Floor;
                int sflags = typeFlags[st];
                if (sflags == 0 || (sflags & BlockClass.Plant) != 0) return Floor;
                return hood[slotBase + HeightmapLayout.SeabedHeightsOffset + cell];
            }
            return hood[slotBase + HeightmapLayout.HeightsOffset + cell];
        }

        private static int NeighbourSolidTop(ArrayView<int> hood, ArrayView<int> typeFlags, int x, int z, int dx, int dz, int missing)
        {
            int nx = x + dx, nz = z + dz;
            if (nx >= 0 && nx < 16 && nz >= 0 && nz < 16)
                return SolidTop(hood, typeFlags, 0, nx + nz * 16);
            int slot = dx < 0 ? 1 : (dx > 0 ? 2 : (dz < 0 ? 3 : 4));
            if (((missing >> (slot - 1)) & 1) != 0) return Floor;
            return SolidTop(hood, typeFlags, slot * HeightmapLayout.Ints, (nx & 15) + (nz & 15) * 16);
        }

        private static void EmitSolidColumn(
            ArrayView<int> hood, ArrayView<int> typeFlags, ArrayView<long> scratch, ArrayView<int> counters,
            int x, int z, int top, int type, int missing, int sections)
        {
            EmitQuad(scratch, counters, 0, sections, top, x, z, 1, 1, 4, type);
            // +X, -X, +Z, -Z: the wall faces where this column stands higher than the neighbour
            EmitWall(scratch, counters, sections, x, z, top, type, 0, NeighbourSolidTop(hood, typeFlags, x, z, 1, 0, missing));
            EmitWall(scratch, counters, sections, x, z, top, type, 1, NeighbourSolidTop(hood, typeFlags, x, z, -1, 0, missing));
            EmitWall(scratch, counters, sections, x, z, top, type, 2, NeighbourSolidTop(hood, typeFlags, x, z, 0, 1, missing));
            EmitWall(scratch, counters, sections, x, z, top, type, 3, NeighbourSolidTop(hood, typeFlags, x, z, 0, -1, missing));
        }

        // The wall's blocks are the ones above the neighbour's solid top up to this column's top, split at section borders.
        private static void EmitWall(ArrayView<long> scratch, ArrayView<int> counters, int sections, int x, int z, int top, int type, int face, int neighbourTop)
        {
            if (top <= neighbourTop) return;
            int a = neighbourTop + 1 - HeightmapLayout.WorldMinY;
            int b = top - HeightmapLayout.WorldMinY;
            int last = sections * 16 - 1;
            if (a < 0) a = 0;
            if (b > last) b = last;
            while (a <= b)
            {
                int s = a >> 4;
                int sectionTop = s * 16 + 15;
                int end = b < sectionTop ? b : sectionTop;
                EmitQuadAt(scratch, counters, 0, sections, s, a & 15, x, z, 1, end - a + 1, face, type);
                a = end + 1;
            }
        }

        // A quad on the block at world Y (the section and local Y follow from it)
        private static void EmitQuad(ArrayView<long> scratch, ArrayView<int> counters, int counterOffset, int sections,
            int worldY, int x, int z, int w, int h, int face, int type)
        {
            int a = worldY - HeightmapLayout.WorldMinY;
            if (a < 0 || a >= sections * 16) return;
            EmitQuadAt(scratch, counters, counterOffset, sections, a >> 4, a & 15, x, z, w, h, face, type);
        }

        private static void EmitQuadAt(ArrayView<long> scratch, ArrayView<int> counters, int counterOffset, int sections,
            int s, int localY, int x, int z, int w, int h, int face, int type)
        {
            int slot = Atomic.Add(ref counters[counterOffset + s], 1);
            if (slot >= ScratchQuadsPerSection)
            {
                Atomic.Add(ref counters[counterOffset + s], -1);
                Atomic.Add(ref counters[counterOffset + sections], 1); // overflow: reported, never silent
            }
            else
            {
                scratch[s * ScratchQuadsPerSection + slot] = PackedQuad.Pack(x, localY, z, w, h, face, type);
            }
        }
    }
}
