namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// One face of a <see cref="ShapeBox"/>: the atlas tile it shows, the window of that tile (sixteenths, the box model's own "uv"), how far the tile is turned and whether the block's tint applies.
    /// Faces are numbered as everywhere: 0 +X, 1 -X, 2 +Z, 3 -Z, 4 +Y, 5 -Y. The window's u runs to the right and v DOWN, seen from outside the box (for the top and bottom faces: u along +X,
    /// v along +Z on top, along -Z below, the same as a cube face of <see cref="BlockRenderPipeline"/>).
    /// </summary>
    /// <param name="Tile">The atlas tile, or <see cref="BlockMaterial.NoTexture"/> for a face that is not there.</param>
    /// <param name="U0">Left edge of the window, 0..16.</param>
    /// <param name="Rotation">Quarter turns of the tile, clockwise as seen from outside (0..3).</param>
    public readonly record struct ShapeFace(int Tile, int U0, int V0, int U1, int V1, int Rotation = 0, bool Tinted = false)
    {
        /// <summary>A face that is not drawn.</summary>
        public static ShapeFace None => new(BlockMaterial.NoTexture, 0, 0, 16, 16);

        /// <summary>The whole tile.</summary>
        public static ShapeFace Whole(int tile) => new(tile, 0, 0, 16, 16);
    }

    /// <summary>A box inside a block cell, in sixteenths of a block (0..16 on each axis), with its six faces.</summary>
    public sealed class ShapeBox
    {
        public int X0 { get; }
        public int Y0 { get; }
        public int Z0 { get; }
        public int X1 { get; }
        public int Y1 { get; }
        public int Z1 { get; }
        public ShapeFace[] Faces { get; }

        public ShapeBox(int x0, int y0, int z0, int x1, int y1, int z1, ShapeFace[] faces)
        {
            if (faces.Length != 6) throw new ArgumentException("A box has six faces", nameof(faces));
            foreach (var v in new[] { x0, y0, z0, x1, y1, z1 })
                if (v < 0 || v > 16) throw new ArgumentOutOfRangeException(nameof(x0), $"A box lies within its cell: {v} is outside 0..16");
            if (x1 < x0 || y1 < y0 || z1 < z0) throw new ArgumentException("A box's far corner is not below its near corner");
            X0 = x0; Y0 = y0; Z0 = z0; X1 = x1; Y1 = y1; Z1 = z1;
            Faces = faces;
        }
    }

    /// <summary>
    /// The shapes of the blocks that are not cubes (stairs, slabs, fences, doors, ...): each is a few <see cref="ShapeBox"/>es of one cell. A block type gets a SLOT here
    /// (<see cref="Add"/>), and says so in its class (<see cref="Meshing.BlockClass.Shaped"/> with <see cref="Meshing.BlockClass.ShapeSlot"/>) and its material
    /// (<see cref="BlockMaterial.Shape"/>): the mesher then emits each box's visible faces as quads of that block type, and the vertex shader places them by the box.
    /// Give the same table to <see cref="Meshing.ColumnMesher.SetShapes"/> and <see cref="BlockRenderPipeline.SetShapes"/>.
    /// </summary>
    public sealed class ShapeTable
    {
        /// <summary>Boxes a shape can have.</summary>
        public const int MaxBoxes = 8;
        /// <summary>Slots the table has (the class table's and the material's bit fields hold 10 bits).</summary>
        public const int MaxSlots = 1024;
        /// <summary>32-bit words a box takes in <see cref="Words"/>.</summary>
        public const int WordsPerBox = 16;
        /// <summary>Words a slot takes.</summary>
        public const int WordsPerSlot = MaxBoxes * WordsPerBox;

        private readonly List<ShapeBox[]> _slots = new();

        public int Count => _slots.Count;

        /// <summary>Add a shape; returns its slot (0, 1, 2, ...).</summary>
        public int Add(IReadOnlyList<ShapeBox> boxes)
        {
            if (boxes.Count == 0) throw new ArgumentException("A shape has at least one box", nameof(boxes));
            if (boxes.Count > MaxBoxes) throw new ArgumentException($"A shape has at most {MaxBoxes} boxes, this one has {boxes.Count}", nameof(boxes));
            if (_slots.Count >= MaxSlots) throw new InvalidOperationException($"The table is full: {MaxSlots} shapes");
            _slots.Add(boxes.ToArray());
            return _slots.Count - 1;
        }

        public IReadOnlyList<ShapeBox> Slot(int slot) => _slots[slot];

        /// <summary>
        /// The table as the GPU reads it: for each slot, <see cref="MaxBoxes"/> boxes of <see cref="WordsPerBox"/> words. Word 0 of a box: bit 31 set when the box exists, bits 0-29 its
        /// x0, y0, z0, x1, y1, z1 (5 bits each). Words 1 + 2 f and 2 + 2 f for face f: tile (16 bits) | rotation (2 bits) &lt;&lt; 16 | tinted &lt;&lt; 18; and the window u0 | v0 &lt;&lt; 8 | u1 &lt;&lt; 16 | v1 &lt;&lt; 24.
        /// </summary>
        public uint[] Words()
        {
            var words = new uint[Math.Max(_slots.Count, 1) * WordsPerSlot];
            for (int s = 0; s < _slots.Count; s++)
                for (int b = 0; b < _slots[s].Length; b++)
                    PackBox(_slots[s][b], words, s * WordsPerSlot + b * WordsPerBox);
            return words;
        }

        /// <summary>Pack one box into <paramref name="words"/> at <paramref name="offset"/>.</summary>
        public static void PackBox(ShapeBox box, uint[] words, int offset)
        {
            words[offset] = 1u << 31 | (uint)box.X0 | (uint)box.Y0 << 5 | (uint)box.Z0 << 10 | (uint)box.X1 << 15 | (uint)box.Y1 << 20 | (uint)box.Z1 << 25;
            for (int f = 0; f < 6; f++)
            {
                var face = box.Faces[f];
                words[offset + 1 + 2 * f] = (uint)(face.Tile & 0xFFFF) | (uint)(face.Rotation & 3) << 16 | (face.Tinted ? 1u << 18 : 0u);
                words[offset + 2 + 2 * f] = (uint)(face.U0 & 255) | (uint)(face.V0 & 255) << 8 | (uint)(face.U1 & 255) << 16 | (uint)(face.V1 & 255) << 24;
            }
        }
    }
}
