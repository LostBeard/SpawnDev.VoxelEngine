namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// How a block type takes part in meshing, as the bit flags <see cref="ColumnMesher"/> reads from its per-type table.
    ///
    /// Meshing runs in passes (opaque, water; plants are their own kernel). A pass has an EMIT set (the blocks whose faces it draws)
    /// and an OCCLUDE set (the blocks that hide a neighbour's face). A face is drawn when the block emits and the neighbour does not
    /// occlude. Leaves emit but do not occlude (so the block behind a leaf still shows its faces, and leaf-to-leaf faces are drawn);
    /// water has a pass of its own that only the opaque blocks and other water occlude.
    ///
    /// Use the presets; the raw bits exist so a game can define its own classes.
    /// </summary>
    public static class BlockClass
    {
        /// <summary>The opaque pass draws this block's faces.</summary>
        public const int EmitOpaque = 1 << 0;
        /// <summary>In the opaque pass this block hides the faces of its neighbours.</summary>
        public const int OccludeOpaque = 1 << 1;
        /// <summary>The water pass draws this block's faces.</summary>
        public const int EmitWater = 1 << 2;
        /// <summary>In the water pass this block hides the faces of its neighbours.</summary>
        public const int OccludeWater = 1 << 3;
        /// <summary>The block is a plant: two crossed planes (quad faces 6 and 7), no cube faces.</summary>
        public const int Plant = 1 << 4;

        /// <summary>Air, and anything never drawn.</summary>
        public const int None = 0;

        /// <summary>A solid cube that hides what is behind it (stone, dirt, grass block, planks).</summary>
        public const int Opaque = EmitOpaque | OccludeOpaque | OccludeWater;

        /// <summary>A cube that is drawn but does not hide its neighbours (leaves, glass): the faces behind it stay.</summary>
        public const int Cutout = EmitOpaque | OccludeWater;

        /// <summary>Water: drawn in the water pass, hidden by opaque blocks and other water, hides nothing in the opaque pass.</summary>
        public const int Water = EmitWater | OccludeWater;

        /// <summary>Grass, flowers, ferns: two crossed planes.</summary>
        public const int Cross = Plant;

        /// <summary>Bit position of a pass's emit flag; its occlude flag is the next bit.</summary>
        public const int OpaqueShift = 0;
        public const int WaterShift = 2;
    }
}
