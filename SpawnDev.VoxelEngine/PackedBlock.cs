namespace SpawnDev.VoxelEngine
{
    /// <summary>
    /// PackedBlock format - encodes block type and damage in a single value.
    ///
    /// Layout (16 bits, stored as int for kernel compatibility, ushort for storage):
    ///   [0:11]  block type (12 bits, 0-4095)
    ///   [12:15] damage level (4 bits, 0-15, where 0=pristine, 15=about to break)
    ///
    /// AubsCraft: damage always 0, uses lower 12 bits for Minecraft block IDs.
    /// Lost Spawns: full damage system (DayZ-style persistent damage).
    ///
    /// Kernels accept ArrayView<int> with PackedBlock encoding.
    /// For sub-word storage optimization, use ArrayView<short> (v4.9.0+).
    /// </summary>
    public static class PackedBlock
    {
        public const int TypeMask = 0xFFF;       // 12 bits
        public const int DamageMask = 0xF;       // 4 bits
        public const int DamageShift = 12;
        public const int MaxBlockTypes = 4096;    // 2^12
        public const int MaxDamageLevel = 15;     // 2^4 - 1

        /// <summary>Pack block type and damage into a single int.</summary>
        public static int Pack(int blockType, int damage = 0)
        {
            return (blockType & TypeMask) | ((damage & DamageMask) << DamageShift);
        }

        /// <summary>Extract block type (0-4095) from packed value.</summary>
        public static int GetType(int packed)
        {
            return packed & TypeMask;
        }

        /// <summary>Extract damage level (0-15) from packed value.</summary>
        public static int GetDamage(int packed)
        {
            return (packed >> DamageShift) & DamageMask;
        }

        /// <summary>Check if packed value represents air (type 0, any damage).</summary>
        public static bool IsAir(int packed)
        {
            return (packed & TypeMask) == 0;
        }

        /// <summary>Pack from short (for sub-word storage). Same layout.</summary>
        public static short PackShort(int blockType, int damage = 0)
        {
            return (short)((blockType & TypeMask) | ((damage & DamageMask) << DamageShift));
        }

        /// <summary>Extract block type from short packed value.</summary>
        public static int GetType(short packed)
        {
            return packed & TypeMask;
        }
    }
}
