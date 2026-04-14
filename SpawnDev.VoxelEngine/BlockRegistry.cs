using System.Runtime.InteropServices;

namespace SpawnDev.VoxelEngine
{
    /// <summary>
    /// Properties for a block type. Uploaded to GPU as a storage buffer
    /// for kernel access (face emission, transparency) and read by game code
    /// for interaction (hardness, sounds, blast resistance).
    ///
    /// Layout must be GPU-friendly: 32 bytes per entry, aligned fields.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BlockProperties
    {
        /// <summary>Seconds to break with bare hands. 0 = instant, -1 = unbreakable.</summary>
        public float Hardness;

        /// <summary>Explosion resistance. Higher = survives larger blasts.</summary>
        public float BlastResistance;

        /// <summary>0=opaque, 1=transparent (glass), 2=translucent (water), 3=plant (cross-quad).</summary>
        public int RenderType;

        /// <summary>Flags: bit 0=gravity, bit 1=liquid, bit 2=flammable, bit 3=structural.</summary>
        public int Flags;

        /// <summary>Sound type index for footsteps/breaking. 0=stone, 1=wood, 2=metal, 3=dirt, 4=sand, 5=glass.</summary>
        public int SoundType;

        /// <summary>Light emission level (0-15). 0=none, 15=max (glowstone/torch).</summary>
        public int LightLevel;

        /// <summary>Padding to 32 bytes.</summary>
        public int Reserved0;
        public int Reserved1;

        // Render type constants
        public const int RenderOpaque = 0;
        public const int RenderTransparent = 1;
        public const int RenderTranslucent = 2;
        public const int RenderPlant = 3;

        // Flag bits
        public const int FlagGravity = 1;
        public const int FlagLiquid = 2;
        public const int FlagFlammable = 4;
        public const int FlagStructural = 8;

        public bool IsOpaque => RenderType == RenderOpaque;
        public bool IsTransparent => RenderType == RenderTransparent;
        public bool IsTranslucent => RenderType == RenderTranslucent;
        public bool IsPlant => RenderType == RenderPlant;
        public bool HasGravity => (Flags & FlagGravity) != 0;
        public bool IsLiquid => (Flags & FlagLiquid) != 0;
        public bool IsFlammable => (Flags & FlagFlammable) != 0;
        public bool IsStructural => (Flags & FlagStructural) != 0;
        public bool EmitsLight => LightLevel > 0;
    }

    /// <summary>
    /// Registry of all block types and their properties.
    /// Uploaded to GPU as a storage buffer for kernel access.
    ///
    /// Usage:
    ///   registry.Register(1, new BlockProperties { Hardness = 1.5f, ... }); // stone
    ///   registry.Register(2, new BlockProperties { RenderType = 1, ... });  // glass
    ///   var gpuBuffer = registry.UploadToGpu(accelerator);
    /// </summary>
    public class BlockRegistry
    {
        private readonly BlockProperties[] _properties;

        /// <summary>Maximum number of block types (matches PackedBlock.MaxBlockTypes).</summary>
        public int MaxTypes => _properties.Length;

        /// <summary>Number of registered block types (non-default entries).</summary>
        public int RegisteredCount { get; private set; }

        /// <summary>
        /// Create a block registry.
        /// Type 0 is always air (default properties, not registered).
        /// </summary>
        public BlockRegistry(int maxTypes = PackedBlock.MaxBlockTypes)
        {
            _properties = new BlockProperties[maxTypes];
            // Air (type 0) has all-zero properties by default
        }

        /// <summary>Register a block type's properties.</summary>
        public void Register(int blockType, BlockProperties props)
        {
            if (blockType <= 0 || blockType >= _properties.Length)
                throw new ArgumentOutOfRangeException(nameof(blockType), $"Block type must be 1-{_properties.Length - 1}");

            if (_properties[blockType].Hardness == 0 && _properties[blockType].RenderType == 0)
                RegisteredCount++; // first registration for this type

            _properties[blockType] = props;
        }

        /// <summary>Get properties for a block type.</summary>
        public BlockProperties Get(int blockType)
        {
            if ((uint)blockType >= (uint)_properties.Length)
                return default;
            return _properties[blockType];
        }

        /// <summary>Check if a block type is solid for collision purposes.</summary>
        public bool IsSolid(int blockType)
        {
            if (blockType == 0) return false; // air
            var props = Get(blockType);
            return props.IsOpaque || props.IsTransparent; // opaque + glass are solid, plants + liquid are not
        }

        /// <summary>Check if a block type emits faces against a neighbor type.</summary>
        public bool EmitsFaceAgainst(int thisType, int neighborType)
        {
            if (thisType == 0) return false; // air never emits
            if (neighborType == 0) return true; // always emit against air

            var thisProps = Get(thisType);
            var neighborProps = Get(neighborType);

            // Opaque against opaque: no face (hidden)
            if (thisProps.IsOpaque && neighborProps.IsOpaque) return false;

            // Transparent/translucent against same type: no face (water-water, glass-glass)
            if (thisType == neighborType && !thisProps.IsOpaque) return false;

            // Everything else: emit face
            return true;
        }

        /// <summary>Get the raw properties array for GPU upload.</summary>
        public BlockProperties[] GetRawArray() => _properties;

        /// <summary>Register common Minecraft-style block types for AubsCraft.</summary>
        public void RegisterDefaults()
        {
            Register(1, new BlockProperties { Hardness = 1.5f, BlastResistance = 6f, SoundType = 0, Flags = BlockProperties.FlagStructural });  // Stone
            Register(2, new BlockProperties { Hardness = 0.6f, BlastResistance = 0.6f, SoundType = 3 }); // Dirt
            Register(3, new BlockProperties { Hardness = 0.6f, BlastResistance = 0.6f, SoundType = 3 }); // Grass
            Register(4, new BlockProperties { Hardness = 0.5f, BlastResistance = 0.5f, SoundType = 4, Flags = BlockProperties.FlagGravity }); // Sand
            Register(5, new BlockProperties { Hardness = 2.0f, BlastResistance = 6f, SoundType = 0, Flags = BlockProperties.FlagStructural }); // Cobblestone
            Register(6, new BlockProperties { Hardness = 2.0f, BlastResistance = 5f, SoundType = 1, Flags = BlockProperties.FlagStructural | BlockProperties.FlagFlammable }); // Oak Wood
            Register(7, new BlockProperties { Hardness = 0.3f, BlastResistance = 0.3f, SoundType = 5, RenderType = BlockProperties.RenderTransparent }); // Glass
            Register(8, new BlockProperties { Hardness = 100f, BlastResistance = 100f, SoundType = 0, RenderType = BlockProperties.RenderTranslucent, Flags = BlockProperties.FlagLiquid }); // Water
            Register(9, new BlockProperties { Hardness = 0f, BlastResistance = 0f, SoundType = 3, RenderType = BlockProperties.RenderPlant }); // Grass (tall)
            Register(10, new BlockProperties { Hardness = 0.5f, BlastResistance = 0.5f, SoundType = 1, Flags = BlockProperties.FlagFlammable }); // Leaves
        }
    }
}
