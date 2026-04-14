namespace SpawnDev.VoxelEngine.Terrain
{
    /// <summary>
    /// Assigns biomes based on elevation, slope, and depth below surface.
    ///
    /// Biome rules are configurable per game:
    /// - Lost Spawns: DayZ-style realistic (beach, forest, cliff, rocky summit)
    /// - AubsCraft: Minecraft-style (plains, mountains, desert)
    ///
    /// The biome determines which block types to place at each depth layer.
    /// </summary>
    public class BiomeAssigner
    {
        /// <summary>
        /// Biome classification result.
        /// </summary>
        public enum Biome
        {
            Beach,
            Forest,
            Cliff,
            RockySummit,
            SparseVegetation,
            Desert,
            Underwater,
        }

        /// <summary>
        /// Rule for classifying a biome based on terrain properties.
        /// Rules are evaluated in order - first match wins.
        /// </summary>
        public class BiomeRule
        {
            public Biome Biome { get; init; }
            public float MinElevation { get; init; } = float.MinValue;
            public float MaxElevation { get; init; } = float.MaxValue;
            public float MinSlope { get; init; } = 0;
            public float MaxSlope { get; init; } = float.MaxValue;

            public bool Matches(float elevation, float slope)
            {
                return elevation >= MinElevation && elevation <= MaxElevation &&
                       slope >= MinSlope && slope <= MaxSlope;
            }
        }

        /// <summary>
        /// Block assignment for a depth layer within a biome.
        /// </summary>
        public class DepthLayer
        {
            /// <summary>Minimum depth below surface (0 = surface).</summary>
            public int MinDepth { get; init; }
            /// <summary>Maximum depth below surface.</summary>
            public int MaxDepth { get; init; } = int.MaxValue;
            /// <summary>Block type to place at this depth.</summary>
            public int BlockType { get; init; }
        }

        private readonly List<BiomeRule> _rules = new();
        private readonly Dictionary<Biome, List<DepthLayer>> _depthLayers = new();
        private Biome _defaultBiome = Biome.Forest;
        private int _defaultBlockType = 1; // stone

        /// <summary>Sea level elevation. Below this = underwater biome.</summary>
        public float SeaLevel { get; set; } = 0f;

        /// <summary>Add a biome classification rule. Rules are evaluated in order.</summary>
        public BiomeAssigner AddRule(BiomeRule rule)
        {
            _rules.Add(rule);
            return this;
        }

        /// <summary>Set block types for a biome's depth layers.</summary>
        public BiomeAssigner SetDepthLayers(Biome biome, params DepthLayer[] layers)
        {
            _depthLayers[biome] = new List<DepthLayer>(layers);
            return this;
        }

        /// <summary>Set the default biome and block type for unmatched terrain.</summary>
        public BiomeAssigner SetDefault(Biome biome, int blockType)
        {
            _defaultBiome = biome;
            _defaultBlockType = blockType;
            return this;
        }

        /// <summary>
        /// Classify terrain at a given position.
        /// </summary>
        public Biome Classify(float elevation, float slope)
        {
            if (elevation < SeaLevel) return Biome.Underwater;

            foreach (var rule in _rules)
            {
                if (rule.Matches(elevation, slope))
                    return rule.Biome;
            }

            return _defaultBiome;
        }

        /// <summary>
        /// Get the block type for a position given its biome and depth below surface.
        /// </summary>
        public int GetBlockType(Biome biome, int depthBelowSurface)
        {
            if (_depthLayers.TryGetValue(biome, out var layers))
            {
                foreach (var layer in layers)
                {
                    if (depthBelowSurface >= layer.MinDepth && depthBelowSurface <= layer.MaxDepth)
                        return layer.BlockType;
                }
            }

            return _defaultBlockType;
        }

        /// <summary>
        /// Create a block assignment function suitable for HeightmapTerrainProvider.GenerateSection.
        /// </summary>
        public Func<float, float, float, float, float, int, int> CreateAssigner()
        {
            return (worldX, worldY, worldZ, surfaceHeight, slope, depth) =>
            {
                float elevation = surfaceHeight - SeaLevel;
                var biome = Classify(elevation, slope);
                return GetBlockType(biome, depth);
            };
        }

        /// <summary>
        /// Create a Lost Spawns (DayZ-style) biome configuration.
        /// Based on Deer Isle terrain characteristics.
        /// </summary>
        public static BiomeAssigner CreateLostSpawns()
        {
            var assigner = new BiomeAssigner { SeaLevel = 0f };

            assigner
                .AddRule(new BiomeRule { Biome = Biome.Beach, MinElevation = 0, MaxElevation = 3, MaxSlope = 1.5f })
                .AddRule(new BiomeRule { Biome = Biome.Cliff, MinSlope = 4f })
                .AddRule(new BiomeRule { Biome = Biome.RockySummit, MinElevation = 55 })
                .AddRule(new BiomeRule { Biome = Biome.SparseVegetation, MinElevation = 45, MinSlope = 2f })
                .SetDefault(Biome.Forest, 1);

            // Depth layers per biome
            assigner.SetDepthLayers(Biome.Beach,
                new DepthLayer { MinDepth = 0, MaxDepth = 0, BlockType = 4 }, // sand surface
                new DepthLayer { MinDepth = 1, MaxDepth = 3, BlockType = 4 }, // sand
                new DepthLayer { MinDepth = 4, BlockType = 1 }); // stone

            assigner.SetDepthLayers(Biome.Forest,
                new DepthLayer { MinDepth = 0, MaxDepth = 0, BlockType = 3 }, // grass surface
                new DepthLayer { MinDepth = 1, MaxDepth = 3, BlockType = 2 }, // dirt
                new DepthLayer { MinDepth = 4, BlockType = 1 }); // stone

            assigner.SetDepthLayers(Biome.Cliff,
                new DepthLayer { MinDepth = 0, BlockType = 1 }); // all stone

            assigner.SetDepthLayers(Biome.RockySummit,
                new DepthLayer { MinDepth = 0, MaxDepth = 1, BlockType = 5 }, // cobblestone surface
                new DepthLayer { MinDepth = 2, BlockType = 1 }); // stone

            assigner.SetDepthLayers(Biome.SparseVegetation,
                new DepthLayer { MinDepth = 0, MaxDepth = 0, BlockType = 2 }, // dirt surface (sparse grass)
                new DepthLayer { MinDepth = 1, MaxDepth = 2, BlockType = 2 }, // dirt
                new DepthLayer { MinDepth = 3, BlockType = 1 }); // stone

            assigner.SetDepthLayers(Biome.Underwater,
                new DepthLayer { MinDepth = 0, MaxDepth = 2, BlockType = 4 }, // sand
                new DepthLayer { MinDepth = 3, BlockType = 1 }); // stone

            return assigner;
        }

        /// <summary>
        /// Create a simple AubsCraft biome configuration.
        /// </summary>
        public static BiomeAssigner CreateAubsCraft()
        {
            var assigner = new BiomeAssigner { SeaLevel = 62f };

            assigner
                .AddRule(new BiomeRule { Biome = Biome.Beach, MinElevation = 0, MaxElevation = 3, MaxSlope = 1f })
                .AddRule(new BiomeRule { Biome = Biome.RockySummit, MinElevation = 100 })
                .SetDefault(Biome.Forest, 1);

            assigner.SetDepthLayers(Biome.Beach,
                new DepthLayer { MinDepth = 0, MaxDepth = 3, BlockType = 4 }, // sand
                new DepthLayer { MinDepth = 4, BlockType = 1 }); // stone

            assigner.SetDepthLayers(Biome.Forest,
                new DepthLayer { MinDepth = 0, MaxDepth = 0, BlockType = 3 }, // grass
                new DepthLayer { MinDepth = 1, MaxDepth = 3, BlockType = 2 }, // dirt
                new DepthLayer { MinDepth = 4, BlockType = 1 }); // stone

            assigner.SetDepthLayers(Biome.RockySummit,
                new DepthLayer { MinDepth = 0, BlockType = 1 }); // stone

            return assigner;
        }
    }
}
