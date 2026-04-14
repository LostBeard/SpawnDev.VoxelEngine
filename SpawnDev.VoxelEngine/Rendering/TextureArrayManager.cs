namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Manages GPU texture arrays for block rendering.
    ///
    /// Supports two modes:
    /// 1. Texture Array (preferred): texture_2d_array in WGSL, one layer per (blockType, face, variant).
    ///    Fragment shader: textureLoad(atlas, texCoord, layerIndex).
    /// 2. Atlas fallback: single texture with UV sub-regions (AubsCraft compat).
    ///
    /// PBR mode (Lost Spawns): 3 layers per block face variant (diffuse + normal + roughness).
    /// Simple mode (AubsCraft): 1 layer per block face variant (diffuse only).
    ///
    /// Texture variants (4 per block type) break visual monotony on large surfaces.
    /// Variant selection: hash(blockWorldPos) % 4 - deterministic, same position = same variant.
    /// </summary>
    public class TextureArrayManager
    {
        /// <summary>Texture mode.</summary>
        public enum TextureMode
        {
            /// <summary>Texture array with per-layer addressing. Best quality + performance.</summary>
            TextureArray,

            /// <summary>Single atlas texture with UV sub-regions. Compatibility fallback.</summary>
            Atlas,
        }

        /// <summary>Material mode for block rendering.</summary>
        public enum MaterialMode
        {
            /// <summary>Diffuse only. One texture layer per face variant. Fast, simple.</summary>
            Simple,

            /// <summary>PBR: diffuse + normal + roughness. Three layers per face variant.</summary>
            PBR,
        }

        /// <summary>Current texture mode.</summary>
        public TextureMode Mode { get; set; } = TextureMode.TextureArray;

        /// <summary>Current material mode.</summary>
        public MaterialMode Material { get; set; } = MaterialMode.Simple;

        /// <summary>Number of texture variants per block type.</summary>
        public int VariantsPerType { get; set; } = 4;

        /// <summary>Texture resolution per layer (square).</summary>
        public int TextureResolution { get; set; } = 256;

        /// <summary>Number of registered block types with textures.</summary>
        public int RegisteredTypeCount { get; private set; }

        // Layer mapping: blockType -> base layer index in the texture array
        private readonly Dictionary<int, int> _typeToBaseLayer = new();
        private int _nextLayer;

        /// <summary>Layers per face variant in the current material mode.</summary>
        public int LayersPerVariant => Material == MaterialMode.PBR ? 3 : 1;

        /// <summary>Total layers per block type (variants * faces * layers per variant).</summary>
        public int LayersPerType => VariantsPerType * 6 * LayersPerVariant;

        /// <summary>
        /// Register a block type for texture array allocation.
        /// Reserves layers in the texture array for all variants and faces.
        /// </summary>
        public int RegisterBlockType(int blockType)
        {
            if (_typeToBaseLayer.ContainsKey(blockType))
                return _typeToBaseLayer[blockType];

            int baseLayer = _nextLayer;
            _typeToBaseLayer[blockType] = baseLayer;
            _nextLayer += LayersPerType;
            RegisteredTypeCount++;
            return baseLayer;
        }

        /// <summary>
        /// Get the texture array layer index for a specific block face and variant.
        /// </summary>
        /// <param name="blockType">Block type ID.</param>
        /// <param name="face">Face direction (0-5).</param>
        /// <param name="variant">Texture variant (0 to VariantsPerType-1).</param>
        /// <param name="sublayer">0=diffuse, 1=normal, 2=roughness (PBR mode only).</param>
        public int GetLayerIndex(int blockType, int face, int variant, int sublayer = 0)
        {
            if (!_typeToBaseLayer.TryGetValue(blockType, out int baseLayer))
                return 0; // fallback to layer 0

            return baseLayer
                + variant * 6 * LayersPerVariant  // skip to variant
                + face * LayersPerVariant          // skip to face
                + sublayer;                        // sublayer (0 for simple mode)
        }

        /// <summary>
        /// Total number of layers needed in the texture array.
        /// Use this to allocate the GPU texture array.
        /// </summary>
        public int TotalLayers => _nextLayer;

        /// <summary>
        /// Compute deterministic texture variant for a block position.
        /// Same position always produces the same variant across sessions.
        /// </summary>
        public static int GetVariant(int worldX, int worldY, int worldZ, int variantCount = 4)
        {
            // Simple hash that produces good distribution
            int hash = worldX * 73856093 ^ worldY * 19349663 ^ worldZ * 83492791;
            return ((hash % variantCount) + variantCount) % variantCount; // ensure positive
        }

        /// <summary>
        /// Generate WGSL fragment shader code for texture sampling.
        /// </summary>
        public string GenerateWGSLSampling()
        {
            if (Material == MaterialMode.PBR)
            {
                return @"
let diffuse = textureSample(blockTextures, texSampler, uv, layerBase);
let normalMap = textureSample(blockTextures, texSampler, uv, layerBase + 1);
let roughness = textureSample(blockTextures, texSampler, uv, layerBase + 2).r;
let normal = normalMap.xyz * 2.0 - 1.0;
";
            }

            return @"
let diffuse = textureSample(blockTextures, texSampler, uv, layerIdx);
";
        }
    }
}
