using System.Numerics;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// Cross-quad mesher for plant-type blocks (tall grass, flowers, saplings, mushrooms).
    ///
    /// Plants render as two X-shaped quads (4 triangles) centered in the block,
    /// rotated 45 degrees from each other. Not axis-aligned, not greedy-merged.
    /// Each plant block produces exactly 2 quads.
    ///
    /// The cross pattern ensures visibility from all horizontal viewing angles.
    /// Plants are not solid (no collision), not greedy-merged (each is unique),
    /// and alpha-tested (not alpha-blended - no draw order issues).
    /// </summary>
    public static class CrossQuadMesher
    {
        /// <summary>
        /// A cross-quad vertex: position + UV + block type for texture lookup.
        /// </summary>
        public struct CrossQuadVertex
        {
            public Vector3 Position;
            public Vector2 UV;
            public int BlockType;
        }

        /// <summary>
        /// Generate cross-quad vertices for all plant blocks in a section.
        /// Returns vertex array (6 vertices per quad, 2 quads per plant = 12 vertices per plant).
        /// </summary>
        public static List<CrossQuadVertex> GenerateCrossQuads(
            ReadOnlySpan<int> blocks,
            int sizeXZ, int sizeY,
            float voxelSize,
            BlockRegistry registry)
        {
            var vertices = new List<CrossQuadVertex>();

            for (int y = 0; y < sizeY; y++)
            {
                for (int z = 0; z < sizeXZ; z++)
                {
                    for (int x = 0; x < sizeXZ; x++)
                    {
                        int packed = blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ];
                        int blockType = packed & 0xFFF;
                        if (blockType == 0) continue;

                        var props = registry.Get(blockType);
                        if (!props.IsPlant) continue;

                        // Center of this block in section-local space
                        float cx = (x + 0.5f) * voxelSize;
                        float cy = y * voxelSize;
                        float cz = (z + 0.5f) * voxelSize;

                        // Cross-quad size (slightly smaller than voxel to avoid z-fighting with neighbors)
                        float halfSize = voxelSize * 0.45f;
                        float height = voxelSize; // full voxel height

                        // Deterministic rotation offset based on block position (breaks visual repetition)
                        float rotOffset = ((x * 7 + z * 13 + y * 3) % 4) * 0.15f;

                        // Quad 1: diagonal from (-X,-Z) to (+X,+Z)
                        float dx1 = halfSize * MathF.Cos(MathF.PI * 0.25f + rotOffset);
                        float dz1 = halfSize * MathF.Sin(MathF.PI * 0.25f + rotOffset);

                        AddQuad(vertices, blockType,
                            new Vector3(cx - dx1, cy, cz - dz1),
                            new Vector3(cx + dx1, cy, cz + dz1),
                            new Vector3(cx - dx1, cy + height, cz - dz1),
                            new Vector3(cx + dx1, cy + height, cz + dz1));

                        // Quad 2: perpendicular diagonal
                        float dx2 = halfSize * MathF.Cos(MathF.PI * 0.75f + rotOffset);
                        float dz2 = halfSize * MathF.Sin(MathF.PI * 0.75f + rotOffset);

                        AddQuad(vertices, blockType,
                            new Vector3(cx - dx2, cy, cz - dz2),
                            new Vector3(cx + dx2, cy, cz + dz2),
                            new Vector3(cx - dx2, cy + height, cz - dz2),
                            new Vector3(cx + dx2, cy + height, cz + dz2));
                    }
                }
            }

            return vertices;
        }

        /// <summary>Count plant blocks in a section (quick pre-check).</summary>
        public static int CountPlantBlocks(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY, BlockRegistry registry)
        {
            int count = 0;
            int total = sizeXZ * sizeXZ * sizeY;
            for (int i = 0; i < total; i++)
            {
                int type = blocks[i] & 0xFFF;
                if (type != 0 && registry.Get(type).IsPlant) count++;
            }
            return count;
        }

        private static void AddQuad(List<CrossQuadVertex> vertices, int blockType,
            Vector3 bl, Vector3 br, Vector3 tl, Vector3 tr)
        {
            // Two triangles: BL-BR-TL, BR-TR-TL (double-sided via winding order)
            // Front face
            vertices.Add(new CrossQuadVertex { Position = bl, UV = new Vector2(0, 1), BlockType = blockType });
            vertices.Add(new CrossQuadVertex { Position = br, UV = new Vector2(1, 1), BlockType = blockType });
            vertices.Add(new CrossQuadVertex { Position = tl, UV = new Vector2(0, 0), BlockType = blockType });
            vertices.Add(new CrossQuadVertex { Position = br, UV = new Vector2(1, 1), BlockType = blockType });
            vertices.Add(new CrossQuadVertex { Position = tr, UV = new Vector2(1, 0), BlockType = blockType });
            vertices.Add(new CrossQuadVertex { Position = tl, UV = new Vector2(0, 0), BlockType = blockType });
        }
    }
}
