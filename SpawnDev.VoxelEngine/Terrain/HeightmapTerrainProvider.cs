using System.Numerics;

namespace SpawnDev.VoxelEngine.Terrain
{
    /// <summary>
    /// Generates voxel block data from a heightmap.
    ///
    /// Reads a binary heightmap (Int16 per cell, meters/voxels above base level).
    /// Bilinear interpolation between cells for sub-cell resolution.
    /// Slope computation from neighbor heights for biome assignment.
    ///
    /// Promoted from Lost Spawns' HeightmapLoader.cs to the engine.
    /// </summary>
    public class HeightmapTerrainProvider
    {
        private readonly short[] _heightData;
        private readonly int _mapWidth;
        private readonly int _mapHeight;
        private readonly float _cellSize; // world units per heightmap cell

        /// <summary>Minimum height value in the heightmap.</summary>
        public short MinHeight { get; private set; }

        /// <summary>Maximum height value in the heightmap.</summary>
        public short MaxHeight { get; private set; }

        /// <summary>
        /// Create a terrain provider from raw heightmap data.
        /// </summary>
        /// <param name="heightData">Int16 height values, row-major.</param>
        /// <param name="mapWidth">Width of the heightmap in cells.</param>
        /// <param name="mapHeight">Height of the heightmap in cells.</param>
        /// <param name="cellSize">World units per heightmap cell.</param>
        public HeightmapTerrainProvider(short[] heightData, int mapWidth, int mapHeight, float cellSize = 1f)
        {
            _heightData = heightData;
            _mapWidth = mapWidth;
            _mapHeight = mapHeight;
            _cellSize = cellSize;

            MinHeight = short.MaxValue;
            MaxHeight = short.MinValue;
            foreach (var h in heightData)
            {
                if (h < MinHeight) MinHeight = h;
                if (h > MaxHeight) MaxHeight = h;
            }
        }

        /// <summary>
        /// Load heightmap from a binary file (Int16 per cell, little-endian).
        /// </summary>
        public static HeightmapTerrainProvider FromBinaryFile(byte[] data, int mapWidth, int mapHeight, float cellSize = 1f)
        {
            var heights = new short[mapWidth * mapHeight];
            Buffer.BlockCopy(data, 0, heights, 0, Math.Min(data.Length, heights.Length * 2));
            return new HeightmapTerrainProvider(heights, mapWidth, mapHeight, cellSize);
        }

        /// <summary>
        /// Get interpolated height at a world position using bilinear interpolation.
        /// </summary>
        public float GetHeight(float worldX, float worldZ)
        {
            float cellX = worldX / _cellSize;
            float cellZ = worldZ / _cellSize;

            int x0 = (int)MathF.Floor(cellX);
            int z0 = (int)MathF.Floor(cellZ);
            int x1 = x0 + 1;
            int z1 = z0 + 1;

            float fx = cellX - x0;
            float fz = cellZ - z0;

            float h00 = GetRawHeight(x0, z0);
            float h10 = GetRawHeight(x1, z0);
            float h01 = GetRawHeight(x0, z1);
            float h11 = GetRawHeight(x1, z1);

            // Bilinear interpolation
            float h0 = h00 + (h10 - h00) * fx;
            float h1 = h01 + (h11 - h01) * fx;
            return h0 + (h1 - h0) * fz;
        }

        /// <summary>
        /// Compute terrain slope at a world position (rise over run, unitless).
        /// Uses finite differences from neighbor heights.
        /// </summary>
        public float GetSlope(float worldX, float worldZ)
        {
            float delta = _cellSize;
            float hL = GetHeight(worldX - delta, worldZ);
            float hR = GetHeight(worldX + delta, worldZ);
            float hD = GetHeight(worldX, worldZ - delta);
            float hU = GetHeight(worldX, worldZ + delta);

            float dhdx = (hR - hL) / (2f * delta);
            float dhdz = (hU - hD) / (2f * delta);

            return MathF.Sqrt(dhdx * dhdx + dhdz * dhdz);
        }

        /// <summary>
        /// Compute terrain normal at a world position.
        /// </summary>
        public Vector3 GetNormal(float worldX, float worldZ)
        {
            float delta = _cellSize;
            float hL = GetHeight(worldX - delta, worldZ);
            float hR = GetHeight(worldX + delta, worldZ);
            float hD = GetHeight(worldX, worldZ - delta);
            float hU = GetHeight(worldX, worldZ + delta);

            return Vector3.Normalize(new Vector3(hL - hR, 2f * delta, hD - hU));
        }

        /// <summary>
        /// Generate section block data from the heightmap.
        /// Fills blocks below the terrain surface with appropriate types.
        /// </summary>
        /// <param name="sectionCoord">Section to generate.</param>
        /// <param name="config">Engine config (voxel size, section size, base Y).</param>
        /// <param name="assignBlock">Callback: given (worldX, worldY, worldZ, surfaceHeight, slope, depth), returns block type.</param>
        public int[] GenerateSection(
            SectionCoord sectionCoord,
            VoxelEngineConfig config,
            Func<float, float, float, float, float, int, int> assignBlock)
        {
            int ss = config.SectionSize;
            var blocks = new int[ss * ss * ss];

            for (int ly = 0; ly < ss; ly++)
            {
                for (int lz = 0; lz < ss; lz++)
                {
                    for (int lx = 0; lx < ss; lx++)
                    {
                        float worldX = (sectionCoord.Cx * ss + lx) * config.VoxelSize;
                        float worldY = config.BaseY + (sectionCoord.Sy * ss + ly) * config.VoxelSize;
                        float worldZ = (sectionCoord.Cz * ss + lz) * config.VoxelSize;

                        float surfaceHeight = GetHeight(worldX, worldZ);
                        float slope = GetSlope(worldX, worldZ);
                        int depth = (int)((surfaceHeight - worldY) / config.VoxelSize);

                        int blockType = 0; // air
                        if (worldY <= surfaceHeight)
                        {
                            blockType = assignBlock(worldX, worldY, worldZ, surfaceHeight, slope, depth);
                        }

                        blocks[lx + lz * ss + ly * ss * ss] = PackedBlock.Pack(blockType);
                    }
                }
            }

            return blocks;
        }

        private float GetRawHeight(int cellX, int cellZ)
        {
            cellX = Math.Clamp(cellX, 0, _mapWidth - 1);
            cellZ = Math.Clamp(cellZ, 0, _mapHeight - 1);
            return _heightData[cellX + cellZ * _mapWidth];
        }
    }
}
