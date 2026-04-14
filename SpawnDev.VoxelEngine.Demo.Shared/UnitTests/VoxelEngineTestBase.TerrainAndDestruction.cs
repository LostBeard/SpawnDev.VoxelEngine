using System.Numerics;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Terrain;
using SpawnDev.VoxelEngine.Destruction;
using SpawnDev.VoxelEngine.Physics;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Terrain generation, explosion destruction, and structural integrity tests.
    // HeightmapTerrainProvider: heightmap sampling, bilinear interpolation, slope.
    // BiomeAssigner: elevation/slope-based classification, depth layers.
    // ExplosionKernels: sphere destruction with blast resistance.
    // StructuralIntegrity: BFS ground-connectedness, gravity fallers.
    // All tests are CPU-only - testing the CPU reference implementations.
    public abstract partial class VoxelEngineTestBase
    {
        // ===== HeightmapTerrainProvider tests =====

        /// <summary>
        /// Test 1: Flat heightmap produces correct constant height.
        /// A 4x4 heightmap with all cells at height 50 should return 50 at any sample point.
        /// </summary>
        [TestMethod]
        public void Terrain_FlatHeightmap_ConstantHeightTest()
        {
            int mapW = 4, mapH = 4;
            short flatHeight = 50;
            var heights = new short[mapW * mapH];
            for (int i = 0; i < heights.Length; i++)
                heights[i] = flatHeight;

            var terrain = new HeightmapTerrainProvider(heights, mapW, mapH, cellSize: 1f);

            // MinHeight and MaxHeight should both be 50
            if (terrain.MinHeight != flatHeight)
                throw new Exception($"Terrain_Flat: MinHeight should be {flatHeight}, got {terrain.MinHeight}");
            if (terrain.MaxHeight != flatHeight)
                throw new Exception($"Terrain_Flat: MaxHeight should be {flatHeight}, got {terrain.MaxHeight}");

            // Sample at various positions - all should return 50
            float[] testX = { 0f, 0.5f, 1.3f, 2.7f, 3f };
            float[] testZ = { 0f, 0.5f, 1.3f, 2.7f, 3f };
            foreach (float wx in testX)
            {
                foreach (float wz in testZ)
                {
                    float h = terrain.GetHeight(wx, wz);
                    if (MathF.Abs(h - flatHeight) > 0.001f)
                        throw new Exception(
                            $"Terrain_Flat: height at ({wx}, {wz}) should be {flatHeight}, got {h}");
                }
            }
        }

        /// <summary>
        /// Test 2: Bilinear interpolation between cells is smooth.
        /// 4x4 heightmap with known gradient: left=0, right=100.
        /// Sampling at midpoints should return interpolated values.
        /// </summary>
        [TestMethod]
        public void Terrain_BilinearInterpolation_SmoothTest()
        {
            int mapW = 4, mapH = 4;
            var heights = new short[mapW * mapH];

            // Create X gradient: column 0=0, column 1=33, column 2=67, column 3=100
            for (int z = 0; z < mapH; z++)
            {
                for (int x = 0; x < mapW; x++)
                {
                    heights[x + z * mapW] = (short)(x * 100 / (mapW - 1)); // 0, 33, 67, 100
                }
            }

            var terrain = new HeightmapTerrainProvider(heights, mapW, mapH, cellSize: 1f);

            // Sample at cell centers (exact grid points)
            float h0 = terrain.GetHeight(0f, 0f);
            float h1 = terrain.GetHeight(1f, 0f);
            float h3 = terrain.GetHeight(3f, 0f);

            if (MathF.Abs(h0 - 0f) > 0.5f)
                throw new Exception($"Terrain_Bilinear: height at (0,0) should be ~0, got {h0}");
            if (MathF.Abs(h1 - 33f) > 0.5f)
                throw new Exception($"Terrain_Bilinear: height at (1,0) should be ~33, got {h1}");
            if (MathF.Abs(h3 - 100f) > 0.5f)
                throw new Exception($"Terrain_Bilinear: height at (3,0) should be ~100, got {h3}");

            // Sample at midpoint between cells 0 and 1: should be ~16.5 (midpoint of 0 and 33)
            float hMid = terrain.GetHeight(0.5f, 0f);
            float expectedMid = (0f + 33f) / 2f;
            if (MathF.Abs(hMid - expectedMid) > 1f)
                throw new Exception(
                    $"Terrain_Bilinear: height at (0.5, 0) should be ~{expectedMid:F1} " +
                    $"(interpolated between 0 and 33), got {hMid:F1}");

            // Verify monotonic increase along X
            float prev = terrain.GetHeight(0f, 1.5f);
            for (float x = 0.25f; x <= 3f; x += 0.25f)
            {
                float current = terrain.GetHeight(x, 1.5f);
                if (current < prev - 0.01f) // allow tiny floating point noise
                    throw new Exception(
                        $"Terrain_Bilinear: height should be monotonically increasing along X gradient, " +
                        $"but {current:F2} at x={x} < {prev:F2} at x={x - 0.25f}");
                prev = current;
            }
        }

        /// <summary>
        /// Test 3: Slope computation - flat terrain = 0, steep terrain = high value.
        /// A flat heightmap should have zero slope everywhere.
        /// A tilted heightmap (consistent gradient) should have a non-zero slope.
        /// </summary>
        [TestMethod]
        public void Terrain_SlopeComputation_FlatVsSteepTest()
        {
            int mapW = 8, mapH = 8;

            // Case A: Flat terrain - all height = 50
            var flatHeights = new short[mapW * mapH];
            for (int i = 0; i < flatHeights.Length; i++)
                flatHeights[i] = 50;
            var flatTerrain = new HeightmapTerrainProvider(flatHeights, mapW, mapH, cellSize: 1f);

            float flatSlope = flatTerrain.GetSlope(4f, 4f);
            if (MathF.Abs(flatSlope) > 0.01f)
                throw new Exception(
                    $"Terrain_Slope: flat terrain slope should be ~0, got {flatSlope:F4}");

            // Case B: Steep terrain - linear gradient along X: height = x * 10
            var steepHeights = new short[mapW * mapH];
            for (int z = 0; z < mapH; z++)
                for (int x = 0; x < mapW; x++)
                    steepHeights[x + z * mapW] = (short)(x * 10);
            var steepTerrain = new HeightmapTerrainProvider(steepHeights, mapW, mapH, cellSize: 1f);

            float steepSlope = steepTerrain.GetSlope(4f, 4f);
            if (steepSlope < 1f)
                throw new Exception(
                    $"Terrain_Slope: steep terrain (10 units/cell gradient) slope should be > 1, got {steepSlope:F4}");

            // Verify steep is much greater than flat
            if (steepSlope <= flatSlope + 0.1f)
                throw new Exception(
                    $"Terrain_Slope: steep ({steepSlope:F4}) should be much greater than flat ({flatSlope:F4})");

            // Case C: Normal vector on flat terrain should point straight up
            var flatNormal = flatTerrain.GetNormal(4f, 4f);
            if (MathF.Abs(flatNormal.Y - 1f) > 0.01f)
                throw new Exception(
                    $"Terrain_Slope: flat terrain normal Y should be ~1.0 (straight up), " +
                    $"got ({flatNormal.X:F3}, {flatNormal.Y:F3}, {flatNormal.Z:F3})");
        }

        // ===== BiomeAssigner tests =====

        /// <summary>
        /// Test 4: Beach at sea level, cliff at high slope, forest default.
        /// Uses the Lost Spawns biome configuration with known elevation/slope inputs.
        /// </summary>
        [TestMethod]
        public void Biome_Classification_CorrectBiomesTest()
        {
            var assigner = BiomeAssigner.CreateLostSpawns();

            // Beach: elevation 0-3, slope < 1.5
            var beachBiome = assigner.Classify(elevation: 1.5f, slope: 0.5f);
            if (beachBiome != BiomeAssigner.Biome.Beach)
                throw new Exception(
                    $"Biome_Classification: elevation=1.5, slope=0.5 should be Beach, got {beachBiome}");

            // Cliff: any elevation, slope >= 4.0
            var cliffBiome = assigner.Classify(elevation: 20f, slope: 5f);
            if (cliffBiome != BiomeAssigner.Biome.Cliff)
                throw new Exception(
                    $"Biome_Classification: elevation=20, slope=5.0 should be Cliff, got {cliffBiome}");

            // Forest: default (mid elevation, moderate slope)
            var forestBiome = assigner.Classify(elevation: 25f, slope: 1f);
            if (forestBiome != BiomeAssigner.Biome.Forest)
                throw new Exception(
                    $"Biome_Classification: elevation=25, slope=1.0 should be Forest (default), got {forestBiome}");

            // Rocky summit: elevation >= 55
            var summitBiome = assigner.Classify(elevation: 60f, slope: 1f);
            if (summitBiome != BiomeAssigner.Biome.RockySummit)
                throw new Exception(
                    $"Biome_Classification: elevation=60, slope=1.0 should be RockySummit, got {summitBiome}");

            // Underwater: below sea level (elevation < 0, which means surfaceHeight < SeaLevel)
            var underwaterBiome = assigner.Classify(elevation: -2f, slope: 0f);
            if (underwaterBiome != BiomeAssigner.Biome.Underwater)
                throw new Exception(
                    $"Biome_Classification: elevation=-2 should be Underwater, got {underwaterBiome}");

            // Sparse vegetation: elevation 45+, slope >= 2
            var sparseBiome = assigner.Classify(elevation: 48f, slope: 2.5f);
            if (sparseBiome != BiomeAssigner.Biome.SparseVegetation)
                throw new Exception(
                    $"Biome_Classification: elevation=48, slope=2.5 should be SparseVegetation, got {sparseBiome}");
        }

        /// <summary>
        /// Test 5: Depth layers - grass surface, dirt below, stone deep.
        /// Forest biome: depth 0=grass(3), depth 1-3=dirt(2), depth 4+=stone(1).
        /// </summary>
        [TestMethod]
        public void Biome_DepthLayers_CorrectBlockTypesTest()
        {
            var assigner = BiomeAssigner.CreateLostSpawns();

            // Forest depth layers
            int surfaceBlock = assigner.GetBlockType(BiomeAssigner.Biome.Forest, 0);
            if (surfaceBlock != 3) // grass
                throw new Exception(
                    $"Biome_DepthLayers: Forest depth 0 should be grass (3), got {surfaceBlock}");

            int dirtBlock1 = assigner.GetBlockType(BiomeAssigner.Biome.Forest, 1);
            int dirtBlock3 = assigner.GetBlockType(BiomeAssigner.Biome.Forest, 3);
            if (dirtBlock1 != 2) // dirt
                throw new Exception(
                    $"Biome_DepthLayers: Forest depth 1 should be dirt (2), got {dirtBlock1}");
            if (dirtBlock3 != 2) // dirt
                throw new Exception(
                    $"Biome_DepthLayers: Forest depth 3 should be dirt (2), got {dirtBlock3}");

            int stoneBlock = assigner.GetBlockType(BiomeAssigner.Biome.Forest, 4);
            int deepStone = assigner.GetBlockType(BiomeAssigner.Biome.Forest, 50);
            if (stoneBlock != 1) // stone
                throw new Exception(
                    $"Biome_DepthLayers: Forest depth 4 should be stone (1), got {stoneBlock}");
            if (deepStone != 1) // stone
                throw new Exception(
                    $"Biome_DepthLayers: Forest depth 50 should be stone (1), got {deepStone}");

            // Beach depth layers: sand surface and subsurface, then stone
            int beachSurface = assigner.GetBlockType(BiomeAssigner.Biome.Beach, 0);
            int beachSand = assigner.GetBlockType(BiomeAssigner.Biome.Beach, 2);
            int beachStone = assigner.GetBlockType(BiomeAssigner.Biome.Beach, 4);
            if (beachSurface != 4) // sand
                throw new Exception(
                    $"Biome_DepthLayers: Beach depth 0 should be sand (4), got {beachSurface}");
            if (beachSand != 4) // sand
                throw new Exception(
                    $"Biome_DepthLayers: Beach depth 2 should be sand (4), got {beachSand}");
            if (beachStone != 1) // stone
                throw new Exception(
                    $"Biome_DepthLayers: Beach depth 4 should be stone (1), got {beachStone}");

            // Cliff: all stone at every depth
            int cliffSurface = assigner.GetBlockType(BiomeAssigner.Biome.Cliff, 0);
            int cliffDeep = assigner.GetBlockType(BiomeAssigner.Biome.Cliff, 10);
            if (cliffSurface != 1) // stone
                throw new Exception(
                    $"Biome_DepthLayers: Cliff depth 0 should be stone (1), got {cliffSurface}");
            if (cliffDeep != 1) // stone
                throw new Exception(
                    $"Biome_DepthLayers: Cliff depth 10 should be stone (1), got {cliffDeep}");

            // CreateAssigner callback integration test
            var assignerFunc = assigner.CreateAssigner();
            // Forest at elevation 20 (above sea level 0), slope 0.5, depth 0 = grass
            int assignedBlock = assignerFunc(10f, 20f, 10f, 20f, 0.5f, 0);
            if (assignedBlock != 3) // grass (forest surface)
                throw new Exception(
                    $"Biome_DepthLayers: CreateAssigner for forest surface should return grass (3), got {assignedBlock}");
        }

        // ===== ExplosionKernels tests =====

        /// <summary>
        /// Test 6: DestroyInSphere removes correct block count.
        /// Place a known 3x3x3 cube (27 blocks) and explode with a radius that covers all of them.
        /// Verify exact destroyed count.
        /// </summary>
        [TestMethod]
        public void Explosion_DestroySphere_CorrectCountTest()
        {
            int sizeXZ = 16, sizeY = 16;
            var blocks = new int[sizeXZ * sizeXZ * sizeY];

            // Place 3x3x3 cube of dirt (type 2) at (7,7,7) to (9,9,9)
            int placed = 0;
            for (int y = 7; y <= 9; y++)
                for (int z = 7; z <= 9; z++)
                    for (int x = 7; x <= 9; x++)
                    {
                        blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ] = PackedBlock.Pack(2); // dirt
                        placed++;
                    }

            if (placed != 27)
                throw new Exception($"Explosion_Destroy: placed {placed} blocks, expected 27");

            // Explode centered at cube center (8.5, 8.5, 8.5) with radius 3.0
            // All 27 blocks: farthest corner center is (7.5,7.5,7.5), distance = sqrt(3) = 1.73 < 3.0
            int destroyed = ExplosionKernels.DestroyInSphere(
                blocks, sizeXZ, sizeY,
                center: new Vector3(8.5f, 8.5f, 8.5f),
                radius: 3.0f);

            if (destroyed != 27)
                throw new Exception(
                    $"Explosion_Destroy: expected 27 blocks destroyed (radius 3.0 covers entire 3x3x3 cube), " +
                    $"got {destroyed}");

            // Verify all blocks are now air
            for (int y = 7; y <= 9; y++)
                for (int z = 7; z <= 9; z++)
                    for (int x = 7; x <= 9; x++)
                    {
                        int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                        if (blocks[idx] != 0)
                            throw new Exception(
                                $"Explosion_Destroy: block at ({x},{y},{z}) should be air after explosion, " +
                                $"got type {blocks[idx] & 0xFFF}");
                    }
        }

        /// <summary>
        /// Test 7: Blast resistance - weak blocks destroyed, strong blocks survive.
        /// Place dirt (blast resistance 0.6) and stone (blast resistance 6.0) side by side.
        /// Explode with moderate power - dirt should be destroyed, stone should survive.
        /// </summary>
        [TestMethod]
        public void Explosion_BlastResistance_WeakDestroyedStrongSurvivesTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();
            // Dirt (type 2): BlastResistance = 0.6
            // Stone (type 1): BlastResistance = 6.0

            int sizeXZ = 16, sizeY = 16;
            var blocks = new int[sizeXZ * sizeXZ * sizeY];

            // Place 3 dirt blocks and 3 stone blocks all at same distance from blast center
            // Dirt at y=8, z=8, x=7,8,9
            blocks[7 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(2); // dirt
            blocks[8 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(2); // dirt
            blocks[9 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(2); // dirt
            // Stone at y=8, z=9, x=7,8,9
            blocks[7 + 9 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // stone
            blocks[8 + 9 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // stone
            blocks[9 + 9 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // stone

            // Blast center at (8.5, 8.5, 8.5), radius 3.0, power 5.0
            // effectivePower = 5.0 / (1 + distSq)
            // For blocks at y=8, z=8: distSq to (8.5,8.5,8.5) varies
            //   center block (8,8,8): dist = sqrt(0.25+0.25+0.25) = 0.866, distSq = 0.75
            //   effectivePower = 5.0 / 1.75 = 2.86 > 0.6 (dirt resistance) -> DESTROYED
            //   effectivePower = 5.0 / 1.75 = 2.86 < 6.0 (stone resistance) -> SURVIVES
            int destroyed = ExplosionKernels.DestroyInSphere(
                blocks, sizeXZ, sizeY,
                center: new Vector3(8.5f, 8.5f, 8.5f),
                radius: 3.0f,
                blastPower: 5.0f,
                registry: registry);

            // All 3 dirt blocks should be destroyed
            for (int x = 7; x <= 9; x++)
            {
                int idx = x + 8 * sizeXZ + 8 * sizeXZ * sizeXZ;
                int blockType = blocks[idx] & 0xFFF;
                if (blockType != 0)
                    throw new Exception(
                        $"Explosion_BlastResistance: dirt at ({x},8,8) should be destroyed (resistance=0.6, power=5), " +
                        $"but type is still {blockType}");
            }

            // All 3 stone blocks should survive
            for (int x = 7; x <= 9; x++)
            {
                int idx = x + 9 * sizeXZ + 8 * sizeXZ * sizeXZ;
                int blockType = blocks[idx] & 0xFFF;
                if (blockType != 1)
                    throw new Exception(
                        $"Explosion_BlastResistance: stone at ({x},8,9) should survive (resistance=6.0, power=5), " +
                        $"but type is {blockType}");
            }

            if (destroyed != 3)
                throw new Exception(
                    $"Explosion_BlastResistance: expected 3 blocks destroyed (3 dirt), got {destroyed}");
        }

        /// <summary>
        /// Test 8: DamageInSphere applies proportional damage (closer = more damage).
        /// Place blocks at varying distances from blast center and verify damage scaling.
        /// </summary>
        [TestMethod]
        public void Explosion_DamageSphere_ProportionalDamageTest()
        {
            int sizeXZ = 16, sizeY = 16;
            var blocks = new int[sizeXZ * sizeXZ * sizeY];

            // Place stone blocks at increasing distances from center (8.5, 8.5, 8.5)
            // Close block at (8, 8, 8): dist = 0.866
            // Mid block at (6, 8, 8): dist = 2.5
            // Far block at (4, 8, 8): dist = 4.5
            blocks[8 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // close
            blocks[6 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // mid
            blocks[4 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // far

            int affected = ExplosionKernels.DamageInSphere(
                blocks, sizeXZ, sizeY,
                center: new Vector3(8.5f, 8.5f, 8.5f),
                radius: 5.0f,
                maxDamage: 15);

            if (affected != 3)
                throw new Exception(
                    $"Explosion_Damage: expected 3 blocks affected, got {affected}");

            // Close block should have more damage than far block
            int closeDamage = PackedBlock.GetDamage(blocks[8 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ]);
            int midDamage = PackedBlock.GetDamage(blocks[6 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ]);
            int farDamage = PackedBlock.GetDamage(blocks[4 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ]);

            // Close block: t = 1 - 0.866/5 = 0.827, damage = min(15, 12+1) = 13
            // Note: close blocks with maxDamage=15 at distance ~0.87 get very high damage,
            // possibly destroyed (damage >= 15 means turned to air)
            // We need to check if block was destroyed or just damaged
            int closeType = blocks[8 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ] & 0xFFF;
            if (closeType == 0)
            {
                // Destroyed (damage >= 15) - this is valid for the closest block
                closeDamage = 15; // treat as max for comparison
            }

            // Far block should still exist and have less damage
            int farType = blocks[4 + 8 * sizeXZ + 8 * sizeXZ * sizeXZ] & 0xFFF;
            if (farType == 0)
            {
                // Far block destroyed too - still check ordering makes sense
                farDamage = 15;
            }

            // At minimum, verify damage is non-zero for all affected blocks
            // (close might be destroyed, far should have damage but might survive)
            if (closeDamage <= 0 && closeType != 0)
                throw new Exception("Explosion_Damage: close block should have damage > 0");

            // The key invariant: closer blocks should have >= damage than farther blocks
            if (closeDamage < farDamage)
                throw new Exception(
                    $"Explosion_Damage: close block damage ({closeDamage}) should be >= " +
                    $"far block damage ({farDamage}), closer blocks take more damage");

            if (midDamage < farDamage && midDamage > 0)
                throw new Exception(
                    $"Explosion_Damage: mid block damage ({midDamage}) should be >= " +
                    $"far block damage ({farDamage})");
        }

        // ===== StructuralIntegrity tests =====

        /// <summary>
        /// Test 9: Pillar with base removed -> entire pillar unsupported.
        /// Build a 5-tall pillar of stone, remove the bottom block, verify all remaining
        /// blocks are unsupported (not connected to ground).
        /// </summary>
        [TestMethod]
        public void Structure_PillarBaseRemoved_AllUnsupportedTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();
            // Stone (type 1): IsStructural = true (FlagStructural set)

            int sizeXZ = 8, sizeY = 16;
            var blocks = new int[sizeXZ * sizeXZ * sizeY];

            // Build pillar of stone at x=4, z=4, y=1 to y=5 (base at y=1, not y=0)
            // y=0 is empty (no ground connection!)
            for (int y = 1; y <= 5; y++)
                blocks[4 + 4 * sizeXZ + y * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // stone

            var unsupported = StructuralIntegrity.FindUnsupported(blocks, sizeXZ, sizeY, registry);

            // All 5 blocks should be unsupported (none at y=0 to seed the BFS)
            if (unsupported.Count != 5)
                throw new Exception(
                    $"Structure_PillarBaseRemoved: expected 5 unsupported blocks " +
                    $"(pillar not touching ground), got {unsupported.Count}");

            // Verify all unsupported blocks are stone at x=4, z=4
            foreach (var (x, y, z, blockType) in unsupported)
            {
                if (x != 4 || z != 4)
                    throw new Exception(
                        $"Structure_PillarBaseRemoved: unsupported block at ({x},{y},{z}) " +
                        $"should be at x=4, z=4");
                if (blockType != 1)
                    throw new Exception(
                        $"Structure_PillarBaseRemoved: unsupported block type should be stone (1), got {blockType}");
            }
        }

        /// <summary>
        /// Test 10: Block connected to ground -> supported.
        /// Build a pillar of stone starting at y=0 (ground level). All blocks should be supported.
        /// </summary>
        [TestMethod]
        public void Structure_GroundConnected_AllSupportedTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            int sizeXZ = 8, sizeY = 16;
            var blocks = new int[sizeXZ * sizeXZ * sizeY];

            // Build pillar from y=0 (ground) to y=7
            for (int y = 0; y <= 7; y++)
                blocks[4 + 4 * sizeXZ + y * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // stone

            var unsupported = StructuralIntegrity.FindUnsupported(blocks, sizeXZ, sizeY, registry);

            if (unsupported.Count != 0)
                throw new Exception(
                    $"Structure_GroundConnected: pillar starting at ground should have 0 unsupported blocks, " +
                    $"got {unsupported.Count}. First unsupported: ({unsupported[0].x},{unsupported[0].y},{unsupported[0].z})");

            // Also test with a wider base: 3x3 floor connected to a pillar
            var blocks2 = new int[sizeXZ * sizeXZ * sizeY];
            // 3x3 floor at y=0
            for (int z = 3; z <= 5; z++)
                for (int x = 3; x <= 5; x++)
                    blocks2[x + z * sizeXZ + 0] = PackedBlock.Pack(1); // stone floor
            // Pillar from y=1 to y=5 at center
            for (int y = 1; y <= 5; y++)
                blocks2[4 + 4 * sizeXZ + y * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // stone

            var unsupported2 = StructuralIntegrity.FindUnsupported(blocks2, sizeXZ, sizeY, registry);
            if (unsupported2.Count != 0)
                throw new Exception(
                    $"Structure_GroundConnected: pillar on 3x3 floor should have 0 unsupported, " +
                    $"got {unsupported2.Count}");
        }

        /// <summary>
        /// Test 11: Gravity block over air -> in faller list.
        /// Sand (type 4, has FlagGravity) placed above air should appear in FindGravityFallers.
        /// Stone (type 1, no gravity) above air should NOT appear.
        /// </summary>
        [TestMethod]
        public void Structure_GravityFaller_SandOverAirTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();
            // Sand (type 4): HasGravity = true (FlagGravity set)
            // Stone (type 1): HasGravity = false

            int sizeXZ = 8, sizeY = 16;
            var blocks = new int[sizeXZ * sizeXZ * sizeY];

            // Sand at y=5, air below at y=4 -> should fall
            blocks[4 + 4 * sizeXZ + 5 * sizeXZ * sizeXZ] = PackedBlock.Pack(4); // sand

            // Stone at y=5 at different position, air below at y=4 -> should NOT fall (no gravity)
            blocks[6 + 4 * sizeXZ + 5 * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // stone

            // Sand at y=3, stone below at y=2 -> should NOT fall (supported)
            blocks[4 + 6 * sizeXZ + 3 * sizeXZ * sizeXZ] = PackedBlock.Pack(4); // sand
            blocks[4 + 6 * sizeXZ + 2 * sizeXZ * sizeXZ] = PackedBlock.Pack(1); // stone support

            // Cascading: sand at y=8, sand at y=7, air at y=6 -> both should fall
            blocks[2 + 2 * sizeXZ + 8 * sizeXZ * sizeXZ] = PackedBlock.Pack(4); // sand at y=8
            blocks[2 + 2 * sizeXZ + 7 * sizeXZ * sizeXZ] = PackedBlock.Pack(4); // sand at y=7
            // y=6 is air

            var fallers = StructuralIntegrity.FindGravityFallers(blocks, sizeXZ, sizeY, registry);

            // Expected fallers:
            // (4,5,4) - sand over air
            // (2,8,2) - sand over sand (which is over air - but FindGravityFallers checks direct below only)
            //   Wait - y=7 is sand, not air. So (2,8,2) has sand below -> NOT a faller
            //   But (2,7,2) has air below at y=6 -> IS a faller
            // So expected: (4,5,4) and (2,7,2) = 2 fallers

            // Check that sand over air at (4,5,4) is a faller
            bool foundSandOverAir = false;
            foreach (var (x, y, z, blockType) in fallers)
            {
                if (x == 4 && y == 5 && z == 4 && blockType == 4)
                    foundSandOverAir = true;
            }
            if (!foundSandOverAir)
                throw new Exception(
                    $"Structure_GravityFaller: sand at (4,5,4) over air should be a faller, but wasn't found. " +
                    $"Found {fallers.Count} fallers total.");

            // Check that sand at (2,7,2) over air at y=6 is a faller
            bool foundCascadeSand = false;
            foreach (var (x, y, z, blockType) in fallers)
            {
                if (x == 2 && y == 7 && z == 2 && blockType == 4)
                    foundCascadeSand = true;
            }
            if (!foundCascadeSand)
                throw new Exception(
                    $"Structure_GravityFaller: sand at (2,7,2) over air should be a faller");

            // Check that stone at (6,5,4) is NOT a faller (no gravity flag)
            foreach (var (x, y, z, blockType) in fallers)
            {
                if (x == 6 && y == 5 && z == 4)
                    throw new Exception(
                        $"Structure_GravityFaller: stone at (6,5,4) should NOT be a faller (no gravity flag)");
            }

            // Check that supported sand at (4,3,6) is NOT a faller (stone below)
            foreach (var (x, y, z, blockType) in fallers)
            {
                if (x == 4 && y == 3 && z == 6)
                    throw new Exception(
                        $"Structure_GravityFaller: sand at (4,3,6) over stone should NOT be a faller");
            }

            // Verify fallers are sorted top-to-bottom (y descending)
            for (int i = 1; i < fallers.Count; i++)
            {
                if (fallers[i].y > fallers[i - 1].y)
                    throw new Exception(
                        $"Structure_GravityFaller: fallers should be sorted top-to-bottom (descending y), " +
                        $"but y={fallers[i].y} at index {i} > y={fallers[i - 1].y} at index {i - 1}");
            }
        }
    }
}
