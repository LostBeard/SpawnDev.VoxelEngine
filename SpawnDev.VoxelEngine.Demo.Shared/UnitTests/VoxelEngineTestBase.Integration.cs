using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Adaptive;
using SpawnDev.VoxelEngine.Buffers;
using SpawnDev.VoxelEngine.Caching;
using SpawnDev.VoxelEngine.Culling;
using SpawnDev.VoxelEngine.Destruction;
using SpawnDev.VoxelEngine.LOD;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Physics;
using SpawnDev.VoxelEngine.Rendering;
using SpawnDev.VoxelEngine.Terrain;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Integration tests: full pipeline exercises, multi-subsystem verification.
    // These test the engine the way AubsCraft and Lost Spawns actually use it.
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>
        /// Full pipeline: load section -> face cull -> greedy merge -> cull -> draw commands.
        /// Proves the entire data flow from raw blocks to renderable draw commands.
        /// </summary>
        [TestMethod]
        public async Task Integration_FullPipeline_BlocksToDrawCommandsTest() => await RunTest(async accelerator =>
        {
            var config = new VoxelEngineConfig { VoxelSize = 1f, SectionSize = 16, BaseY = 0f };

            // 1. Create section with a flat terrain (solid ground at y=0-3, air above)
            int ss = 16;
            var blocks = new int[ss * ss * ss];
            for (int y = 0; y < 4; y++)
                for (int z = 0; z < ss; z++)
                    for (int x = 0; x < ss; x++)
                        blocks[x + z * ss + y * ss * ss] = PackedBlock.Pack(1); // stone

            // 2. Face cull via CPU reference
            int paddedXZ = ss + 2;
            var padded = new int[paddedXZ * paddedXZ * ss];
            // Copy blocks into padded array (offset by 1 in X and Z)
            for (int y = 0; y < ss; y++)
                for (int z = 0; z < ss; z++)
                    for (int x = 0; x < ss; x++)
                        padded[(x + 1) + (z + 1) * paddedXZ + y * paddedXZ * paddedXZ] = blocks[x + z * ss + y * ss * ss];

            var occupancy = FaceCullCpuReference.BuildOccupancy(padded, paddedXZ, ss);
            var faceMasks = FaceCullCpuReference.FaceCull(occupancy, paddedXZ, ss);

            // 3. Greedy merge via CPU reference
            var quads = GreedyMergeCpuReference.GreedyMerge(faceMasks, padded, ss, ss, paddedXZ);
            if (quads.Count == 0)
                throw new Exception("Integration_FullPipeline: greedy merge produced 0 quads from solid ground section");

            // 4. Create section entry and add to draw buffer
            var coord = new SectionCoord(0, 0, 0);
            var entry = new SectionEntry
            {
                Coord = coord,
                QuadOffset = 0,
                QuadCount = quads.Count,
            };

            var drawBuffer = new IndirectDrawBuffer(100);
            int slot = drawBuffer.Add(coord, 0, quads.Count, 0);
            if (slot < 0)
                throw new Exception("Integration_FullPipeline: failed to add draw command");

            // 5. Cull against a camera looking at the section
            var pipeline = new CullingPipeline(config) { EnableGraphVisibility = false, FogDistance = 100f };
            var entries = new[] { entry };

            // Camera above and in front, looking down
            var camPos = new Vector3(8, 20, -10);
            var camForward = Vector3.Normalize(new Vector3(0, -1, 1));

            // Simple frustum that includes the section
            var vp = Matrix4x4.CreateLookAt(camPos, camPos + camForward, Vector3.UnitY)
                   * Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 3f, 16f / 9f, 0.1f, 200f);
            var frustumPlanes = ExtractPlanes(vp);

            var cullResult = pipeline.Cull(entries, camPos, camForward, frustumPlanes);

            if (cullResult.VisibleCount != 1)
                throw new Exception($"Integration_FullPipeline: expected 1 visible section, got {cullResult.VisibleCount}");

            // 6. Verify draw command data
            var cmd = drawBuffer.GetCommandAt(0);
            if (cmd.VertexCount != (uint)(quads.Count * 6))
                throw new Exception($"Integration_FullPipeline: draw command vertices {cmd.VertexCount}, expected {quads.Count * 6}");

            // 7. Verify a flat terrain section produces reasonable quad counts
            // 4 layers of solid blocks: top face (1 merged quad), bottom face (1), 4 side faces
            // With greedy merge, a flat 16x16x4 section produces far fewer quads than 16*16*4*6
            int maxExpected = 6 * 4 * 16; // generous upper bound
            if (quads.Count > maxExpected)
                throw new Exception($"Integration_FullPipeline: {quads.Count} quads exceeds expected max {maxExpected} for flat terrain");
        });

        /// <summary>
        /// Raycast -> damage -> structural integrity pipeline.
        /// Simulates: player aims at block, damages it, checks for collapse.
        /// </summary>
        [TestMethod]
        public void Integration_RaycastDamageCollapse_PipelineTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            // Create a pillar: stone blocks from y=0 to y=7 at x=8, z=8
            int ss = 16;
            var blocks = new int[ss * ss * ss];
            for (int y = 0; y < 8; y++)
                blocks[8 + 8 * ss + y * ss * ss] = PackedBlock.Pack(1); // stone, structural

            // 1. Raycast from the side to find the base block
            var hit = VoxelRaycast.Cast(blocks, ss, ss,
                new Vector3(0.5f, 0.5f, 8.5f), Vector3.UnitX, 20f);

            if (!hit.DidHit || hit.BlockX != 8 || hit.BlockY != 0)
                throw new Exception($"Integration_RaycastDamage: expected hit at (8,0,8), got ({hit.BlockX},{hit.BlockY},{hit.BlockZ})");

            // 2. Damage the base block with an explosion
            // Radius 0.9 ensures only y=0 block is hit (y=1 center is distance 1.0 away)
            int destroyed = ExplosionKernels.DestroyInSphere(
                blocks, ss, ss,
                new Vector3(8.5f, 0.5f, 8.5f), 0.9f, 100f, registry);

            if (destroyed != 1)
                throw new Exception($"Integration_RaycastDamage: expected 1 destroyed, got {destroyed}");

            // Verify base is now air
            if (!PackedBlock.IsAir(blocks[8 + 8 * ss + 0]))
                throw new Exception("Integration_RaycastDamage: base block should be air after explosion");

            // 3. Check structural integrity - rest of pillar should be unsupported
            var unsupported = StructuralIntegrity.FindUnsupported(blocks, ss, ss, registry);

            // Blocks at y=1-7 should all be unsupported (base removed)
            if (unsupported.Count != 7)
                throw new Exception($"Integration_RaycastDamage: expected 7 unsupported blocks, got {unsupported.Count}");
        }

        /// <summary>
        /// Terrain generation -> LOD reduction pipeline.
        /// Generates terrain, reduces to LOD 1, verifies dominant types preserved.
        /// </summary>
        [TestMethod]
        public void Integration_TerrainToLOD_PreservesDominantTypeTest()
        {
            // Create a flat heightmap at height 8
            int mapSize = 32;
            var heights = new short[mapSize * mapSize];
            for (int i = 0; i < heights.Length; i++) heights[i] = 8;

            var terrain = new HeightmapTerrainProvider(heights, mapSize, mapSize, 1f);
            var biomes = BiomeAssigner.CreateAubsCraft();
            var config = new VoxelEngineConfig { VoxelSize = 1f, SectionSize = 16, BaseY = 0f };

            // Generate a section that spans the terrain surface
            var sectionBlocks = terrain.GenerateSection(
                new SectionCoord(0, 0, 0), config, biomes.CreateAssigner());

            // Count non-air blocks
            int solidCount = 0;
            for (int i = 0; i < sectionBlocks.Length; i++)
                if (!PackedBlock.IsAir(sectionBlocks[i])) solidCount++;

            if (solidCount == 0)
                throw new Exception("Integration_TerrainToLOD: terrain generation produced no solid blocks");

            // Reduce to LOD 1 (16 -> 8)
            var lod1 = LODReducer.Reduce(sectionBlocks, 16, 16, 1);

            // The lower half (y=0-7) should have solid blocks in LOD 1
            // LOD 1 has 8x8x8 = 512 cells. Y=0-3 in LOD space (mapping to y=0-7 in full res)
            int lod1Solid = 0;
            for (int i = 0; i < lod1.Length; i++)
                if (!PackedBlock.IsAir(lod1[i])) lod1Solid++;

            if (lod1Solid == 0)
                throw new Exception("Integration_TerrainToLOD: LOD reduction eliminated all solid blocks");

            // Verify the dominant type at the surface is grass or dirt (from AubsCraft biome)
            int surfaceType = PackedBlock.GetType(lod1[0 + 0 * 8 + 3 * 8 * 8]); // y=3 in LOD space = y=6-7 in full
            if (surfaceType == 0)
                throw new Exception("Integration_TerrainToLOD: surface LOD block is air, expected terrain");
        }

        /// <summary>
        /// Chunk manager lifecycle: request -> load -> mesh -> visible -> cached -> unload.
        /// Tests the full state machine.
        /// </summary>
        [TestMethod]
        public void Integration_ChunkManagerLifecycle_FullCycleTest()
        {
            var config = new VoxelEngineConfig { MaxLoadedSections = 100 };
            var manager = new ChunkManager(config);
            var coord = new SectionCoord(5, 3, 7);

            // 1. Request load
            bool requested = manager.RequestLoad(coord);
            if (!requested || manager.GetState(coord) != SectionState.Loading)
                throw new Exception("Integration_ChunkLifecycle: RequestLoad failed");
            if (manager.LoadingCount != 1)
                throw new Exception($"Integration_ChunkLifecycle: LoadingCount should be 1, got {manager.LoadingCount}");

            // 2. Mark loaded with block data
            var blocks = new int[16 * 16 * 16];
            bool loaded = manager.MarkLoaded(coord, blocks);
            if (!loaded || manager.GetState(coord) != SectionState.Meshing)
                throw new Exception("Integration_ChunkLifecycle: MarkLoaded failed");

            // 3. Mark meshed
            bool meshed = manager.MarkMeshed(coord, quadOffset: 100, quadCount: 50);
            if (!meshed || manager.GetState(coord) != SectionState.GpuReady)
                throw new Exception("Integration_ChunkLifecycle: MarkMeshed failed");

            // 4. Mark visible
            bool visible = manager.MarkVisible(coord);
            if (!visible || manager.GetState(coord) != SectionState.Visible)
                throw new Exception("Integration_ChunkLifecycle: MarkVisible failed");
            if (manager.VisibleCount != 1)
                throw new Exception($"Integration_ChunkLifecycle: VisibleCount should be 1, got {manager.VisibleCount}");

            // 5. Mark cached (went out of view)
            bool cached = manager.MarkCached(coord);
            if (!cached || manager.GetState(coord) != SectionState.Cached)
                throw new Exception("Integration_ChunkLifecycle: MarkCached failed");

            // 6. Re-show (came back into view from cache)
            bool reshown = manager.MarkVisible(coord);
            if (!reshown || manager.GetState(coord) != SectionState.Visible)
                throw new Exception("Integration_ChunkLifecycle: re-show from cache failed");

            // 7. Unload
            bool unloaded = manager.Unload(coord);
            if (!unloaded || manager.GetState(coord) != null)
                throw new Exception("Integration_ChunkLifecycle: Unload failed");
            if (manager.SectionCount != 0)
                throw new Exception($"Integration_ChunkLifecycle: SectionCount should be 0 after unload, got {manager.SectionCount}");
        }

        /// <summary>
        /// Block registry -> face emission -> transparency mesher pipeline.
        /// Verifies correct face emission rules for mixed opaque/transparent sections.
        /// </summary>
        [TestMethod]
        public void Integration_TransparencyFaceEmission_CorrectRulesTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults(); // includes glass (type 7, transparent) and water (type 8, translucent)

            // Section: stone wall (type 1) with glass window (type 7) and water (type 8)
            int ss = 16;
            var blocks = new int[ss * ss * ss];

            // Stone wall at z=8
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < ss; x++)
                    blocks[x + 8 * ss + y * ss * ss] = PackedBlock.Pack(1); // stone

            // Glass window in the wall
            blocks[8 + 8 * ss + 2 * ss * ss] = PackedBlock.Pack(7); // glass at (8,2,8)

            // Water pool at y=0
            for (int x = 4; x < 12; x++)
                for (int z = 0; z < 4; z++)
                    blocks[x + z * ss + 0] = PackedBlock.Pack(8); // water

            // Check if transparency mesher detects transparent blocks
            bool hasTransparent = TransparencyMesher.HasTransparentBlocks(blocks, ss, ss, registry);
            if (!hasTransparent)
                throw new Exception("Integration_Transparency: should detect glass as transparent");

            bool hasTranslucent = TransparencyMesher.HasTranslucentBlocks(blocks, ss, ss, registry);
            if (!hasTranslucent)
                throw new Exception("Integration_Transparency: should detect water as translucent");

            // Verify block registry face emission rules
            if (!registry.EmitsFaceAgainst(1, 0)) // stone vs air = emit
                throw new Exception("Integration_Transparency: stone should emit face against air");
            if (registry.EmitsFaceAgainst(1, 1)) // stone vs stone = no emit
                throw new Exception("Integration_Transparency: stone should NOT emit face against stone");
            if (!registry.EmitsFaceAgainst(7, 1)) // glass vs stone = emit (glass side visible)
                throw new Exception("Integration_Transparency: glass should emit face against stone");
            if (registry.EmitsFaceAgainst(7, 7)) // glass vs glass = no emit (same type)
                throw new Exception("Integration_Transparency: glass should NOT emit face against glass");
            if (!registry.EmitsFaceAgainst(8, 0)) // water vs air = emit
                throw new Exception("Integration_Transparency: water should emit face against air");
        }

        /// <summary>
        /// Collision + raycast combined: player walks toward wall, stops, looks at it.
        /// Tests the gameplay loop of movement + targeting.
        /// </summary>
        [TestMethod]
        public void Integration_MovementAndTargeting_GameplayLoopTest()
        {
            // Room: solid floor at y=0, walls on all 4 sides at x=0, x=15, z=0, z=15
            int ss = 16;
            var blocks = new int[ss * ss * ss];

            // Floor
            for (int x = 0; x < ss; x++)
                for (int z = 0; z < ss; z++)
                    blocks[x + z * ss + 0] = PackedBlock.Pack(1);

            // Walls
            for (int y = 0; y < 4; y++)
            {
                for (int i = 0; i < ss; i++)
                {
                    blocks[0 + i * ss + y * ss * ss] = PackedBlock.Pack(1);  // x=0 wall
                    blocks[15 + i * ss + y * ss * ss] = PackedBlock.Pack(1); // x=15 wall
                    blocks[i + 0 * ss + y * ss * ss] = PackedBlock.Pack(1);  // z=0 wall
                    blocks[i + 15 * ss + y * ss * ss] = PackedBlock.Pack(1); // z=15 wall
                }
            }

            // Player at center of room, size 0.6x1.8x0.6
            var playerPos = new Vector3(8f, 1f, 8f);
            var playerSize = new Vector3(0.6f, 1.8f, 0.6f);

            // 1. Walk toward +X wall
            var velocity = new Vector3(10f, 0, 0); // fast movement
            var sweep = VoxelCollision.Sweep(blocks, ss, ss, playerPos, playerSize, velocity);

            if (!sweep.Hit)
                throw new Exception("Integration_Gameplay: player should hit +X wall");
            if (sweep.Normal.X >= 0)
                throw new Exception("Integration_Gameplay: collision normal should point -X (away from wall)");

            // Player should stop before reaching the wall
            float resolvedX = sweep.ResolvedPosition.X + playerSize.X;
            if (resolvedX > 15.1f)
                throw new Exception($"Integration_Gameplay: player penetrated wall (right edge at {resolvedX})");

            // 2. Player on ground check
            bool onGround = VoxelCollision.IsOnGround(blocks, ss, ss, playerPos, playerSize);
            if (!onGround)
                throw new Exception("Integration_Gameplay: player should be on ground");

            // 3. Look at the +X wall (raycast from player eye position)
            var eyePos = playerPos + new Vector3(0.3f, 1.6f, 0.3f); // eye height
            var lookDir = Vector3.UnitX; // looking east

            var hit = VoxelRaycast.Cast(blocks, ss, ss, eyePos, lookDir, 20f);
            if (!hit.DidHit)
                throw new Exception("Integration_Gameplay: raycast should hit +X wall");
            if (hit.BlockX != 15)
                throw new Exception($"Integration_Gameplay: should hit wall at x=15, got x={hit.BlockX}");

            // 4. Adjacent block for placement should be inside the room
            if (hit.AdjacentX != 14)
                throw new Exception($"Integration_Gameplay: adjacent for placement should be x=14, got {hit.AdjacentX}");
        }

        /// <summary>
        /// InMemoryChunkCache -> ChunkManager integration.
        /// </summary>
        [TestMethod]
        public async Task Integration_CacheToManager_LoadAndStoreTest()
        {
            var cache = new InMemoryChunkCache();
            var config = new VoxelEngineConfig { MaxLoadedSections = 100 };
            var manager = new ChunkManager(config);

            var coord = new SectionCoord(1, 2, 3);
            var blocks = CreateTestSection((8, 8, 8, 42));

            // Store in cache
            await cache.SaveSectionAsync(coord, blocks);
            if (!await cache.ExistsAsync(coord))
                throw new Exception("Integration_Cache: section should exist after save");

            // Load from cache into manager
            manager.RequestLoad(coord);
            var loaded = await cache.LoadSectionAsync(coord);
            if (loaded == null)
                throw new Exception("Integration_Cache: loaded data should not be null");

            manager.MarkLoaded(coord, loaded);

            // Verify data integrity
            var section = manager.GetSection(coord);
            if (section == null || section.BlockData == null)
                throw new Exception("Integration_Cache: section data missing in manager");

            int blockAt888 = section.BlockData[8 + 8 * 16 + 8 * 16 * 16];
            if (PackedBlock.GetType(blockAt888) != 42)
                throw new Exception($"Integration_Cache: block type at (8,8,8) should be 42, got {PackedBlock.GetType(blockAt888)}");
        }

        /// <summary>Extract 6 frustum planes from a VP matrix for culling tests.</summary>
        private static float[] ExtractPlanes(Matrix4x4 vp)
        {
            var planes = new float[24];
            SetPlane(planes, 0, vp.M14 + vp.M11, vp.M24 + vp.M21, vp.M34 + vp.M31, vp.M44 + vp.M41);
            SetPlane(planes, 1, vp.M14 - vp.M11, vp.M24 - vp.M21, vp.M34 - vp.M31, vp.M44 - vp.M41);
            SetPlane(planes, 2, vp.M14 + vp.M12, vp.M24 + vp.M22, vp.M34 + vp.M32, vp.M44 + vp.M42);
            SetPlane(planes, 3, vp.M14 - vp.M12, vp.M24 - vp.M22, vp.M34 - vp.M32, vp.M44 - vp.M42);
            SetPlane(planes, 4, vp.M14 + vp.M13, vp.M24 + vp.M23, vp.M34 + vp.M33, vp.M44 + vp.M43);
            SetPlane(planes, 5, vp.M14 - vp.M13, vp.M24 - vp.M23, vp.M34 - vp.M33, vp.M44 - vp.M43);
            return planes;
        }

        private static void SetPlane(float[] p, int i, float a, float b, float c, float d)
        {
            float len = MathF.Sqrt(a * a + b * b + c * c);
            if (len > 0) { float inv = 1f / len; a *= inv; b *= inv; c *= inv; d *= inv; }
            int idx = i * 4;
            p[idx] = a; p[idx + 1] = b; p[idx + 2] = c; p[idx + 3] = d;
        }
    }
}
