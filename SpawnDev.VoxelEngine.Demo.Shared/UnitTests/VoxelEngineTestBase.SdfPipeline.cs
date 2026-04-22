using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.SDF;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // End-to-end SdfMeshPipeline integration: SDF in -> quad mesh out.
    public abstract partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task SdfPipeline_AllAirSdf_ReturnsEmptyResult() => await RunTest(async accelerator =>
        {
            int size = 8;
            var sdf = new short[size * size * size];
            Array.Fill(sdf, (short)-500);

            using var pipeline = new SdfMeshPipeline(accelerator);
            using var sdfBuffer = accelerator.Allocate1D(sdf);
            using var mesh = await pipeline.MeshSdfAsync(sdfBuffer, size, 0, 0, 0, 1f);

            if (mesh.HasMesh)
                throw new Exception($"All-air SDF must return empty mesh, got {mesh.QuadCount} quads");
            if (mesh.QuadCount != 0)
                throw new Exception($"Empty mesh must have 0 quads, got {mesh.QuadCount}");
            if (mesh.VertexCount != 0)
                throw new Exception($"Empty mesh must have 0 vertices, got {mesh.VertexCount}");
        });

        [TestMethod]
        public async Task SdfPipeline_AnalyticSphere_VerticesOnSurface() => await RunTest(async accelerator =>
        {
            // Place a sphere at the chunk center and verify every DMC vertex lies on the surface.
            int size = 10;
            float voxelSize = 1f;
            float cx = (size - 1) * 0.5f; // chunk-local center (voxel coords)
            float cy = (size - 1) * 0.5f;
            float cz = (size - 1) * 0.5f;
            float radius = 3f;

            var sdf = new short[size * size * size];
            for (int y = 0; y < size; y++)
            for (int z = 0; z < size; z++)
            for (int x = 0; x < size; x++)
            {
                float dx = x - cx;
                float dy = y - cy;
                float dz = z - cz;
                float dist = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                // Inside sphere -> positive SDF (solid), outside -> negative (air).
                float sdfValue = radius - dist;
                sdf[x + z * size + y * size * size] = SdfChunk.ToFixedPoint(sdfValue);
            }

            using var pipeline = new SdfMeshPipeline(accelerator);
            using var sdfBuffer = accelerator.Allocate1D(sdf);
            using var mesh = await pipeline.MeshSdfAsync(sdfBuffer, size, 0, 0, 0, voxelSize);

            if (!mesh.HasMesh)
                throw new Exception("Analytic sphere must produce a non-empty mesh");
            if (mesh.VertexPositions == null)
                throw new Exception("Sphere mesh must have vertex positions");

            var positions = await mesh.VertexPositions.CopyToHostAsync();

            // Every vertex must lie within one voxel of the sphere surface.
            // (DMC mass point is on edge crossings, which are within a voxel of the true surface.)
            float maxDeviation = 0f;
            for (int v = 0; v < mesh.VertexCount; v++)
            {
                float vx = positions[v * 3 + 0];
                float vy = positions[v * 3 + 1];
                float vz = positions[v * 3 + 2];
                float dist = MathF.Sqrt((vx - cx) * (vx - cx) + (vy - cy) * (vy - cy) + (vz - cz) * (vz - cz));
                float deviation = Math.Abs(dist - radius);
                if (deviation > maxDeviation) maxDeviation = deviation;
                if (deviation > voxelSize)
                    throw new Exception($"Vertex {v} at ({vx},{vy},{vz}) is {deviation} from sphere surface (> {voxelSize})");
            }
        });

        [TestMethod]
        public async Task SdfPipeline_AnalyticSphere_QuadCountIsReasonable() => await RunTest(async accelerator =>
        {
            // For a sphere of radius 3 in a 10^3 chunk, surface area is ~4*pi*9 = ~113 voxels.
            // DMC produces roughly one quad per voxel of surface area, so quads are in range [50, 250].
            int size = 10;
            float voxelSize = 1f;
            float cx = (size - 1) * 0.5f, cy = (size - 1) * 0.5f, cz = (size - 1) * 0.5f;
            float radius = 3f;

            var sdf = new short[size * size * size];
            for (int y = 0; y < size; y++)
            for (int z = 0; z < size; z++)
            for (int x = 0; x < size; x++)
            {
                float dx = x - cx, dy = y - cy, dz = z - cz;
                float dist = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                sdf[x + z * size + y * size * size] = SdfChunk.ToFixedPoint(radius - dist);
            }

            using var pipeline = new SdfMeshPipeline(accelerator);
            using var sdfBuffer = accelerator.Allocate1D(sdf);
            using var mesh = await pipeline.MeshSdfAsync(sdfBuffer, size, 0, 0, 0, voxelSize);

            if (mesh.QuadCount < 50 || mesh.QuadCount > 250)
                throw new Exception($"Sphere r=3 in 10^3 chunk expected 50-250 quads, got {mesh.QuadCount}");
        });

        [TestMethod]
        public async Task SdfPipeline_ChunkWorldOffset_VerticesInWorldSpace() => await RunTest(async accelerator =>
        {
            // Move the chunk origin to (100, 200, 300) and verify vertices come out offset accordingly.
            int size = 6;
            float voxelSize = 1f;
            float worldOffsetX = 100f, worldOffsetY = 200f, worldOffsetZ = 300f;

            // Horizontal boundary in chunk-local space at y=3.
            var sdf = new short[size * size * size];
            for (int y = 0; y < size; y++)
            for (int z = 0; z < size; z++)
            for (int x = 0; x < size; x++)
                sdf[x + z * size + y * size * size] = (y >= 3) ? (short)500 : (short)-500;

            using var pipeline = new SdfMeshPipeline(accelerator);
            using var sdfBuffer = accelerator.Allocate1D(sdf);
            using var mesh = await pipeline.MeshSdfAsync(sdfBuffer, size,
                worldOffsetX, worldOffsetY, worldOffsetZ, voxelSize);

            if (!mesh.HasMesh || mesh.VertexPositions == null)
                throw new Exception("Offset chunk must produce a mesh");

            var positions = await mesh.VertexPositions.CopyToHostAsync();
            // Each vertex Y (world) = worldOffsetY + 2.5 (local Y = 2.5 for boundary at y=3)
            float expectedY = worldOffsetY + 2.5f;
            for (int v = 0; v < mesh.VertexCount; v++)
            {
                float vy = positions[v * 3 + 1];
                if (Math.Abs(vy - expectedY) > 0.01f)
                    throw new Exception($"Vertex {v} Y={vy}, expected {expectedY} (world offset applied)");

                float vx = positions[v * 3 + 0];
                float vz = positions[v * 3 + 2];
                // All vertices must be within the chunk's world bounds on X and Z.
                if (vx < worldOffsetX || vx > worldOffsetX + size * voxelSize)
                    throw new Exception($"Vertex {v} X={vx} out of chunk world X range");
                if (vz < worldOffsetZ || vz > worldOffsetZ + size * voxelSize)
                    throw new Exception($"Vertex {v} Z={vz} out of chunk world Z range");
            }
        });

        [TestMethod]
        public async Task SdfPipeline_GenerateAndMesh_EndToEndProducesMesh() => await RunTest(async accelerator =>
        {
            // Sweep Y values around the base terrain height (64) so the test
            // isn't fragile to noise variation for any single seed. At least one
            // chunk along this column MUST straddle the surface.
            int size = 8;
            using var pipeline = new SdfMeshPipeline(accelerator);
            int sampledY = 0;
            int meshesFound = 0;
            for (int y = 0; y < 120; y += size)
            {
                var (sdf, mesh) = await pipeline.GenerateAndMeshAsync(
                    chunkSize: size,
                    chunkWorldX: 0f, chunkWorldY: y, chunkWorldZ: 0f,
                    voxelSize: 1f,
                    seed: 12345);
                try
                {
                    sampledY++;
                    if (mesh.HasMesh) meshesFound++;
                }
                finally
                {
                    sdf.Dispose();
                    mesh.Dispose();
                }
            }
            if (meshesFound == 0)
                throw new Exception($"GenerateAndMeshAsync produced no mesh across {sampledY} chunk positions spanning Y=0..120");
        });
    }
}
