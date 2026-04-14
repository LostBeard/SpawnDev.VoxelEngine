using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Culling;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Frustum cull debug + fixed tests
    public partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task FrustumCull_ManualPlanes_KnownGood() => await RunTest(async accelerator =>
        {
            // Skip the matrix extraction entirely - define planes manually for a known frustum
            // Camera at origin, looking down +Z, 90-degree FOV, near=1, far=100
            // This tests the GPU kernel independently of plane extraction
            var planes = new float[24];

            // Left plane: x + z >= 0 (45 degrees left of center)
            SetPlaneNormalized(planes, 0, 1, 0, 1, 0);
            // Right plane: -x + z >= 0
            SetPlaneNormalized(planes, 1, -1, 0, 1, 0);
            // Bottom plane: y + z >= 0
            SetPlaneNormalized(planes, 2, 0, 1, 1, 0);
            // Top plane: -y + z >= 0
            SetPlaneNormalized(planes, 3, 0, -1, 1, 0);
            // Near plane: z >= 1
            SetPlaneNormalized(planes, 4, 0, 0, 1, -1);
            // Far plane: z <= 100 -> -z + 100 >= 0
            SetPlaneNormalized(planes, 5, 0, 0, -1, 100);

            // Chunks in front of camera, within frustum
            var centers = new float[] {
                0, 0, 20,    // Center, 20 units away - VISIBLE
                5, 0, 20,    // Slightly right - VISIBLE
                0, 0, 50,    // Far center - VISIBLE
                0, 0, 0.5f,  // Before near plane - CULLED
                0, 0, 150,   // Beyond far plane - CULLED
                -50, 0, 20,  // Way left - CULLED (outside 45-degree FOV at distance 20)
            };
            int chunkCount = 6;
            float halfX = 4, halfY = 4, halfZ = 4;
            float fogDistSq = 200 * 200; // Large fog, not the culling factor

            // CPU reference
            var cpuVisible = FrustumCullCpuReference.CullChunks(
                centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            // GPU kernel
            var gpuVisible = await RunFrustumCullOnGpu(
                accelerator, centers, halfX, halfY, halfZ, planes, 0, 0, 0, fogDistSq);

            // Expect: chunks 0,1,2 visible (in frustum), chunks 3,4,5 culled
            // Note: chunk 3 at z=0.5 with halfZ=4 means AABB extends from z=-3.5 to z=4.5
            // which crosses the near plane at z=1, so it may be visible (AABB partially in)
            // Chunk 5 at x=-50 with halfX=4: AABB from x=-54 to x=-46, at z=20
            // Left plane: x+z>=0 -> -46+20=-26 < 0, so p-vertex test: max_x+z = -46+24 = -22 < 0 -> CULLED

            cpuVisible.Sort();
            gpuVisible.Sort();

            if (cpuVisible.Count != gpuVisible.Count)
                throw new Exception(
                    $"Visible count mismatch: CPU=[{string.Join(",", cpuVisible)}], GPU=[{string.Join(",", gpuVisible)}]");

            for (int i = 0; i < cpuVisible.Count; i++)
            {
                if (cpuVisible[i] != gpuVisible[i])
                    throw new Exception(
                        $"Visible set mismatch: CPU=[{string.Join(",", cpuVisible)}], GPU=[{string.Join(",", gpuVisible)}]");
            }

            // At minimum, chunk 0 (center, z=20) should be visible
            if (!cpuVisible.Contains(0))
                throw new Exception($"Chunk 0 at (0,0,20) should be visible. Visible: [{string.Join(",", cpuVisible)}]");

            // Chunk 5 (x=-50) should be culled
            if (cpuVisible.Contains(5))
                throw new Exception("Chunk 5 at (-50,0,20) should be culled by left plane");
        });

        [TestMethod]
        public async Task FrustumCull_MatrixExtraction_PointInFrustum() => await RunTest(async accelerator =>
        {
            // Build VP matrix and verify a known point is inside
            var viewProj = FrustumCullCpuReference.BuildTestViewProj(
                0, 0, 0,    // camera at origin
                0, 0, 1,    // looking at +Z
                1.2f,        // ~69 degree FOV
                1.0f,        // aspect ratio
                0.1f, 500f); // near/far

            var planes = FrustumCullCpuReference.ExtractFrustumPlanes(viewProj);

            // Test: is point (0, 0, 30) inside the frustum?
            // It's directly in front of the camera at distance 30
            float x = 0, y = 0, z = 30;
            bool inside = true;
            string failPlane = "";
            for (int i = 0; i < 6; i++)
            {
                float a = planes[i * 4 + 0];
                float b = planes[i * 4 + 1];
                float c = planes[i * 4 + 2];
                float d = planes[i * 4 + 3];
                float dist = a * x + b * y + c * z + d;
                string[] names = { "Left", "Right", "Bottom", "Top", "Near", "Far" };
                if (dist < 0)
                {
                    inside = false;
                    failPlane = $"{names[i]}: normal=({a:F4},{b:F4},{c:F4}) d={d:F4}, dist={dist:F4}";
                    break;
                }
            }

            if (!inside)
                throw new Exception(
                    $"Point (0,0,30) should be inside frustum but failed plane: {failPlane}. " +
                    $"Camera at origin looking +Z, FOV=1.2rad, near=0.1, far=500");

            await Task.CompletedTask;
        });

        private static void SetPlaneNormalized(float[] planes, int index, float a, float b, float c, float d)
        {
            float len = MathF.Sqrt(a * a + b * b + c * c);
            if (len > 0.0001f) { a /= len; b /= len; c /= len; d /= len; }
            planes[index * 4 + 0] = a;
            planes[index * 4 + 1] = b;
            planes[index * 4 + 2] = c;
            planes[index * 4 + 3] = d;
        }
    }
}
