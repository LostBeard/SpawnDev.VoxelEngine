using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.SDF;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // SdfChunk data structure tests (CPU, no GPU required - runs on all backends).
    public abstract partial class VoxelEngineTestBase
    {
        [TestMethod]
        public void SdfChunk_DefaultIsFullyAir_Test()
        {
            var chunk = new SdfChunk();
            if (!chunk.IsUniform)
                throw new Exception("Default chunk must be uniform");
            if (!chunk.IsFullyAir)
                throw new Exception("Default chunk must be fully air (UniformValue = short.MinValue)");
            if (chunk.IsFullySolid)
                throw new Exception("Default chunk must not be fully solid");
            if (chunk.HasSurface)
                throw new Exception("Uniform air chunk has no surface to mesh");
        }

        [TestMethod]
        public void SdfChunk_CreateSolid_IsFullySolidTest()
        {
            var chunk = SdfChunk.CreateSolid();
            if (!chunk.IsFullySolid)
                throw new Exception("CreateSolid must produce fully solid chunk");
            if (chunk.IsFullyAir)
                throw new Exception("Solid chunk is not air");
            if (chunk.HasSurface)
                throw new Exception("Uniform solid chunk has no surface to mesh");
        }

        [TestMethod]
        public void SdfChunk_CreateAir_IsFullyAirTest()
        {
            var chunk = SdfChunk.CreateAir();
            if (!chunk.IsFullyAir)
                throw new Exception("CreateAir must produce fully air chunk");
        }

        [TestMethod]
        public void SdfChunk_GetValueFromUniform_ReturnsUniformValueTest()
        {
            var air = SdfChunk.CreateAir();
            if (air.GetValue(0, 0, 0) != short.MinValue)
                throw new Exception($"Air chunk GetValue(0,0,0) must be short.MinValue, got {air.GetValue(0, 0, 0)}");
            if (air.GetValue(15, 16, 17) != short.MinValue)
                throw new Exception("Air chunk GetValue at arbitrary index must be short.MinValue");

            var solid = SdfChunk.CreateSolid();
            if (solid.GetValue(5, 5, 5) != short.MaxValue)
                throw new Exception($"Solid chunk GetValue must be short.MaxValue, got {solid.GetValue(5, 5, 5)}");
        }

        [TestMethod]
        public void SdfChunk_FixedPointRoundtrip_PreservesValueTest()
        {
            float[] testValues = { 0f, 1f, -1f, 0.5f, -0.5f, 100f, -100f, 127f, -128f };
            foreach (var v in testValues)
            {
                short fixedPoint = SdfChunk.ToFixedPoint(v);
                float back = SdfChunk.ToFloat(fixedPoint);
                float diff = Math.Abs(back - v);
                // Tolerance: 1 fixed-point step = 1 / 256 = ~0.004
                if (diff > 1.0f / SdfChunk.FixedPointScale)
                    throw new Exception($"Roundtrip {v} -> {fixedPoint} -> {back} (diff {diff}) exceeds fixed-point step");
            }
        }

        [TestMethod]
        public void SdfChunk_FixedPointClamp_SaturatesTest()
        {
            // Values above 127.996 should clamp to short.MaxValue
            short clampedHigh = SdfChunk.ToFixedPoint(1000f);
            if (clampedHigh != short.MaxValue)
                throw new Exception($"ToFixedPoint(1000) should clamp to short.MaxValue, got {clampedHigh}");

            short clampedLow = SdfChunk.ToFixedPoint(-1000f);
            if (clampedLow != short.MinValue)
                throw new Exception($"ToFixedPoint(-1000) should clamp to short.MinValue, got {clampedLow}");
        }

        [TestMethod]
        public void SdfChunk_SetValueOnUniform_AllocatesStorageTest()
        {
            var chunk = SdfChunk.CreateAir();
            if (!chunk.IsUniform)
                throw new Exception("Precondition: chunk starts uniform");

            // Setting to the same uniform value is a no-op
            chunk.SetValue(0, 0, 0, short.MinValue);
            if (!chunk.IsUniform)
                throw new Exception("SetValue to same uniform must not allocate storage");

            // Setting to a different value must allocate and fill with the uniform default
            chunk.SetValue(5, 6, 7, 0);
            if (chunk.IsUniform)
                throw new Exception("SetValue with different value must allocate storage");
            if (chunk.GetValue(5, 6, 7) != 0)
                throw new Exception($"SetValue failed: expected 0 at (5,6,7), got {chunk.GetValue(5, 6, 7)}");

            // All other cells must still hold the previous uniform value
            if (chunk.GetValue(0, 0, 0) != short.MinValue)
                throw new Exception($"After SetValue, other cells must hold original uniform value, got {chunk.GetValue(0, 0, 0)}");
            if (chunk.GetValue(10, 10, 10) != short.MinValue)
                throw new Exception("After SetValue, distant cells must hold original uniform value");
        }

        [TestMethod]
        public void SdfChunk_SetDistance_MatchesFixedPointTest()
        {
            var chunk = new SdfChunk(16);
            chunk.SetDistance(3, 4, 5, 1.5f);
            float back = chunk.GetDistance(3, 4, 5);
            float diff = Math.Abs(back - 1.5f);
            if (diff > 1.0f / SdfChunk.FixedPointScale)
                throw new Exception($"SetDistance/GetDistance roundtrip failed: set 1.5, got {back}");
        }

        [TestMethod]
        public void SdfChunk_TryCompact_UniformDataCompactsTest()
        {
            var chunk = new SdfChunk(8);
            // Force allocation via any SetValue
            chunk.SetValue(0, 0, 0, 42);
            chunk.SetValue(1, 1, 1, 42);
            // Fill entire buffer with 42 - requires direct access
            if (chunk.Values == null)
                throw new Exception("Precondition: Values must be allocated after SetValue");
            Array.Fill(chunk.Values, (short)42);

            if (!chunk.TryCompact())
                throw new Exception("TryCompact must succeed when all values are equal");
            if (!chunk.IsUniform)
                throw new Exception("After TryCompact, chunk must be uniform");
            if (chunk.UniformValue != 42)
                throw new Exception($"After TryCompact, UniformValue must be 42, got {chunk.UniformValue}");
        }

        [TestMethod]
        public void SdfChunk_TryCompact_NonUniformDoesNotCompactTest()
        {
            var chunk = new SdfChunk(4);
            chunk.SetValue(0, 0, 0, 10);
            chunk.SetValue(1, 0, 0, 20);

            if (chunk.TryCompact())
                throw new Exception("TryCompact must fail when values differ");
            if (chunk.IsUniform)
                throw new Exception("Chunk with differing values must not become uniform");
        }

        [TestMethod]
        public void SdfChunk_VoxelCount_MatchesSizeCubedTest()
        {
            var chunk8 = new SdfChunk(8);
            if (chunk8.VoxelCount != 8 * 8 * 8)
                throw new Exception($"VoxelCount for size 8 must be 512, got {chunk8.VoxelCount}");
            var chunk32 = new SdfChunk(32);
            if (chunk32.VoxelCount != 32 * 32 * 32)
                throw new Exception($"VoxelCount for size 32 must be 32768, got {chunk32.VoxelCount}");
        }

        [TestMethod]
        public void SdfChunk_HasSurface_TrueForMixedValuesTest()
        {
            var chunk = new SdfChunk(4);
            chunk.SetValue(0, 0, 0, 100);  // solid
            chunk.SetValue(1, 0, 0, -100); // air
            if (!chunk.HasSurface)
                throw new Exception("Chunk with solid+air cells must have a surface");
        }
    }
}
