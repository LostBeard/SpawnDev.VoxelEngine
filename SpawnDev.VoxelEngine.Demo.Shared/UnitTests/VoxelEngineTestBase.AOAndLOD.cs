using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.LOD;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // AO and LOD tests: per-vertex ambient occlusion + LOD reduction + LOD selection.
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>
        /// AO: isolated block in air has AO = 0 on all faces (fully lit).
        /// </summary>
        [TestMethod]
        public void AO_IsolatedBlock_FullyLitTest()
        {
            var section = CreateTestSection((8, 8, 8, 1));

            for (int face = 0; face < 6; face++)
            {
                int ao = AmbientOcclusion.ComputeQuadAO(section, 16, 16, 8, 8, 8, face);
                // All 4 corners should be 0 (no neighbors blocking light)
                int c0 = ao & 3, c1 = (ao >> 2) & 3, c2 = (ao >> 4) & 3, c3 = (ao >> 6) & 3;
                if (c0 != 0 || c1 != 0 || c2 != 0 || c3 != 0)
                    throw new Exception($"AO_IsolatedBlock face {face}: expected all 0, got {c0},{c1},{c2},{c3}");
            }
        }

        /// <summary>
        /// AO: block in a corner (3 solid neighbors) has AO = 3 on the corner vertex.
        /// Place blocks at (0,0,0), (1,0,0), (0,0,1) - the +Y face of (0,0,0) has
        /// corner at (+X,+Z) with two side neighbors and the diagonal.
        /// </summary>
        [TestMethod]
        public void AO_CornerBlock_MaxOcclusionTest()
        {
            // Fill a 2x1x2 L-shape: blocks at (0,0,0), (1,0,0), (0,0,1), (1,0,1) - solid floor
            // Plus block at (0,1,0) - the test block
            // The +Y face of (0,0,0) should have high AO on corners where neighbors exist above
            int size = 16;
            var section = new int[size * size * size];
            // Ground floor
            section[0 + 0 * size + 0] = PackedBlock.Pack(1); // (0,0,0)
            section[1 + 0 * size + 0] = PackedBlock.Pack(1); // (1,0,0)
            section[0 + 1 * size + 0] = PackedBlock.Pack(1); // (0,0,1)
            section[1 + 1 * size + 0] = PackedBlock.Pack(1); // (1,0,1)
            // Walls above
            section[1 + 0 * size + 1 * size * size] = PackedBlock.Pack(1); // (1,1,0) - +X wall
            section[0 + 1 * size + 1 * size * size] = PackedBlock.Pack(1); // (0,1,1) - +Z wall
            section[1 + 1 * size + 1 * size * size] = PackedBlock.Pack(1); // (1,1,1) - corner wall

            // Check +Y face of block (0,0,0) at y=0
            // The corner at +X,+Z should be fully occluded (3 neighbors: +X wall, +Z wall, corner)
            int ao = AmbientOcclusion.ComputeQuadAO(section, size, size, 0, 0, 0, VoxelMeshConstants.FacePosY);
            int c2 = (ao >> 4) & 3; // corner 2 is +X,+Z
            if (c2 != 3)
                throw new Exception($"AO_CornerBlock: +Y face corner(+X,+Z) expected AO=3, got {c2}");
        }

        /// <summary>
        /// AO: quad flip determination works correctly.
        /// When diagonal AO values are unequal, the quad should flip.
        /// </summary>
        [TestMethod]
        public void AO_QuadFlip_CorrectTest()
        {
            // c0=3, c1=0, c2=3, c3=0 -> c0+c2=6 > c1+c3=0 -> should flip
            int packed1 = 3 | (0 << 2) | (3 << 4) | (0 << 6);
            if (!AmbientOcclusion.ShouldFlipQuad(packed1))
                throw new Exception("AO_QuadFlip: expected flip for (3,0,3,0)");

            // c0=0, c1=3, c2=0, c3=3 -> c0+c2=0 < c1+c3=6 -> should NOT flip
            int packed2 = 0 | (3 << 2) | (0 << 4) | (3 << 6);
            if (AmbientOcclusion.ShouldFlipQuad(packed2))
                throw new Exception("AO_QuadFlip: should NOT flip for (0,3,0,3)");

            // c0=1, c1=1, c2=1, c3=1 -> equal -> should NOT flip
            int packed3 = 1 | (1 << 2) | (1 << 4) | (1 << 6);
            if (AmbientOcclusion.ShouldFlipQuad(packed3))
                throw new Exception("AO_QuadFlip: should NOT flip for equal (1,1,1,1)");
        }

        /// <summary>
        /// AO: light multiplier values are correct per level.
        /// </summary>
        [TestMethod]
        public void AO_LightMultiplier_CorrectValuesTest()
        {
            if (MathF.Abs(AmbientOcclusion.AOToLightMultiplier(0) - 1.0f) > 0.001f)
                throw new Exception("AO multiplier: level 0 should be 1.0");
            if (MathF.Abs(AmbientOcclusion.AOToLightMultiplier(1) - 0.8f) > 0.001f)
                throw new Exception("AO multiplier: level 1 should be 0.8");
            if (MathF.Abs(AmbientOcclusion.AOToLightMultiplier(2) - 0.6f) > 0.001f)
                throw new Exception("AO multiplier: level 2 should be 0.6");
            if (MathF.Abs(AmbientOcclusion.AOToLightMultiplier(3) - 0.4f) > 0.001f)
                throw new Exception("AO multiplier: level 3 should be 0.4");
        }

        /// <summary>
        /// LOD: reduction of a uniform section preserves the block type.
        /// </summary>
        [TestMethod]
        public void LOD_UniformSection_PreservesTypeTest()
        {
            // Fill entire section with stone (type 1)
            int size = 16;
            var section = new int[size * size * size];
            for (int i = 0; i < section.Length; i++)
                section[i] = PackedBlock.Pack(1);

            // LOD 1: 16 -> 8
            var lod1 = LODReducer.Reduce(section, size, size, 1);
            if (lod1.Length != 8 * 8 * 8)
                throw new Exception($"LOD_Uniform: LOD1 size {lod1.Length}, expected {8 * 8 * 8}");

            for (int i = 0; i < lod1.Length; i++)
            {
                int type = PackedBlock.GetType(lod1[i]);
                if (type != 1)
                    throw new Exception($"LOD_Uniform: LOD1[{i}] type {type}, expected 1");
            }

            // LOD 2: 16 -> 4
            var lod2 = LODReducer.Reduce(section, size, size, 2);
            if (lod2.Length != 4 * 4 * 4)
                throw new Exception($"LOD_Uniform: LOD2 size {lod2.Length}, expected {4 * 4 * 4}");
        }

        /// <summary>
        /// LOD: reduction picks dominant type (most common non-air).
        /// </summary>
        [TestMethod]
        public void LOD_MixedSection_DominantTypeTest()
        {
            int size = 16;
            var section = new int[size * size * size];

            // Fill bottom 2x2x2 with: 6 stone (type 1) + 2 dirt (type 2)
            // Stone should win as dominant
            section[PackedBlock.Pack(1)] = PackedBlock.Pack(1); // dummy, fix below
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++)
                    for (int x = 0; x < 2; x++)
                    {
                        int idx = x + z * size + y * size * size;
                        section[idx] = (x == 0 && y == 0) ? PackedBlock.Pack(2) : PackedBlock.Pack(1);
                    }

            // LOD 1: the 2x2x2 group at (0,0,0) should reduce to stone (6 stone vs 1 dirt + 1 stone at 0,0,0)
            var lod1 = LODReducer.Reduce(section, size, size, 1);
            int resultType = PackedBlock.GetType(lod1[0]);
            if (resultType != 1)
                throw new Exception($"LOD_Mixed: dominant type {resultType}, expected 1 (stone)");
        }

        /// <summary>
        /// LOD: air-only groups reduce to air.
        /// </summary>
        [TestMethod]
        public void LOD_EmptySection_AllAirTest()
        {
            int size = 16;
            var section = new int[size * size * size]; // all air

            var lod1 = LODReducer.Reduce(section, size, size, 1);
            for (int i = 0; i < lod1.Length; i++)
            {
                if (lod1[i] != 0)
                    throw new Exception($"LOD_Empty: LOD1[{i}] = {lod1[i]}, expected 0 (air)");
            }
        }

        /// <summary>
        /// LOD Selector: distance-based level selection with hysteresis.
        /// </summary>
        [TestMethod]
        public void LODSelector_DistanceLevels_CorrectTest()
        {
            var selector = new LODSelector
            {
                LodDistances = new[] { 8f, 16f, 32f },
                HysteresisFactor = 0.2f,
            };

            // Close: LOD 0
            int lod = selector.SelectLOD(4f, 0);
            if (lod != 0)
                throw new Exception($"LODSelector: dist=4 expected LOD 0, got {lod}");

            // Medium: LOD 1
            lod = selector.SelectLOD(12f, 0);
            if (lod != 1)
                throw new Exception($"LODSelector: dist=12 expected LOD 1, got {lod}");

            // Far: LOD 2
            lod = selector.SelectLOD(20f, 0);
            if (lod != 2)
                throw new Exception($"LODSelector: dist=20 expected LOD 2, got {lod}");

            // Very far: LOD 3
            lod = selector.SelectLOD(40f, 0);
            if (lod != 3)
                throw new Exception($"LODSelector: dist=40 expected LOD 3, got {lod}");
        }

        /// <summary>
        /// LOD Selector: hysteresis prevents oscillation at threshold boundaries.
        /// </summary>
        [TestMethod]
        public void LODSelector_Hysteresis_NoOscillationTest()
        {
            var selector = new LODSelector
            {
                LodDistances = new[] { 8f, 16f, 32f },
                HysteresisFactor = 0.2f, // 20% hysteresis
            };

            // At LOD 0, threshold to LOD 1 is at 8 * 1.2 = 9.6 (need to go past to switch)
            int lod = selector.SelectLOD(8.5f, currentLod: 0);
            if (lod != 0) // should stay at 0 (not past 9.6)
                throw new Exception($"LODSelector hysteresis: dist=8.5 from LOD0 should stay 0, got {lod}");

            lod = selector.SelectLOD(10f, currentLod: 0);
            if (lod != 1) // past 9.6, should switch to 1
                throw new Exception($"LODSelector hysteresis: dist=10 from LOD0 should switch to 1, got {lod}");

            // At LOD 1, threshold to go back to LOD 0 is at 8 * 0.8 = 6.4
            lod = selector.SelectLOD(7f, currentLod: 1);
            if (lod != 1) // should stay at 1 (not below 6.4)
                throw new Exception($"LODSelector hysteresis: dist=7 from LOD1 should stay 1, got {lod}");

            lod = selector.SelectLOD(5f, currentLod: 1);
            if (lod != 0) // below 6.4, should switch back to 0
                throw new Exception($"LODSelector hysteresis: dist=5 from LOD1 should switch to 0, got {lod}");
        }
    }
}
