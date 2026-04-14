using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // PackedBlock and PackedQuad format tests
    public partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task PackedBlock_PackUnpack_Roundtrip() => await RunTest(async accelerator =>
        {
            // Test all edge values
            for (int type = 0; type < 4096; type += 137) // sample across range
            {
                for (int damage = 0; damage <= 15; damage++)
                {
                    int packed = PackedBlock.Pack(type, damage);
                    int gotType = PackedBlock.GetType(packed);
                    int gotDamage = PackedBlock.GetDamage(packed);

                    if (gotType != type)
                        throw new Exception($"Type roundtrip failed: packed({type},{damage}) -> type={gotType}");
                    if (gotDamage != damage)
                        throw new Exception($"Damage roundtrip failed: packed({type},{damage}) -> damage={gotDamage}");
                }
            }

            // Edge: max values
            int maxPacked = PackedBlock.Pack(4095, 15);
            if (PackedBlock.GetType(maxPacked) != 4095 || PackedBlock.GetDamage(maxPacked) != 15)
                throw new Exception("Max value roundtrip failed");

            // Air check
            if (!PackedBlock.IsAir(0)) throw new Exception("0 should be air");
            if (!PackedBlock.IsAir(PackedBlock.Pack(0, 5))) throw new Exception("type 0 with damage should be air");
            if (PackedBlock.IsAir(PackedBlock.Pack(1, 0))) throw new Exception("type 1 should not be air");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task PackedQuad_SectionLocal_Roundtrip() => await RunTest(async accelerator =>
        {
            // Test section-local coordinates (0-15)
            for (int x = 0; x < 16; x++)
            {
                for (int y = 0; y < 16; y++)
                {
                    for (int z = 0; z < 16; z++)
                    {
                        long packed = PackedQuad.Pack(x, y, z, 1, 1, 0, 1);
                        PackedQuad.Unpack(packed, out int gx, out int gy, out int gz,
                            out int gw, out int gh, out int gf, out int gt);

                        if (gx != x || gy != y || gz != z)
                            throw new Exception($"Position roundtrip failed: ({x},{y},{z}) -> ({gx},{gy},{gz})");
                        if (gw != 1 || gh != 1) throw new Exception("1x1 quad dimensions wrong");
                    }
                }
            }

            // Test dimensions (1-16)
            for (int w = 1; w <= 16; w++)
            {
                for (int h = 1; h <= 16; h++)
                {
                    long packed = PackedQuad.Pack(0, 0, 0, w, h, 0, 1);
                    PackedQuad.Unpack(packed, out _, out _, out _,
                        out int gw, out int gh, out _, out _);
                    if (gw != w || gh != h)
                        throw new Exception($"Dimension roundtrip failed: {w}x{h} -> {gw}x{gh}");
                }
            }

            // Test all 6 face directions
            for (int face = 0; face < 6; face++)
            {
                long packed = PackedQuad.Pack(0, 0, 0, 1, 1, face, 1);
                PackedQuad.Unpack(packed, out _, out _, out _, out _, out _, out int gf, out _);
                if (gf != face) throw new Exception($"Face roundtrip failed: {face} -> {gf}");
            }

            // Test block type (0-4095)
            for (int type = 0; type < 4096; type += 100)
            {
                long packed = PackedQuad.Pack(0, 0, 0, 1, 1, 0, type);
                PackedQuad.Unpack(packed, out _, out _, out _, out _, out _, out _, out int gt);
                if (gt != type) throw new Exception($"Block type roundtrip failed: {type} -> {gt}");
            }

            // Test damage + AO in full unpack
            long fullPacked = PackedQuad.Pack(5, 10, 3, 8, 4, 2, 1000, 12, 0xAB);
            PackedQuad.UnpackFull(fullPacked, out int fx, out int fy, out int fz,
                out int fw, out int fh, out int ff, out int ft, out int fd, out int fao);
            if (fx != 5 || fy != 10 || fz != 3) throw new Exception("Full unpack position wrong");
            if (fw != 8 || fh != 4) throw new Exception("Full unpack dimensions wrong");
            if (ff != 2) throw new Exception("Full unpack face wrong");
            if (ft != 1000) throw new Exception("Full unpack type wrong");
            if (fd != 12) throw new Exception($"Full unpack damage wrong: got {fd}");
            if (fao != 0xAB) throw new Exception($"Full unpack AO wrong: got {fao}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task GreedyMerge_DamagedBlocks_MergeBySameType() => await RunTest(async accelerator =>
        {
            // Blocks with same type but different damage levels should merge
            int chunkXZ = 4;
            int height = 4;
            int paddedXZ = chunkXZ + 2;
            int stride = paddedXZ * paddedXZ;
            var blocks = new int[stride * height];

            // Fill a 4x4 layer at y=0 with type 1, varying damage
            for (int z = 1; z <= chunkXZ; z++)
            {
                for (int x = 1; x <= chunkXZ; x++)
                {
                    int damage = (x + z) % 16; // varying damage 0-15
                    blocks[x + z * paddedXZ + 0 * stride] = PackedBlock.Pack(1, damage);
                }
            }

            var cpuOccupancy = FaceCullCpuReference.BuildOccupancy(blocks, paddedXZ, height);
            var cpuFaceMasks = FaceCullCpuReference.FaceCull(cpuOccupancy, paddedXZ, height);
            var originalMasks = (long[])cpuFaceMasks.Clone();

            var quads = GreedyMergeCpuReference.GreedyMerge(cpuFaceMasks, blocks, chunkXZ, height, paddedXZ);

            // The top face (+Y at y=0) should merge into ONE quad because all blocks are type 1
            // (damage varies but type is the same)
            int topQuadCount = 0;
            foreach (var q in quads)
            {
                PackedQuad.Unpack(q, out _, out int qy, out _, out _, out _, out int face, out _);
                if (face == VoxelMeshConstants.FacePosY && qy == 0)
                    topQuadCount++;
            }

            if (topQuadCount != 1)
                throw new Exception($"Damaged blocks of same type should merge: got {topQuadCount} top quads, expected 1");

            // Coverage must still be exact
            var error = GreedyMergeCpuReference.VerifyQuadCoverage(quads, originalMasks, chunkXZ, height);
            if (error != null) throw new Exception($"Coverage error: {error}");

            await Task.CompletedTask;
        });
    }
}
