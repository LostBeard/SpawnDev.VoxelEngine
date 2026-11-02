using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // AMBIENT OCCLUSION: baked into the quads by the column mesher (per corner, 2 bits each), shaded per vertex by the block shader.
    public abstract partial class VoxelEngineTestBase
    {
        // The quad's own axes for each face (the vertex shader's: A runs along the width, B along the height) as unit steps
        private static readonly (int ax, int ay, int az, int bx, int by, int bz)[] CmAxes =
        {
            (0, 0, 1, 0, 1, 0),    // +X: A +Z, B +Y
            (0, 0, -1, 0, 1, 0),   // -X: A -Z, B +Y
            (-1, 0, 0, 0, 1, 0),   // +Z: A -X, B +Y
            (1, 0, 0, 0, 1, 0),    // -Z: A +X, B +Y
            (1, 0, 0, 0, 0, 1),    // +Y: A +X, B +Z
            (1, 0, 0, 0, 0, -1),   // -Y: A +X, B -Z
        };

        // Does the block at a world cell occlude in the opaque pass? A missing neighbour column, the air above and below the column and the ring corners do not
        private static bool CmOccluder(ushort[] col, ushort[]?[] nbr, int sections, int x, int y, int z)
        {
            int t = CmBlock(col, nbr, sections, x, y, z);
            return t > 0 && ((CmFlags[t] >> (BlockClass.OpaqueShift + 1)) & 1) != 0;
        }

        // The corners' occlusion of a face of a cell, 2 bits each in the order corner = a * 2 + b: the two cells beside the corner and the one between them, in the layer in front of the face
        private static int CmCpuAo(ushort[] col, ushort[]?[] nbr, int sections, int face, int x, int y, int z)
        {
            var (dx, dy, dz) = CmDirs[face];
            var ax = CmAxes[face];
            int result = 0;
            for (int c = 0; c < 4; c++)
            {
                int sa = (c >> 1) == 1 ? 1 : -1, sb = (c & 1) == 1 ? 1 : -1;
                int fx = x + dx, fy = y + dy, fz = z + dz;
                bool side1 = CmOccluder(col, nbr, sections, fx + sa * ax.ax, fy + sa * ax.ay, fz + sa * ax.az);
                bool side2 = CmOccluder(col, nbr, sections, fx + sb * ax.bx, fy + sb * ax.by, fz + sb * ax.bz);
                bool corner = CmOccluder(col, nbr, sections, fx + sa * ax.ax + sb * ax.bx, fy + sa * ax.ay + sb * ax.by, fz + sa * ax.az + sb * ax.bz);
                int v = side1 && side2 ? 3 : (side1 ? 1 : 0) + (side2 ? 1 : 0) + (corner ? 1 : 0);
                result |= v << (2 * c);
            }
            return result;
        }

        private static bool CmConstantAlongA(int ao) => (ao & 3) == ((ao >> 4) & 3) && ((ao >> 2) & 3) == ((ao >> 6) & 3);
        private static bool CmConstantAlongB(int ao) => (ao & 3) == ((ao >> 2) & 3) && ((ao >> 4) & 3) == ((ao >> 6) & 3);

        // The unit faces of a list of quads with their occlusion, and the number of quads; checks as it goes that no cell is covered twice and that a quad's occlusion is every one of its cells' and constant along
        // each axis it was merged along
        private static async Task<(HashSet<CmFace> faces, int quads, int merged)> CmDecodeAo(ColumnMesh.QuadList list, ushort[] col, ushort[]?[] nbr, int sections, bool aoOn, string pass)
        {
            var set = new HashSet<CmFace>();
            if (list.Buffer == null) return (set, 0, 0);
            var quads = await list.CopyToHostAsync();
            int merged = 0;
            for (int s = 0; s < list.SectionCount; s++)
                for (int q = list.SectionStart[s]; q < list.SectionStart[s + 1]; q++)
                {
                    PackedQuad.UnpackFull(quads[q], out int x, out int y, out int z, out int w, out int h, out int face, out int type, out _, out int ao);
                    int baseY = s * 16;
                    if (!aoOn && ao != 0) throw new Exception($"{pass}: a quad has occlusion {ao:X2} with the occlusion switched off");
                    if (w > 1 && !CmConstantAlongA(ao)) throw new Exception($"{pass}: a quad merged {w} wide has occlusion {ao:X2}, which changes along its width");
                    if (h > 1 && !CmConstantAlongB(ao)) throw new Exception($"{pass}: a quad merged {h} high has occlusion {ao:X2}, which changes along its height");
                    if (w * h > 1) merged++;
                    for (int i = 0; i < w; i++)
                        for (int j = 0; j < h; j++)
                        {
                            CmFace unit = face switch
                            {
                                0 or 1 => new CmFace(face, x, baseY + y + j, z + i, type),
                                2 or 3 => new CmFace(face, x + i, baseY + y + j, z, type),
                                _ => new CmFace(face, x + i, baseY + y, z + j, type),
                            };
                            if (!set.Add(unit)) throw new Exception($"{pass}: unit face {unit} is covered by two quads");
                            if (aoOn)
                            {
                                int want = CmCpuAo(col, nbr, sections, unit.Face, unit.X, unit.Y, unit.Z);
                                if (want != ao) throw new Exception($"{pass}: face {unit.Face} of cell ({unit.X}, {unit.Y}, {unit.Z}) has occlusion {ao:X2} in its quad, {want:X2} by the neighbours");
                            }
                        }
                }
            return (set, list.Count, merged);
        }

        [TestMethod]
        public async Task ColumnMesher_AmbientOcclusion_BakesEachCellsCorners_AndMergesOnlyAlongWhatDoesNotChangeTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 3;
            using var mesher = new ColumnMesher(accelerator, sections);
            mesher.SetBlockClasses(CmFlags);

            var col = CmTerrain(sections, 5);
            // Stone pillars that meet across the boundary of two sections (y 15 and 16), and a stair of blocks in a corner
            col[CmIdx(8, 15, 8)] = CmStone; col[CmIdx(9, 16, 8)] = CmStone; col[CmIdx(8, 16, 9)] = CmStone;
            for (int i = 0; i < 4; i++) col[CmIdx(1 + i, 10 + i, 1)] = CmStone;
            // Neighbours: -X and -Z loaded, +X and +Z not (a missing column is open for the occlusion, and still hides the faces towards it)
            var nbr = new ushort[]?[] { CmTerrain(sections, 6), null, CmTerrain(sections, 7), null };

            using var gpuCol = CmUpload(accelerator, col);
            var gpuNbr = nbr.Select(n => n == null ? null : CmUpload(accelerator, n)).ToArray();
            try
            {
                mesher.AmbientOcclusion = false;
                int plainQuads;
                using (var plain = await mesher.MeshAsync(gpuCol, gpuNbr[0], gpuNbr[1], gpuNbr[2], gpuNbr[3]))
                {
                    var (faces, quads, _) = await CmDecodeAo(plain.Opaque, col, nbr, sections, false, "opaque, occlusion off");
                    CmAssertSame(CmExpectedCubeFaces(col, nbr, sections, BlockClass.OpaqueShift, true), faces, "opaque faces, occlusion off");
                    plainQuads = quads;
                }

                mesher.AmbientOcclusion = true;
                using (var mesh = await mesher.MeshAsync(gpuCol, gpuNbr[0], gpuNbr[1], gpuNbr[2], gpuNbr[3]))
                {
                    if (mesh.Overflow != 0) throw new Exception($"{mesh.Overflow} quads overflowed");
                    var (faces, quads, merged) = await CmDecodeAo(mesh.Opaque, col, nbr, sections, true, "opaque");
                    // The same faces are there as without it: occlusion shades them, it never adds or takes one away
                    CmAssertSame(CmExpectedCubeFaces(col, nbr, sections, BlockClass.OpaqueShift, true), faces, "opaque faces, occlusion on");
                    if (quads < plainQuads) throw new Exception($"occlusion cannot make fewer quads: {quads} against {plainQuads}");
                    if (merged == 0) throw new Exception("flat ground still merges: no quad of more than one cell");
                    TestContext_Log($"column: {plainQuads} quads without occlusion, {quads} with ({merged} of them merged)");
                    // The water pass and the plants carry none
                    var water = await CmDecodeAo(mesh.Water, col, nbr, sections, false, "water");
                    var plants = await CmDecodeAo(mesh.Plants, col, nbr, sections, false, "plants");
                }
            }
            finally { foreach (var b in gpuNbr) b?.Dispose(); }
        });

        [TestMethod]
        public async Task ColumnMesher_AmbientOcclusion_AFlatFloorStaysOneQuad_AndABlockOnItDarkensItsFootTest() => await RunTest(async accelerator =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 1;
            using var mesher = new ColumnMesher(accelerator, sections) { AmbientOcclusion = true };
            mesher.SetBlockClasses(CmFlags);
            var col = CmEmpty(sections);
            for (int z = 0; z < 16; z++) for (int x = 0; x < 16; x++) col[CmIdx(x, 0, z)] = CmStone;
            var nbr = new ushort[]?[] { null, null, null, null };
            using (var gpu = CmUpload(accelerator, col))
            using (var mesh = await mesher.MeshAsync(gpu))
            {
                var (_, quads, _) = await CmDecodeAo(mesh.Opaque, col, nbr, sections, true, "floor");
                // top face: one 16 x 16 quad; the four sides of the slab (missing neighbours hide them: MissingNeighborsAreSolid), the bottom: one
                if (quads > 2) throw new Exception($"an open floor is {quads} quads, expected at most the top and the bottom");
            }
            col[CmIdx(8, 1, 8)] = CmStone;   // a block standing on it
            using (var gpu = CmUpload(accelerator, col))
            using (var mesh = await mesher.MeshAsync(gpu))
            {
                var (faces, quads, _) = await CmDecodeAo(mesh.Opaque, col, nbr, sections, true, "floor with a block");
                CmAssertSame(CmExpectedCubeFaces(col, nbr, sections, BlockClass.OpaqueShift, true), faces, "the faces of a floor with a block");
                // The floor beside the block: the cell west of it (7, 0, 8) has its +X... the top face's corner towards the block is one occluder dark
                int ao = CmCpuAo(col, nbr, sections, 4, 7, 0, 8);
                if (ao == 0) throw new Exception("the floor cell beside the block is occluded at its corners nearest it");
                TestContext_Log($"floor with a block: {quads} quads");
            }
        });

        // A single quad on the top face of the block at (0, 0, 0) with chosen corner occlusion, seen from above: the shade of a pixel near each corner (null: the quad is not drawn)
        private static async Task<(byte r, byte g, byte b, byte a)> AoRender(GPUDevice device, GPUQueue queue, int ao, int dx, int dy)
        {
            var materials = new BlockMaterial[2];
            materials[1] = BlockMaterial.Uniform(BrLayerCyan);
            var eye = new Vector3(0.5f, 2.5f, 0.5f);
            var frame = BrNeutral(BrView(eye, new Vector3(0.5f, 1f, 0.5f), -Vector3.UnitZ) * BrProj90, eye);
            frame.Ambient = new Vector4(0.5f, 0.5f, 0.5f, 0f);
            using var textures = BlockTextureAtlas.FromRgba(device, queue, BrAtlas(), 32, 32, BrTile);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            pipeline.BeginFrame(frame);
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            GPUBuffer? buffer = null;
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    buffer = CreateQuadBuffer(device, queue, new[] { PackedQuad.Pack(0, 0, 0, 1, 1, 4, 1, ao: ao) });
                    pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, buffer);
                    pipeline.Draw(pass, 0, 1, 0);
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var pixels = await ReadBackPixels(device, queue, colorTex, 64, 64);
                return GetPixel(pixels, 64, 32 + dx, 32 + dy);
            }
            finally
            {
                if (buffer != null) { buffer.Destroy(); buffer.Dispose(); }
                DisposeTargets(colorTex, depthTex, colorView, depthView);
            }
        }

        [TestMethod]
        public async Task BlockRender_AmbientOcclusion_ShadesEachCornerOfAQuadTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            // The cyan top face (0, 128, 128 under half ambient light). Seen from above with -Z up: +X to the right and +Z down, so corner c = a * 2 + b of the top face (A +X, B +Z) is
            // c0 top left, c1 bottom left, c2 top right, c3 bottom right. A pixel 9 out from the middle (0.42 of a block) is 0.92 of the way to a corner.
            void Expect((byte r, byte g, byte b, byte a) got, int green, string what, int tolerance = 7)
            {
                if (Math.Abs(got.g - green) > tolerance || Math.Abs(got.b - green) > tolerance || got.r > 6)
                    throw new Exception($"{what}: ({got.r},{got.g},{got.b}), expected green and blue {green} +-{tolerance}");
            }
            Expect(await AoRender(device, queue, 0, 0, 0), 128, "no occlusion: the light as it was");
            int topLeft = 3;   // corner 0 shut, the rest open
            // Where the dark corner is: 0.92 * 0.92 of its weight at the pixel next to it, 0.08 * 0.08 at the far one: 3 * 0.2 * 0.85 = 0.51 darker
            Expect(await AoRender(device, queue, topLeft, -9, -9), (int)(128 * (1 - 0.6 * 0.846)), "next to the dark corner");
            Expect(await AoRender(device, queue, topLeft, 9, 9), 128, "the far corner is open");
            Expect(await AoRender(device, queue, topLeft, 9, -9), 128, "so is a neighbouring one", 9);
            // Both of the left corners one occluder (a wall along the left side): the whole left edge is 0.8 and the right edge open
            int leftWall = 1 | (1 << 2);   // c0 = 1, c1 = 1
            Expect(await AoRender(device, queue, leftWall, -9, 0), (int)(128 * (1 - 0.2 * 0.92)), "beside a wall");
            Expect(await AoRender(device, queue, leftWall, 9, 0), 128, "away from the wall", 9);
            // A corner shut on one diagonal only (c0 and c3 shut, c1 and c2 open): the cut runs between the open pair, the dark corners stay in their own triangles
            int diagonal = 3 | (3 << 6);
            Expect(await AoRender(device, queue, diagonal, -9, -9), (int)(128 * (1 - 0.6 * 0.846)), "the first of two dark corners");
            Expect(await AoRender(device, queue, diagonal, 9, 9), (int)(128 * (1 - 0.6 * 0.846)), "the second");
            Expect(await AoRender(device, queue, diagonal, 9, -9), 128, "the open pair stays open", 12);
            // The other diagonal (c1 and c2 shut, c0 and c3 open): the cut flips to run between the open pair, so the middle is open and each dark corner is its own triangle's
            int anti = (3 << 2) | (3 << 4);
            Expect(await AoRender(device, queue, anti, 0, 0), 128, "the middle of two opposite dark corners, cut between the open ones", 12);
            Expect(await AoRender(device, queue, anti, -9, 9), (int)(128 * (1 - 0.6 * 0.846)), "the first of them");
            Expect(await AoRender(device, queue, anti, 9, -9), (int)(128 * (1 - 0.6 * 0.846)), "the second");
        });

        // Logging a test line (TestContext is NUnit's in the desktop lane only)
        private static void TestContext_Log(string line) => Console.WriteLine("[ao] " + line);
    }
}
