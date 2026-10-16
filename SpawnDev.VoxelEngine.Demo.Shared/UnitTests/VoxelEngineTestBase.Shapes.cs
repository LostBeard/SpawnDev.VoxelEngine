using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Shaped blocks (stairs, slabs, fences, doors...): a block that is not a cube is a few BOXES of its cell. The mesher turns each into the faces that can be seen, the vertex shader places a
    // face by its box and takes the tile, the window of the tile and the turn from the shape table.
    public abstract partial class VoxelEngineTestBase
    {
        private static ShapeBox ShBox(int x0, int y0, int z0, int x1, int y1, int z1, ShapeFace side, ShapeFace top, ShapeFace bottom) =>
            new(x0, y0, z0, x1, y1, z1, new[] { side, side, side, side, top, bottom });

        [TestMethod]
        public async Task Shapes_ATable_PacksItsBoxesAndFacesTest() => await Task.Run(() =>
        {
            var table = new ShapeTable();
            var slab = ShBox(0, 0, 0, 16, 8, 16, new ShapeFace(3, 0, 8, 16, 16), new ShapeFace(4, 0, 0, 16, 16, 2, true), ShapeFace.None);
            var post = ShBox(6, 0, 6, 10, 16, 10, ShapeFace.Whole(5), ShapeFace.Whole(5), ShapeFace.Whole(5));
            int a = table.Add(new[] { slab });
            int b = table.Add(new[] { slab, post });
            if (a != 0 || b != 1) throw new Exception($"slots are numbered from 0, got {a} and {b}");
            var words = table.Words();
            if (words.Length != 2 * ShapeTable.WordsPerSlot) throw new Exception($"two slots are {2 * ShapeTable.WordsPerSlot} words, got {words.Length}");
            uint w0 = words[0];
            if ((w0 >> 31) != 1 || (w0 & 31) != 0 || ((w0 >> 15) & 31) != 16 || ((w0 >> 20) & 31) != 8 || ((w0 >> 25) & 31) != 16) throw new Exception($"the slab's bounds are packed wrong: {w0:x8}");
            if ((words[1] & 0xFFFF) != 3) throw new Exception("face 0 of the slab shows tile 3");
            if (words[2] != (0u | 8u << 8 | 16u << 16 | 16u << 24)) throw new Exception($"its window is (0,8)-(16,16), got {words[2]:x8}");
            // The top face (face 4): tile 4, turned twice, tinted
            if (words[1 + 8] != (4u | 2u << 16 | 1u << 18)) throw new Exception($"the top face word is tile | turn | tint, got {words[9]:x8}");
            if ((words[1 + 10] & 0xFFFF) != 0xFFFF) throw new Exception("a face with no texture says so");
            // The second box of the second slot starts 16 words after the first
            int second = ShapeTable.WordsPerSlot + ShapeTable.WordsPerBox;
            if ((words[second] >> 31) != 1 || (words[second] & 31) != 6) throw new Exception("the post is the second box of slot 1");
            if ((words[ShapeTable.WordsPerSlot + 2 * ShapeTable.WordsPerBox] >> 31) != 0) throw new Exception("a box that is not there has no exists bit");
            try { table.Add(Enumerable.Repeat(slab, 33).ToArray()); throw new Exception("thirty-three boxes must be refused"); } catch (ArgumentException) { }
            try { _ = new ShapeBox(0, 0, 0, 17, 1, 1, new ShapeFace[6]); throw new Exception("a box outside its cell must be refused"); } catch (ArgumentOutOfRangeException) { }
        });

        [TestMethod]
        public async Task Shapes_TheMesher_EmitsTheVisibleFacesOfEachBox_AndHidesTheOnesAgainstAnOpaqueNeighbourTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            const int sections = 2;
            using var arena = new QuadArena(accelerator);
            using var mesher = new ColumnMesher(accelerator, sections) { MissingNeighborsAreSolid = false, EmbedSectionIndex = true, Arena = arena };
            var table = new ShapeTable();
            // A stair: the lower half of the cell, and the back half of the upper
            var stair = new[]
            {
                ShBox(0, 0, 0, 16, 8, 16, ShapeFace.Whole(0), ShapeFace.Whole(0), ShapeFace.Whole(0)),
                ShBox(8, 8, 0, 16, 16, 16, ShapeFace.Whole(0), ShapeFace.Whole(0), ShapeFace.None),   // no bottom: it rests on the lower box
            };
            int slot = table.Add(stair);
            mesher.SetBlockClasses(new int[] { BlockClass.None, BlockClass.Opaque, BlockClass.ShapeSlot(slot) });
            mesher.SetShapes(table);
            var col = new ushort[16 * 16 * 16 * sections];
            col[4 + 4 * 16 + 3 * 256] = 2;          // the stair at (4, 3, 4)
            using var alone = CmUpload(accelerator, col);
            using (var mesh = await mesher.MeshAsync(alone))
            {
                var quads = await mesh.Opaque.CopyToHostAsync();
                // Box 0: six faces; box 1: five (no bottom). Nothing hides any of them
                if (quads.Length != 11) throw new Exception($"a stair alone has 6 + 5 faces, got {quads.Length}");
                if (!quads.All(PackedQuad.IsShapeFace)) throw new Exception("every quad of a shaped block is flagged as a box face");
                if (quads.Count(q => PackedQuad.UnpackBox(q) == 0) != 6 || quads.Count(q => PackedQuad.UnpackBox(q) == 1) != 5) throw new Exception("the quads say which box they belong to");
                foreach (var q in quads)
                {
                    PackedQuad.Unpack(q, out int x, out int y, out int z, out int w, out int h, out int face, out int type);
                    if (x != 4 || y != 3 || z != 4 || w != 1 || h != 1 || type != 2) throw new Exception($"a box face is a 1 x 1 quad of the cell and its block type, got ({x},{y},{z}) {w}x{h} type {type}");
                }
            }

            // Stone beside it on the +X side: the box faces on the +X boundary (both boxes reach x = 16) cannot be seen. The stone itself shows its -X face (a stair does not hide it)
            col[5 + 4 * 16 + 3 * 256] = 1;
            using var beside = CmUpload(accelerator, col);
            using (var mesh = await mesher.MeshAsync(beside))
            {
                var quads = await mesh.Opaque.CopyToHostAsync();
                var shapeQuads = quads.Where(PackedQuad.IsShapeFace).ToList();
                if (shapeQuads.Count != 9) throw new Exception($"against stone the stair loses its two +X faces: 9 left, got {shapeQuads.Count}");
                if (shapeQuads.Any(q => { PackedQuad.Unpack(q, out _, out _, out _, out _, out _, out int face, out _); return face == 0; })) throw new Exception("no +X face is left");
                var stone = quads.Where(q => !PackedQuad.IsShapeFace(q)).ToList();
                if (!stone.Any(q => { PackedQuad.Unpack(q, out _, out _, out _, out _, out _, out int face, out _); return face == 1; })) throw new Exception("the stone shows its -X face toward the stair");
            }

            // A column that holds no shaped block, with the table set, is meshed as before (and one without a table at all)
            var plain = new ushort[16 * 16 * 16 * sections];
            plain[4 + 4 * 16 + 3 * 256] = 1;
            using var gpuPlain = CmUpload(accelerator, plain);
            using (var mesh = await mesher.MeshAsync(gpuPlain))
                if (mesh.Opaque.Count != 6) throw new Exception($"a lone cube is six faces, got {mesh.Opaque.Count}");
        });

        /// <summary>Renders one cell (4, 3, 4) of a column through the mesher and the arena, seen from the front (south) or from above, and returns the colours at the given pixels of a 64 px image (16 px a block, the cell's centre at (32, 32)).</summary>
        private static async Task<(byte r, byte g, byte b)[]> ShRender(GPUDevice device, GPUQueue queue, Accelerator accelerator, byte[] atlas, int tiles, ShapeTable table, int[] classes,
            BlockMaterial[] materials, ushort[] column, bool fromAbove, (int x, int y)[] pixels, bool fromBehind = false)
        {
            const int sections = 2;
            using var arena = new QuadArena(accelerator);
            using var mesher = new ColumnMesher(accelerator, sections) { MissingNeighborsAreSolid = false, EmbedSectionIndex = true, Arena = arena };
            mesher.SetBlockClasses(classes);
            mesher.SetShapes(table);
            using var gpu = CmUpload(accelerator, column);
            using var mesh = await mesher.MeshAsync(gpu);
            using var textures = BlockTextureAtlas.FromRgba(device, queue, atlas, 16 * tiles, 16, 16);
            using var pipeline = new BlockRenderPipeline();
            pipeline.Init(device, queue, "bgra8unorm", maxSections: 4);
            pipeline.SetTextures(textures);
            pipeline.SetMaterials(materials);
            pipeline.SetShapes(table);
            pipeline.SetSectionOrigin(0, Vector3.Zero);
            var centre = new Vector3(4.5f, 3.5f, 4.5f);
            var (eye, target, up) = fromAbove
                ? (centre + new Vector3(0, 2.5f, 0), centre - new Vector3(0, 0.5f, 0), -Vector3.UnitZ)
                : fromBehind
                    ? (centre - new Vector3(0, 0, 2.5f), centre, Vector3.UnitY)
                    : (centre + new Vector3(0, 0, 2.5f), centre, Vector3.UnitY);
            pipeline.BeginFrame(BrNeutral(BrView(eye, target, up) * BrProj90, eye));
            var (colorTex, depthTex, colorView, depthView) = CreateOffscreenTargets(device, 64, 64);
            try
            {
                using var encoder = device.CreateCommandEncoder();
                using (var pass = pipeline.BeginRenderPass(encoder, colorView, depthView, Vector3.Zero))
                {
                    var list = mesh.Opaque;
                    if (list.Count > 0)
                    {
                        pipeline.BindPass(pass, BlockRenderPipeline.Pass.Opaque, list.Buffer!.GetGPUBuffer()!);
                        pipeline.Draw(pass, list.FirstQuad, list.Count, 0);
                    }
                    pass.End();
                }
                using var cmd = encoder.Finish();
                queue.Submit(new[] { cmd });
                await queue.OnSubmittedWorkDone();
                var px = await ReadBackPixels(device, queue, colorTex, 64, 64);
                return pixels.Select(p => { var (r, g, b, _) = GetPixel(px, 64, p.x, p.y); return (r, g, b); }).ToArray();
            }
            finally { DisposeTargets(colorTex, depthTex, colorView, depthView); }
        }

        private static bool ShDominant((byte r, byte g, byte b) c, char channel)
        {
            int r = c.r, g = c.g, b = c.b;
            return channel switch { 'r' => r > 100 && r > g * 2 && r > b * 2, 'g' => g > 100 && g > r * 2 && g > b * 2, 'b' => b > 100 && b > r * 2 && b > g * 2, _ => r + g + b < 12 };
        }

        [TestMethod]
        public async Task Shapes_ASlab_IsOnlyItsHalf_AndShowsTheWindowOfItsTileTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            // Tile 0: red in its upper half, blue in its lower; tile 1: green
            var atlas = BaSolidAtlas(16, 2, 1, k => k == 1 ? ((byte)0, (byte)255, (byte)0, (byte)255) : ((byte)0, (byte)0, (byte)255, (byte)255));
            for (int y = 0; y < 8; y++) for (int x = 0; x < 16; x++) { int i = (y * 32 + x) * 4; atlas[i] = 255; atlas[i + 2] = 0; }
            ShapeTable Table(ShapeFace side)
            {
                var t = new ShapeTable();
                t.Add(new[] { ShBox(0, 0, 0, 16, 8, 16, side, ShapeFace.Whole(1), ShapeFace.Whole(1)) });
                return t;
            }
            var classes = new int[] { BlockClass.None, BlockClass.ShapeSlot(0) };
            var materials = new[] { BlockMaterial.Missing, BlockMaterial.Uniform(0, flags: BlockMaterial.Shape(0)) };
            var col = new ushort[16 * 16 * 16 * 2];
            col[4 + 4 * 16 + 3 * 256] = 1;
            // Seen from the front the cell spans rows 24 (y = 4) to 40 (y = 3); the slab is rows 32..40: its middle row 36 is inside it, row 28 is above it
            var pixels = new[] { (32, 36), (32, 28), (32, 33), (32, 39) };
            var lower = await ShRender(device, queue, accelerator, atlas, 2, Table(new ShapeFace(0, 0, 8, 16, 16)), classes, materials, col, false, pixels);
            if (!ShDominant(lower[0], 'b')) throw new Exception($"the slab's side with the lower half of the tile is blue, got {lower[0]}");
            if (!ShDominant(lower[1], '0')) throw new Exception($"above the slab there is nothing, got {lower[1]}");
            if (!ShDominant(lower[2], 'b') || !ShDominant(lower[3], 'b')) throw new Exception($"the whole side shows the window: {lower[2]} {lower[3]}");
            var upper = await ShRender(device, queue, accelerator, atlas, 2, Table(new ShapeFace(0, 0, 0, 16, 8)), classes, materials, col, false, pixels);
            if (!ShDominant(upper[0], 'r') || !ShDominant(upper[3], 'r')) throw new Exception($"with the upper window the side is red, got {upper[0]} {upper[3]}");
            // The window can be a part of the width too: the left half of the tile's lower row (all blue here, so use the whole-tile test for mixing): a window of the whole tile squeezes the red/blue halves into the slab's 8 px
            var whole = await ShRender(device, queue, accelerator, atlas, 2, Table(ShapeFace.Whole(0)), classes, materials, col, false, new[] { (32, 33), (32, 38) });
            if (!ShDominant(whole[0], 'r') || !ShDominant(whole[1], 'b')) throw new Exception($"the whole tile on a half-high face is squeezed: red above, blue below, got {whole[0]} {whole[1]}");
            // From above the top is green, at the slab's height
            var top = await ShRender(device, queue, accelerator, atlas, 2, Table(new ShapeFace(0, 0, 8, 16, 16)), classes, materials, col, true, new[] { (32, 32), (28, 28), (36, 36) });
            if (!top.All(c => ShDominant(c, 'g'))) throw new Exception($"the slab's top is tile 1 (green), got {top[0]} {top[1]} {top[2]}");
        });

        [TestMethod]
        public async Task Shapes_ASecondSlot_AndASecondBox_AreDrawnWhereTheyAreTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            // Tiles: 0 red, 1 green, 2 blue. Slot 0 is a full blue slab; slot 1 is a low red step (the lower quarter, full width) with a green post standing on it (the middle quarter, full height)
            var atlas = BaSolidAtlas(16, 3, 1, k => k switch { 0 => ((byte)255, (byte)0, (byte)0, (byte)255), 1 => ((byte)0, (byte)255, (byte)0, (byte)255), _ => ((byte)0, (byte)0, (byte)255, (byte)255) });
            var table = new ShapeTable();
            table.Add(new[] { ShBox(0, 0, 0, 16, 8, 16, ShapeFace.Whole(2), ShapeFace.Whole(2), ShapeFace.Whole(2)) });
            table.Add(new[]
            {
                ShBox(0, 0, 0, 16, 4, 16, ShapeFace.Whole(0), ShapeFace.Whole(0), ShapeFace.Whole(0)),
                ShBox(6, 4, 6, 10, 16, 10, ShapeFace.Whole(1), ShapeFace.Whole(1), ShapeFace.None),
            });
            var classes = new int[] { BlockClass.None, BlockClass.ShapeSlot(0), BlockClass.ShapeSlot(1) };
            var materials = new[] { BlockMaterial.Missing, BlockMaterial.Uniform(2, flags: BlockMaterial.Shape(0)), BlockMaterial.Uniform(2, flags: BlockMaterial.Shape(1)) };
            var col = new ushort[16 * 16 * 16 * 2];
            col[4 + 4 * 16 + 3 * 256] = 2;      // slot 1 in the cell
            // Front view: the cell is columns 24..40, rows 24 (top) to 40. The post is columns 30..34 from the step's top (row 36) to the top; the step is rows 36..40 across
            var got = await ShRender(device, queue, accelerator, atlas, 3, table, classes, materials, col, false, new[] { (32, 28), (26, 38), (26, 28), (32, 38) });
            if (!ShDominant(got[0], 'g')) throw new Exception($"the post (second box of slot 1) is green, got {got[0]}");
            if (!ShDominant(got[1], 'r') || !ShDominant(got[3], 'r')) throw new Exception($"the step (first box) is red, got {got[1]} {got[3]}");
            if (!ShDominant(got[2], '0')) throw new Exception($"beside the post above the step there is nothing, got {got[2]}");
        });

        [TestMethod]
        public async Task Shapes_ASlotWithManyBoxes_DrawsEveryOneOfThemTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            // Tiles: 0 red, 1 green. Twenty-one one-sixteenth-wide posts side by side; the last (box 20) is green. A fence has nine boxes, iron bars up to eighteen, so the box number must reach well past 8
            var atlas = BaSolidAtlas(16, 2, 1, k => k == 1 ? ((byte)0, (byte)255, (byte)0, (byte)255) : ((byte)255, (byte)0, (byte)0, (byte)255));
            var table = new ShapeTable();
            table.Add(new[] { ShBox(0, 0, 0, 16, 16, 16, ShapeFace.Whole(1), ShapeFace.Whole(1), ShapeFace.Whole(1)) });   // slot 0: something else, so slot 1 is tested
            // Boxes 0..15: a row at the back of the cell (z 14..15 sixteenths, red); boxes 16..20: five posts in front of it (z 15..16), the last (box 20, at x 4..5) green
            var boxes = Enumerable.Range(0, 16).Select(i => ShBox(i, 0, 14, i + 1, 16, 15, ShapeFace.Whole(0), ShapeFace.None, ShapeFace.None))
                .Concat(Enumerable.Range(0, 5).Select(i => ShBox(i, 0, 15, i + 1, 16, 16, ShapeFace.Whole(i == 4 ? 1 : 0), ShapeFace.None, ShapeFace.None))).ToArray();
            table.Add(boxes);
            var classes = new int[] { BlockClass.None, BlockClass.ShapeSlot(1) };
            var materials = new[] { BlockMaterial.Missing, BlockMaterial.Uniform(0, flags: BlockMaterial.Shape(1)) };
            var col = new ushort[16 * 16 * 16 * 2];
            col[4 + 4 * 16 + 3 * 256] = 1;
            // Front view: the cell is columns 24..40; box i (i/16 .. (i+1)/16 of the width) is column 24 + i
            // From the front the cell is columns 24..40 (x 0 at column 24): columns 24..28 are the posts of boxes 16..20, the rest the row behind them (boxes 5..15)
            var got = await ShRender(device, queue, accelerator, atlas, 2, table, classes, materials, col, false, Enumerable.Range(0, 16).Select(i => (24 + i, 32)).ToArray());
            for (int i = 0; i < 4; i++) if (!ShDominant(got[i], 'r')) throw new Exception($"box {16 + i} is red, got {got[i]}");
            if (!ShDominant(got[4], 'g')) throw new Exception($"box 20 (the twenty-first) is green, got {got[4]}");
            for (int i = 5; i < 16; i++) if (!ShDominant(got[i], 'r')) throw new Exception($"box {i} is red, got {got[i]}");
        });

        [TestMethod]
        public async Task Shapes_ATurnedFace_TurnsItsTileClockwise_SeenFromOutsideTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            // Tile 0: red in its top-left quarter, blue elsewhere
            var atlas = BaSolidAtlas(16, 1, 1, _ => ((byte)0, (byte)0, (byte)255, (byte)255));
            for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) { int i = (y * 16 + x) * 4; atlas[i] = 255; atlas[i + 2] = 0; }
            var classes = new int[] { BlockClass.None, BlockClass.ShapeSlot(0) };
            var materials = new[] { BlockMaterial.Missing, BlockMaterial.Uniform(0, flags: BlockMaterial.Shape(0)) };
            var col = new ushort[16 * 16 * 16 * 2];
            col[4 + 4 * 16 + 3 * 256] = 1;
            // From above (+X to the right, +Z down): the cell is columns 24..40, rows 24..40; its quarters are at (28,28) (-X,-Z), (36,28), (28,36), (36,36)
            var corners = new[] { (28, 28), (36, 28), (28, 36), (36, 36) };
            var expected = new[] { 0, 1, 3, 2 };   // the red quarter after 0, 1, 2, 3 clockwise quarter turns: top-left, top-right, bottom-right, bottom-left (an index into corners)
            for (int turn = 0; turn < 4; turn++)
            {
                var t = new ShapeTable();
                t.Add(new[] { ShBox(0, 0, 0, 16, 16, 16, ShapeFace.None, new ShapeFace(0, 0, 0, 16, 16, turn), ShapeFace.None) });
                var got = await ShRender(device, queue, accelerator, atlas, 1, t, classes, materials, col, true, corners);
                for (int c = 0; c < 4; c++)
                    if (ShDominant(got[c], 'r') != (c == expected[turn]))
                        throw new Exception($"turned {turn} quarter(s), the red quarter should be corner {expected[turn]}; corner {c} is {got[c]}");
            }
        });

        [TestMethod]
        public async Task Shapes_TheSideFaces_ShowTheTileUprightAndUnmirrored_FromEverySideTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            CmRequireGpuMeshing(accelerator);
            // The same tile (red top-left quarter) on all four sides of a full box: seen from outside the red quarter is always at the top left
            var atlas = BaSolidAtlas(16, 1, 1, _ => ((byte)0, (byte)0, (byte)255, (byte)255));
            for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) { int i = (y * 16 + x) * 4; atlas[i] = 255; atlas[i + 2] = 0; }
            var classes = new int[] { BlockClass.None, BlockClass.ShapeSlot(0) };
            var materials = new[] { BlockMaterial.Missing, BlockMaterial.Uniform(0, flags: BlockMaterial.Shape(0)) };
            var col = new ushort[16 * 16 * 16 * 2];
            col[4 + 4 * 16 + 3 * 256] = 1;
            var t = new ShapeTable();
            t.Add(new[] { ShBox(0, 0, 0, 16, 16, 16, ShapeFace.Whole(0), ShapeFace.None, ShapeFace.None) });
            // Seen from the south the face is +Z; from the north (behind) it is -Z. The block spans columns 24..40, rows 24..40 either way
            var corners = new[] { (28, 28), (36, 28), (28, 36), (36, 36) };
            foreach (bool behind in new[] { false, true })
            {
                var got = await ShRender(device, queue, accelerator, atlas, 1, t, classes, materials, col, false, corners, fromBehind: behind);
                if (!ShDominant(got[0], 'r') || ShDominant(got[1], 'r') || ShDominant(got[2], 'r') || ShDominant(got[3], 'r'))
                    throw new Exception($"{(behind ? "from the north" : "from the south")} the red quarter is at the top left, got {got[0]} {got[1]} {got[2]} {got[3]}");
            }
        });
    }
}
