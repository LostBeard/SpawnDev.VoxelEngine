using System.Numerics;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Culling;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // VisibilityGraph tests: Sodium-style graph-based occlusion culling.
    // Tests ComputeConnectivity (per-section flood fill) and ComputeVisibleSections (BFS traversal).
    // These are CPU-only tests - the visibility graph runs on CPU before GPU frustum culling.
    public abstract partial class VoxelEngineTestBase
    {
        /// <summary>
        /// Helper: create a sizeXZ * sizeXZ * sizeY section filled with a specific block type.
        /// blockType=0 for all-air, non-zero for all-solid.
        /// </summary>
        private static int[] CreateVisibilitySection(int sizeXZ, int sizeY, int blockType)
        {
            var blocks = new int[sizeXZ * sizeXZ * sizeY];
            if (blockType != 0)
            {
                for (int i = 0; i < blocks.Length; i++)
                    blocks[i] = PackedBlock.Pack(blockType);
            }
            return blocks;
        }

        /// <summary>
        /// Helper: create a section with specific blocks set in a solid fill.
        /// Starts solid, then carves out air at specified positions.
        /// </summary>
        private static int[] CreateSolidSectionWithAir(int sizeXZ, int sizeY, params (int x, int y, int z)[] airPositions)
        {
            var blocks = CreateVisibilitySection(sizeXZ, sizeY, 1); // all stone
            foreach (var (x, y, z) in airPositions)
            {
                blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ] = 0; // carve air
            }
            return blocks;
        }

        // ===== ComputeConnectivity tests =====

        /// <summary>
        /// Test 1: Open section (all air) connects all faces to all other faces.
        /// Air fills the entire section so light/sight can pass from any face to any other face.
        /// Every face-to-face bit should be set in the connectivity mask.
        /// </summary>
        [TestMethod]
        public void Visibility_OpenSection_AllFacesConnectedTest()
        {
            int size = 8; // small for fast test, large enough to be meaningful
            var blocks = CreateVisibilitySection(size, size, 0); // all air

            long connectivity = VisibilityGraph.ComputeConnectivity(blocks, size, size);

            // For each face pair (entry -> exit), the bit should be set
            // Layout: bit = entryFace * 8 + exitFace
            for (int entry = 0; entry < 6; entry++)
            {
                for (int exit = 0; exit < 6; exit++)
                {
                    if (exit == entry) continue; // skip same-face

                    int bit = entry * 8 + exit;
                    bool connected = (connectivity & (1L << bit)) != 0;

                    if (!connected)
                        throw new Exception(
                            $"Visibility_OpenSection: face {entry} should connect to face {exit} " +
                            $"in all-air section, but bit {bit} is not set. Connectivity=0x{connectivity:X12}");
                }
            }

            // Count total connected pairs: 6 faces * 5 other faces = 30 connections
            int connectedCount = 0;
            for (int b = 0; b < 48; b++)
            {
                if ((connectivity & (1L << b)) != 0)
                    connectedCount++;
            }

            if (connectedCount < 30)
                throw new Exception(
                    $"Visibility_OpenSection: expected at least 30 face-to-face connections, got {connectedCount}");
        }

        /// <summary>
        /// Test 2: Solid section (all stone) has no connections.
        /// No air blocks means no path through the section. Connectivity = 0.
        /// </summary>
        [TestMethod]
        public void Visibility_SolidSection_NoConnectionsTest()
        {
            int size = 8;
            var blocks = CreateVisibilitySection(size, size, 1); // all stone

            long connectivity = VisibilityGraph.ComputeConnectivity(blocks, size, size);

            if (connectivity != 0)
                throw new Exception(
                    $"Visibility_SolidSection: all-solid section should have 0 connectivity, " +
                    $"got 0x{connectivity:X12}");
        }

        /// <summary>
        /// Test 3: Tunnel section - air corridor through solid connects only tunnel-facing faces.
        /// Carve a straight 1-block-wide tunnel from -X face to +X face at y=4, z=4.
        /// Should connect face 0 (+X) to face 1 (-X) and vice versa.
        /// Should NOT connect +X to +Y (no vertical path exists).
        /// </summary>
        [TestMethod]
        public void Visibility_Tunnel_ConnectsOnlyTunnelFacesTest()
        {
            int size = 8;
            // Start with all stone
            var airPositions = new List<(int x, int y, int z)>();

            // Carve tunnel along X axis at y=4, z=4
            for (int x = 0; x < size; x++)
            {
                airPositions.Add((x, 4, 4));
            }

            var blocks = CreateSolidSectionWithAir(size, size, airPositions.ToArray());

            long connectivity = VisibilityGraph.ComputeConnectivity(blocks, size, size);

            // Face indices: PosX=0, NegX=1, PosZ=2, NegZ=3, PosY=4, NegY=5
            // Tunnel runs X axis: should connect +X <-> -X

            // +X -> -X connection (bit = 0*8 + 1 = 1)
            bool posXToNegX = (connectivity & (1L << (0 * 8 + 1))) != 0;
            if (!posXToNegX)
                throw new Exception(
                    $"Visibility_Tunnel: +X should connect to -X through X-axis tunnel. " +
                    $"Connectivity=0x{connectivity:X12}");

            // -X -> +X connection (bit = 1*8 + 0 = 8)
            bool negXToPosX = (connectivity & (1L << (1 * 8 + 0))) != 0;
            if (!negXToPosX)
                throw new Exception(
                    $"Visibility_Tunnel: -X should connect to +X through X-axis tunnel. " +
                    $"Connectivity=0x{connectivity:X12}");

            // +X -> +Y should NOT connect (no vertical air path)
            // bit = 0*8 + 4 = 4
            bool posXToPosY = (connectivity & (1L << (0 * 8 + 4))) != 0;
            if (posXToPosY)
                throw new Exception(
                    $"Visibility_Tunnel: +X should NOT connect to +Y (no vertical path). " +
                    $"Connectivity=0x{connectivity:X12}");

            // +X -> +Z should NOT connect
            // bit = 0*8 + 2 = 2
            bool posXToPosZ = (connectivity & (1L << (0 * 8 + 2))) != 0;
            if (posXToPosZ)
                throw new Exception(
                    $"Visibility_Tunnel: +X should NOT connect to +Z (no Z-axis path). " +
                    $"Connectivity=0x{connectivity:X12}");
        }

        // ===== ComputeVisibleSections tests =====

        /// <summary>
        /// Test 4: Camera section is always visible.
        /// Regardless of connectivity, the section containing the camera is always in the visible set.
        /// </summary>
        [TestMethod]
        public void Visibility_CameraSection_AlwaysVisibleTest()
        {
            var cameraSection = new SectionCoord(5, 3, 7);

            // All sections return 0 connectivity (fully blocked) except camera
            var visible = VisibilityGraph.ComputeVisibleSections(
                cameraSection,
                coord => 0L, // everything blocked
                maxDistance: 4);

            if (!visible.Contains(cameraSection))
                throw new Exception(
                    $"Visibility_CameraSection: camera section {cameraSection} must always be visible, " +
                    $"but it's missing from the visible set ({visible.Count} sections visible)");

            // With everything blocked, only the camera section should be visible
            if (visible.Count != 1)
                throw new Exception(
                    $"Visibility_CameraSection: with zero connectivity, only camera section should be visible, " +
                    $"got {visible.Count} sections");
        }

        /// <summary>
        /// Test 5: Section behind solid wall plane is NOT visible via BFS.
        /// Camera at (0,0,0). An infinite wall plane at x=1 (all sections with Cx==1
        /// have zero connectivity). Section at (2,0,0) behind the wall should not be
        /// reachable because BFS cannot route around an infinite plane.
        /// </summary>
        [TestMethod]
        public void Visibility_BehindSolidWall_NotVisibleTest()
        {
            var cameraSection = new SectionCoord(0, 0, 0);
            var behindWall = new SectionCoord(2, 0, 0);

            // Camera section: fully connected (open air)
            // Wall plane at x=1: zero connectivity (solid blocks, infinite in Y and Z)
            // Behind wall: fully connected (open air)
            var visible = VisibilityGraph.ComputeVisibleSections(
                cameraSection,
                coord =>
                {
                    if (coord.Cx == 1) return 0L; // solid wall plane - blocks all paths at x=1
                    return ~0L; // everything else is fully open
                },
                maxDistance: 8);

            if (!visible.Contains(cameraSection))
                throw new Exception("Visibility_BehindWall: camera section must be visible");

            if (visible.Contains(behindWall))
                throw new Exception(
                    $"Visibility_BehindWall: section {behindWall} behind solid wall plane should NOT be visible, " +
                    $"but it was found in the visible set ({visible.Count} total visible)");
        }

        /// <summary>
        /// Test 6: Section reachable through L-shaped tunnel IS visible.
        /// Camera at (0,0,0). Tunnel goes +X then +Z:
        ///   (0,0,0) -> (1,0,0) -> (1,0,1)
        /// Section (1,0,0) connects +X to +Z (L-bend).
        /// Section (1,0,1) should be reachable.
        /// </summary>
        [TestMethod]
        public void Visibility_LShapedTunnel_ReachableTest()
        {
            var cameraSection = new SectionCoord(0, 0, 0);
            var bendSection = new SectionCoord(1, 0, 0);
            var endSection = new SectionCoord(1, 0, 1);

            // Build connectivity masks:
            // Camera: fully open
            // Bend section at (1,0,0): connects -X (face 1, entry from camera) to +Z (face 2)
            //   bit = entryFace(1) * 8 + exitFace(2) = 10
            //   Also need -X to have connections: bit for face 1 must be set
            long bendConnectivity = 0;
            bendConnectivity |= (1L << (1 * 8 + 2)); // -X -> +Z
            bendConnectivity |= (1L << (2 * 8 + 1)); // +Z -> -X (reverse path)

            var visible = VisibilityGraph.ComputeVisibleSections(
                cameraSection,
                coord =>
                {
                    if (coord == cameraSection) return ~0L; // fully open
                    if (coord == bendSection) return bendConnectivity;
                    if (coord == endSection) return ~0L; // fully open
                    return 0L; // everything else is solid
                },
                maxDistance: 8);

            if (!visible.Contains(cameraSection))
                throw new Exception("Visibility_LTunnel: camera section must be visible");

            if (!visible.Contains(bendSection))
                throw new Exception(
                    $"Visibility_LTunnel: bend section {bendSection} should be visible (reachable from camera via +X), " +
                    $"visible set has {visible.Count} sections");

            if (!visible.Contains(endSection))
                throw new Exception(
                    $"Visibility_LTunnel: end section {endSection} should be visible " +
                    $"(reachable through L-shaped tunnel via bend section), " +
                    $"visible set has {visible.Count} sections");
        }

        /// <summary>
        /// Test 7: ComputeConnectivity returns correct mask for known section data.
        /// Build a section with a known 3D air pattern and verify the exact connectivity bits.
        /// Cross-shaped air passage: connects all 4 horizontal faces (+X, -X, +Z, -Z)
        /// but NOT vertical faces (+Y, -Y) since the cross is at a fixed Y level.
        /// </summary>
        [TestMethod]
        public void Visibility_CrossPassage_CorrectMaskTest()
        {
            int size = 8;
            var airPositions = new List<(int x, int y, int z)>();

            // Carve a + shaped passage at y=4:
            // Horizontal bar along X: x=0..7, z=4
            for (int x = 0; x < size; x++)
                airPositions.Add((x, 4, 4));
            // Vertical bar along Z: x=4, z=0..7 (some overlap with X bar at (4,4,4))
            for (int z = 0; z < size; z++)
                airPositions.Add((4, 4, z));

            var blocks = CreateSolidSectionWithAir(size, size, airPositions.ToArray());

            long connectivity = VisibilityGraph.ComputeConnectivity(blocks, size, size);

            // Verify horizontal connections exist:
            // +X <-> -X (through X bar)
            bool posXToNegX = (connectivity & (1L << (0 * 8 + 1))) != 0;
            bool negXToPosX = (connectivity & (1L << (1 * 8 + 0))) != 0;
            if (!posXToNegX || !negXToPosX)
                throw new Exception(
                    $"Visibility_CrossPassage: +X <-> -X should be connected through horizontal bar. " +
                    $"Connectivity=0x{connectivity:X12}");

            // +Z <-> -Z (through Z bar)
            bool posZToNegZ = (connectivity & (1L << (2 * 8 + 3))) != 0;
            bool negZToPosZ = (connectivity & (1L << (3 * 8 + 2))) != 0;
            if (!posZToNegZ || !negZToPosZ)
                throw new Exception(
                    $"Visibility_CrossPassage: +Z <-> -Z should be connected through vertical bar. " +
                    $"Connectivity=0x{connectivity:X12}");

            // Cross connections: +X <-> +Z (through the intersection at (4,4,4))
            bool posXToPosZ = (connectivity & (1L << (0 * 8 + 2))) != 0;
            bool posZToPosX = (connectivity & (1L << (2 * 8 + 0))) != 0;
            if (!posXToPosZ || !posZToPosX)
                throw new Exception(
                    $"Visibility_CrossPassage: +X <-> +Z should be connected through cross intersection. " +
                    $"Connectivity=0x{connectivity:X12}");

            // Vertical faces should NOT connect (air is at a single Y level, no path to top/bottom faces)
            // +Y (face 4) and -Y (face 5) should not connect to any horizontal face
            bool posYToPosX = (connectivity & (1L << (4 * 8 + 0))) != 0;
            bool negYToPosX = (connectivity & (1L << (5 * 8 + 0))) != 0;
            if (posYToPosX)
                throw new Exception(
                    $"Visibility_CrossPassage: +Y should NOT connect to +X (no vertical air path). " +
                    $"Connectivity=0x{connectivity:X12}");
            if (negYToPosX)
                throw new Exception(
                    $"Visibility_CrossPassage: -Y should NOT connect to +X (no vertical air path). " +
                    $"Connectivity=0x{connectivity:X12}");
        }
    }
}
