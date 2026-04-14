using System.Numerics;

namespace SpawnDev.VoxelEngine.Culling
{
    /// <summary>
    /// Graph-based visibility determination (Sodium-style).
    ///
    /// Each section has a 48-bit connectivity mask: 6 faces * 8 sub-regions per face.
    /// A sub-region is "connected" if light/sight can pass through it (the section
    /// has a path from that face sub-region to at least one other face).
    ///
    /// BFS from the camera's section outward through connected faces determines which
    /// sections are potentially visible. Sections not reachable = definitely occluded
    /// (underground caves, enclosed rooms).
    ///
    /// This runs on CPU before GPU frustum culling. It eliminates entire underground
    /// networks from the GPU culling pass, saving both bandwidth and compute.
    ///
    /// Reference: CaffeineMC/sodium-fabric VisibilityGraph.
    /// </summary>
    public class VisibilityGraph
    {
        /// <summary>
        /// Face indices matching VoxelMeshConstants.
        /// </summary>
        private const int PosX = 0, NegX = 1, PosZ = 2, NegZ = 3, PosY = 4, NegY = 5;

        /// <summary>
        /// Opposite face lookup. When entering through face F, we entered the neighbor
        /// through the opposite face.
        /// </summary>
        private static readonly int[] OppositeFace = { NegX, PosX, NegZ, PosZ, NegY, PosY };

        /// <summary>
        /// Direction offsets for each face: (dx, dy, dz) in section coordinates.
        /// </summary>
        private static readonly (int dx, int dy, int dz)[] FaceOffsets =
        {
            (1, 0, 0),   // +X
            (-1, 0, 0),  // -X
            (0, 0, 1),   // +Z
            (0, 0, -1),  // -Z
            (0, 1, 0),   // +Y
            (0, -1, 0),  // -Y
        };

        /// <summary>
        /// Perform BFS visibility traversal from the camera's section.
        /// Returns the set of section coordinates that are potentially visible.
        ///
        /// Sections are visible if reachable from the camera via connected faces,
        /// AND the traversal direction is outward-only (away from camera to prevent
        /// visiting sections behind the camera through indirect paths).
        /// </summary>
        /// <param name="cameraSectionCoord">Section containing the camera.</param>
        /// <param name="getConnectivity">Returns the 48-bit connectivity for a section, or -1 if not loaded.</param>
        /// <param name="maxDistance">Maximum BFS distance in sections.</param>
        /// <returns>Set of visible section coordinates.</returns>
        public static HashSet<SectionCoord> ComputeVisibleSections(
            SectionCoord cameraSectionCoord,
            Func<SectionCoord, long> getConnectivity,
            int maxDistance = 16)
        {
            var visible = new HashSet<SectionCoord>();
            var queue = new Queue<(SectionCoord coord, int entryFace, int depth)>();

            // Camera section is always visible, entered from "all faces"
            visible.Add(cameraSectionCoord);

            // Seed BFS: try all 6 directions from camera section
            long camConn = getConnectivity(cameraSectionCoord);
            if (camConn < 0) camConn = ~0L; // not loaded = assume fully connected

            for (int face = 0; face < 6; face++)
            {
                // Check if camera section connects through this face
                if (!HasConnection(camConn, face))
                    continue;

                var neighbor = GetNeighbor(cameraSectionCoord, face);
                if (visible.Contains(neighbor))
                    continue;

                queue.Enqueue((neighbor, OppositeFace[face], 1));
            }

            // BFS traversal
            while (queue.Count > 0)
            {
                var (coord, entryFace, depth) = queue.Dequeue();

                if (depth > maxDistance)
                    continue;

                if (!visible.Add(coord))
                    continue; // already visited

                long conn = getConnectivity(coord);
                if (conn < 0) conn = ~0L; // not loaded = assume fully connected

                // Check which faces this section connects the entry face to
                for (int exitFace = 0; exitFace < 6; exitFace++)
                {
                    if (exitFace == entryFace)
                        continue; // don't go back the way we came

                    // Check connectivity: can sight pass from entryFace to exitFace?
                    if (!HasFaceToFaceConnection(conn, entryFace, exitFace))
                        continue;

                    var neighbor = GetNeighbor(coord, exitFace);
                    if (visible.Contains(neighbor))
                        continue;

                    queue.Enqueue((neighbor, OppositeFace[exitFace], depth + 1));
                }
            }

            return visible;
        }

        /// <summary>
        /// Compute connectivity mask for a section from its block data.
        /// For each pair of faces (entryFace, exitFace), flood fill from one face's
        /// sub-regions through air blocks to determine if they connect.
        ///
        /// The 48-bit mask encodes: for each of 6 faces, 8 bits indicating which
        /// other faces are reachable through air blocks.
        /// Bit layout: face * 8 + otherFace (with 2 spare bits per face).
        /// </summary>
        public static long ComputeConnectivity(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY)
        {
            long connectivity = 0;

            // For each face pair, check if air connects them via flood fill
            for (int entryFace = 0; entryFace < 6; entryFace++)
            {
                // Flood fill from all air blocks on the entry face
                var reachable = FloodFillFromFace(blocks, sizeXZ, sizeY, entryFace);

                for (int exitFace = 0; exitFace < 6; exitFace++)
                {
                    if (exitFace == entryFace) continue;

                    // Check if any air block on the exit face was reached
                    if (FaceReached(reachable, sizeXZ, sizeY, exitFace))
                    {
                        int bit = entryFace * 8 + exitFace;
                        connectivity |= (1L << bit);
                    }
                }
            }

            return connectivity;
        }

        /// <summary>Check if a face has any outgoing connection (any bit set for that face).</summary>
        private static bool HasConnection(long connectivity, int face)
        {
            // Any of the 8 bits for this face set?
            long faceMask = 0xFFL << (face * 8);
            return (connectivity & faceMask) != 0;
        }

        /// <summary>Check if entry face connects to exit face.</summary>
        private static bool HasFaceToFaceConnection(long connectivity, int entryFace, int exitFace)
        {
            int bit = entryFace * 8 + exitFace;
            return (connectivity & (1L << bit)) != 0;
        }

        /// <summary>Get neighbor section coordinate in the given face direction.</summary>
        private static SectionCoord GetNeighbor(SectionCoord coord, int face)
        {
            var (dx, dy, dz) = FaceOffsets[face];
            return new SectionCoord(coord.Cx + dx, coord.Sy + dy, coord.Cz + dz);
        }

        /// <summary>
        /// Flood fill from all air blocks on one face of the section.
        /// Returns a visited array (true = reachable from the entry face via air).
        /// </summary>
        private static bool[] FloodFillFromFace(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY, int face)
        {
            int total = sizeXZ * sizeY * sizeXZ;
            var visited = new bool[total];
            var queue = new Queue<int>();

            // Seed: all air blocks on the entry face
            EnqueueFaceBlocks(blocks, sizeXZ, sizeY, face, visited, queue);

            // BFS through air blocks
            while (queue.Count > 0)
            {
                int idx = queue.Dequeue();
                int x = idx % sizeXZ;
                int z = (idx / sizeXZ) % sizeXZ;
                int y = idx / (sizeXZ * sizeXZ);

                // Check 6 neighbors
                TryEnqueue(blocks, sizeXZ, sizeY, x + 1, y, z, visited, queue);
                TryEnqueue(blocks, sizeXZ, sizeY, x - 1, y, z, visited, queue);
                TryEnqueue(blocks, sizeXZ, sizeY, x, y + 1, z, visited, queue);
                TryEnqueue(blocks, sizeXZ, sizeY, x, y - 1, z, visited, queue);
                TryEnqueue(blocks, sizeXZ, sizeY, x, y, z + 1, visited, queue);
                TryEnqueue(blocks, sizeXZ, sizeY, x, y, z - 1, visited, queue);
            }

            return visited;
        }

        /// <summary>Seed the queue with all air blocks on a specific face of the section.</summary>
        private static void EnqueueFaceBlocks(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY, int face,
            bool[] visited, Queue<int> queue)
        {
            // Iterate the 2D grid of blocks on this face
            for (int a = 0; a < sizeXZ; a++)
            {
                int bMax = face >= 4 ? sizeXZ : sizeY; // Y faces iterate XZ, other faces iterate height
                for (int b = 0; b < bMax; b++)
                {
                    int x, y, z;
                    switch (face)
                    {
                        case PosX: x = sizeXZ - 1; y = b; z = a; break;
                        case NegX: x = 0; y = b; z = a; break;
                        case PosZ: x = a; y = b; z = sizeXZ - 1; break;
                        case NegZ: x = a; y = b; z = 0; break;
                        case PosY: x = a; y = sizeY - 1; z = b; break;
                        case NegY: x = a; y = 0; z = b; break;
                        default: continue;
                    }

                    int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                    if (PackedBlock.IsAir(blocks[idx]) && !visited[idx])
                    {
                        visited[idx] = true;
                        queue.Enqueue(idx);
                    }
                }
            }
        }

        /// <summary>Check if any block on a specific face was reached by flood fill.</summary>
        private static bool FaceReached(bool[] visited, int sizeXZ, int sizeY, int face)
        {
            for (int a = 0; a < sizeXZ; a++)
            {
                int bMax = face >= 4 ? sizeXZ : sizeY;
                for (int b = 0; b < bMax; b++)
                {
                    int x, y, z;
                    switch (face)
                    {
                        case PosX: x = sizeXZ - 1; y = b; z = a; break;
                        case NegX: x = 0; y = b; z = a; break;
                        case PosZ: x = a; y = b; z = sizeXZ - 1; break;
                        case NegZ: x = a; y = b; z = 0; break;
                        case PosY: x = a; y = sizeY - 1; z = b; break;
                        case NegY: x = a; y = 0; z = b; break;
                        default: continue;
                    }

                    int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
                    if (visited[idx]) return true;
                }
            }
            return false;
        }

        private static void TryEnqueue(ReadOnlySpan<int> blocks, int sizeXZ, int sizeY,
            int x, int y, int z, bool[] visited, Queue<int> queue)
        {
            if ((uint)x >= (uint)sizeXZ || (uint)y >= (uint)sizeY || (uint)z >= (uint)sizeXZ)
                return;

            int idx = x + z * sizeXZ + y * sizeXZ * sizeXZ;
            if (visited[idx]) return;
            if (!PackedBlock.IsAir(blocks[idx])) return;

            visited[idx] = true;
            queue.Enqueue(idx);
        }
    }
}
