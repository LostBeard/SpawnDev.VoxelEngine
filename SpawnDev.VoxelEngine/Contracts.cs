using System.Numerics;
using System.Runtime.InteropServices;

namespace SpawnDev.VoxelEngine
{
    /// <summary>
    /// Section coordinates identifying a 16x16x16 section in the world.
    /// Cx/Cz are chunk column indices, Sy is the section Y index within the column.
    /// AubsCraft: Sy = 0-23 (384 blocks / 16 = 24 sections per column, Y range -64 to 320).
    /// Lost Spawns: TBD, same section-based architecture.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SectionCoord : IEquatable<SectionCoord>
    {
        public int Cx;
        public int Sy;
        public int Cz;

        public SectionCoord(int cx, int sy, int cz)
        {
            Cx = cx;
            Sy = sy;
            Cz = cz;
        }

        /// <summary>World-space AABB minimum for this section at the given voxel size.</summary>
        public Vector3 WorldMin(float voxelSize, int sectionSize = 16, float baseY = -64f)
        {
            return new Vector3(
                Cx * sectionSize * voxelSize,
                baseY + Sy * sectionSize * voxelSize,
                Cz * sectionSize * voxelSize);
        }

        /// <summary>World-space AABB maximum for this section.</summary>
        public Vector3 WorldMax(float voxelSize, int sectionSize = 16, float baseY = -64f)
        {
            return WorldMin(voxelSize, sectionSize, baseY) + new Vector3(sectionSize * voxelSize);
        }

        public bool Equals(SectionCoord other) => Cx == other.Cx && Sy == other.Sy && Cz == other.Cz;
        public override bool Equals(object? obj) => obj is SectionCoord s && Equals(s);
        public override int GetHashCode() => HashCode.Combine(Cx, Sy, Cz);
        public override string ToString() => $"({Cx}, {Sy}, {Cz})";
        public static bool operator ==(SectionCoord a, SectionCoord b) => a.Equals(b);
        public static bool operator !=(SectionCoord a, SectionCoord b) => !a.Equals(b);
    }

    /// <summary>
    /// Entry in the section registry - one per loaded section.
    /// Contains the section's location and its mesh data location in the shared quad buffer.
    /// Consumed by culling (frustum test against section AABB) and rendering (draw command generation).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SectionEntry
    {
        /// <summary>Section coordinates in the world grid.</summary>
        public SectionCoord Coord;

        /// <summary>Offset into the shared PackedQuad buffer where this section's quads start.</summary>
        public int QuadOffset;

        /// <summary>Number of quads in this section's mesh.</summary>
        public int QuadCount;

        /// <summary>Per-face quad counts for face masking (skip back-facing groups).</summary>
        public int QuadCountPosX;
        public int QuadCountNegX;
        public int QuadCountPosZ;
        public int QuadCountNegZ;
        public int QuadCountPosY;
        public int QuadCountNegY;

        /// <summary>
        /// 48-bit connectivity bitmask for cave culling (Sprint 5).
        /// 6 faces * 8 sub-regions per face = 48 bits.
        /// Bit set = this sub-region connects to an adjacent section's matching sub-region.
        /// Used by BFS traversal to determine which underground sections are reachable from surface.
        /// Zero until cave culling is implemented - defaults to "fully connected" (all bits set).
        /// </summary>
        public long Connectivity;

        /// <summary>Distance squared from camera for draw ordering. Updated each frame.</summary>
        public float DistanceSq;

        /// <summary>LOD level for this section (0 = full detail). Set by LOD selector.</summary>
        public int LodLevel;
    }

    /// <summary>
    /// Output from the culling pipeline - indices into the SectionEntry array
    /// for sections that passed frustum + visibility culling.
    /// GPU buffer: ArrayView of int (indices) + atomic counter for visible count.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CullResult
    {
        /// <summary>Number of visible sections after culling.</summary>
        public int VisibleCount;

        /// <summary>
        /// Maximum capacity for visible section indices.
        /// The actual indices are stored in a separate ArrayView&lt;int&gt; buffer.
        /// </summary>
        public int Capacity;
    }

    /// <summary>
    /// Indirect draw command matching WebGPU's drawIndirect layout.
    /// One command per visible section (or per visible face group with face masking).
    /// Packed into a single GPUBuffer for indirect drawing - one submit, all sections.
    ///
    /// Layout matches GPUDrawIndirectParameters:
    ///   vertexCount (u32) - number of vertices to draw (quadCount * 6)
    ///   instanceCount (u32) - always 1
    ///   firstVertex (u32) - offset into vertex buffer (quadOffset * 6)
    ///   firstInstance (u32) - section index (for per-instance data lookup)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DrawCommand
    {
        /// <summary>Number of vertices to draw (QuadCount * 6 for 2 triangles per quad).</summary>
        public uint VertexCount;

        /// <summary>Instance count. Always 1 for section draws.</summary>
        public uint InstanceCount;

        /// <summary>First vertex offset (QuadOffset * 6 into the vertex-pulled quad buffer).</summary>
        public uint FirstVertex;

        /// <summary>Section index for per-instance data lookup in the vertex shader.</summary>
        public uint FirstInstance;

        /// <summary>Create a draw command for a section's quads.</summary>
        public static DrawCommand FromSection(int quadOffset, int quadCount, int sectionIndex)
        {
            return new DrawCommand
            {
                VertexCount = (uint)(quadCount * 6),
                InstanceCount = 1,
                FirstVertex = (uint)(quadOffset * 6),
                FirstInstance = (uint)sectionIndex,
            };
        }

        /// <summary>Create a draw command for a specific face group within a section.</summary>
        public static DrawCommand FromFaceGroup(int quadOffset, int quadCount, int sectionIndex)
        {
            return FromSection(quadOffset, quadCount, sectionIndex);
        }
    }

    /// <summary>
    /// Result from a voxel raycast (DDA traversal).
    /// Contains the hit position, face, block type, and traversal distance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct RaycastHit
    {
        /// <summary>Whether the ray hit a solid block within max distance.</summary>
        public bool DidHit;

        /// <summary>World-space position of the hit point on the block face.</summary>
        public Vector3 HitPosition;

        /// <summary>Section-local block coordinates that were hit (0-15 each).</summary>
        public int BlockX;
        public int BlockY;
        public int BlockZ;

        /// <summary>Face direction that was hit (0-5, matching VoxelMeshConstants face constants).</summary>
        public int HitFace;

        /// <summary>Block type at the hit position (12-bit PackedBlock type).</summary>
        public int BlockType;

        /// <summary>Damage level at the hit position (4-bit PackedBlock damage).</summary>
        public int DamageLevel;

        /// <summary>Distance from ray origin to hit point.</summary>
        public float Distance;

        /// <summary>
        /// Block coordinates ADJACENT to the hit face (where a new block would be placed).
        /// Computed as: hit block + face normal direction.
        /// </summary>
        public int AdjacentX;
        public int AdjacentY;
        public int AdjacentZ;

        /// <summary>No hit result.</summary>
        public static readonly RaycastHit None = new() { DidHit = false };
    }

    /// <summary>
    /// Configuration for the voxel engine instance.
    /// Set once at initialization, consumed by all subsystems.
    /// </summary>
    public class VoxelEngineConfig
    {
        /// <summary>Size of one voxel in world units. 0.5 = Lost Spawns, 1.0 = AubsCraft.</summary>
        public float VoxelSize { get; set; } = 1.0f;

        /// <summary>Section size along all 3 axes (cubic sections). Always 16.</summary>
        public int SectionSize { get; set; } = 16;

        /// <summary>Maximum draw distance in sections. Affects vertex budget and chunk loading.</summary>
        public int DrawDistance { get; set; } = 16;

        /// <summary>Maximum number of sections that can be loaded simultaneously.</summary>
        public int MaxLoadedSections { get; set; } = 4096;

        /// <summary>Maximum number of quads across all loaded sections.</summary>
        public int MaxTotalQuads { get; set; } = 1_000_000;

        /// <summary>Base Y coordinate in world space (AubsCraft: -64, Lost Spawns: 0).</summary>
        public float BaseY { get; set; } = -64f;

        /// <summary>Number of vertical sections per column (AubsCraft: 24 for 384 blocks).</summary>
        public int SectionsPerColumn { get; set; } = 24;
    }
}
