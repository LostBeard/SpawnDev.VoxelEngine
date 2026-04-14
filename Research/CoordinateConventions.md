# Coordinate Conventions

**Author:** Data (Claude CLI #2)
**Date:** 2026-04-14
**Requested by:** Geordi (Sprint 8.5 in the implementation plan)

SpawnDev.VoxelEngine uses three coordinate spaces. The library speaks section-local and section coordinates. World-space conversion is the consumer's responsibility.

---

## 1. Section-Local Coordinates

**Range:** (0, 0, 0) to (SectionSize-1, SectionSize-1, SectionSize-1). Default: (0-15, 0-15, 0-15).

**Used by:** All meshing kernels, face culling, greedy merge, ambient occlusion, raycast, collision.

**Indexing:** Flat array indexed as `blocks[x + z * sizeXZ + y * sizeXZ * sizeXZ]`

- X = horizontal (east/west)
- Y = vertical (up/down) - Y increases upward
- Z = horizontal (north/south)

**Block center:** A block at integer coordinates (x, y, z) occupies the volume from (x, y, z) to (x+1, y+1, z+1). Its center is at (x+0.5, y+0.5, z+0.5).

**Padded coordinates:** For face culling, blocks are padded with a 1-voxel border of neighbor data. Padded coordinates are (x+1, z+1) relative to section-local. Array size: (sizeXZ+2) * (sizeXZ+2) * sizeY.

---

## 2. Section Coordinates (SectionCoord)

**Type:** `SectionCoord { int Cx, int Sy, int Cz }`

**Cx, Cz:** Chunk column indices. World X = Cx * SectionSize * VoxelSize.
**Sy:** Section Y index within the column.

**AubsCraft:** Sy = 0-23 (24 sections, 384 blocks, Y range -64 to 320).
**Lost Spawns:** Sy = 0-31 (32 sections, 256m at 0.5m voxels, Y range 0 to 256).

**Used by:** ChunkManager, IndirectDrawBuffer, CullingPipeline, VisibilityGraph, LOD selection.

**Neighbor lookup:**
- +X neighbor: (Cx+1, Sy, Cz)
- +Y neighbor: (Cx, Sy+1, Cz)
- +Z neighbor: (Cx, Sy, Cz+1)

---

## 3. World Coordinates

**Not used directly by the library.** Consumer responsibility.

**Conversion from section-local to world:**
```
worldX = (Cx * SectionSize + localX) * VoxelSize
worldY = BaseY + (Sy * SectionSize + localY) * VoxelSize
worldZ = (Cz * SectionSize + localZ) * VoxelSize
```

**Conversion from world to section:**
```
voxelX = worldX / VoxelSize
voxelY = (worldY - BaseY) / VoxelSize
voxelZ = worldZ / VoxelSize

Cx = floor(voxelX / SectionSize)
Sy = floor(voxelY / SectionSize)
Cz = floor(voxelZ / SectionSize)

localX = voxelX - Cx * SectionSize
localY = voxelY - Sy * SectionSize
localZ = voxelZ - Cz * SectionSize
```

**SectionCoord helper methods:**
- `WorldMin(voxelSize, sectionSize, baseY)` - AABB minimum corner
- `WorldMax(voxelSize, sectionSize, baseY)` - AABB maximum corner

---

## 4. PackedQuad Coordinates

**Section-local:** X, Y, Z are 4 bits each (0-15). Width and height are 4 bits (1-16 stored as 0-15).

The consuming renderer adds the section's world offset to produce final vertex positions:
```
vertexWorldPos = sectionWorldMin + quadLocalPos * voxelSize
```

---

## 5. Face Direction Convention

| Index | Direction | Normal |
|-------|-----------|--------|
| 0 | +X | (1, 0, 0) |
| 1 | -X | (-1, 0, 0) |
| 2 | +Z | (0, 0, 1) |
| 3 | -Z | (0, 0, -1) |
| 4 | +Y | (0, 1, 0) |
| 5 | -Y | (0, -1, 0) |

Defined in `VoxelMeshConstants`. Used consistently across all meshing, culling, and face masking code.

---

## 6. Raycast Coordinates

`VoxelRaycast.Cast()` operates in section-local space. The caller converts world-space rays:
```
localOrigin = (worldOrigin - sectionWorldMin) / voxelSize
localDir = worldDir  // direction doesn't change, only origin shifts
```

`VoxelRaycast.CastWorld()` handles this conversion internally, walking across section boundaries.

`RaycastHit.AdjacentX/Y/Z` is the block position where a new block would be placed (one step along the hit face normal from the hit block).
