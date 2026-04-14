# SpawnDev.VoxelEngine v0.1.0 - Foundation

## Goal

Extract the voxel rendering infrastructure that both AubsCraft and Lost Spawns need into a shared library. Both projects currently solve the same problems independently (meshing, buffer management, culling, LOD). Fix once, both benefit.

## Phase 1: Core Meshing (the biggest win)

### 1a. Binary Greedy Meshing Kernels

**Files:** `Meshing/BinaryGreedyMesher.cs`

Three ILGPU kernels:

1. **OccupancyMaskKernel** - Build uint64 columns from block data
   - Input: `ArrayView<short>` blocks (sub-word, half memory)
   - Output: `ArrayView<long>` occupancy columns
   - Dispatch: `[CS_P, CS_P]` - one thread per column
   - Needs neighbor chunk border data (+2 padding)

2. **FaceCullKernel** - Bitwise face culling (64 faces per op)
   - Input: `ArrayView<long>` occupancy columns
   - Output: `ArrayView<long>` face masks (6 per column pair)
   - Dispatch: `[CS_P, CS_P]` - one thread per column pair
   - Pure bitwise ops - perfect GPU work

3. **GreedyMergeKernel** - Per-layer greedy merge
   - Input: `ArrayView<long>` face masks, `ArrayView<short>` blocks (for type comparison)
   - Output: `ArrayView<long>` packed quads (8 bytes each), `ArrayView<int>` counter per face group
   - Dispatch: `[CS, 6]` - one thread per layer per face direction
   - Sequential merge loop per thread, 372 independent invocations
   - Atomic counter with rollback bounds check
   - DelegateSpecialization for block-type comparison (enables per-face texture ID matching)

**Unit tests:**
- Flat plane: 256 quads -> 1 quad (verify merge)
- Checkerboard: no merging possible (verify no false merges)
- L-shape: partial merge (verify correct quad count)
- Chunk boundary: verify padding from neighbors works
- All 6 face directions independently
- Compare GPU output vs CPU reference implementation

### 1b. Face Masking

**Files:** `Meshing/FaceMaskService.cs`

Separate output quads into 6 groups by face normal direction. At draw time, skip groups facing away from camera. 36% measured reduction.

Store `faceGroupOffset[6]` and `faceGroupCount[6]` per chunk.

## Phase 2: Buffer Management

### 2a. Vertex Pool

**Files:** `Buffers/VertexPool.cs`

- Fixed-size bucket allocator backed by one large GPUBuffer
- FIFO free list for O(1) alloc/free
- Each chunk uses 1-6 buckets (one per face group)
- Bucket size configurable (default: enough for worst-case chunk at target LOD)
- Device-adaptive: query `MaxBufferSize`, derive bucket count from actual hardware

### 2b. Indirect Draw Buffer

**Files:** `Buffers/IndirectDrawBuffer.cs`

- ALL draw commands in ONE GPUBuffer (Chrome 300x validation win)
- Layout: `[vertexCount, instanceCount, firstVertex, firstInstance]` x 4 bytes x maxDraws
- Compute kernel writes surviving commands after culling
- Face masking skips back-facing groups by not emitting their draw commands

### 2c. Eviction Manager

**Files:** `Buffers/EvictionManager.cs`

- Priority eviction: `score = distSq * vertexCount`
- Evict-before-discard loop (up to N chunks)
- Compaction as last resort (GPU-copy live data to front)
- Never silently discard

## Phase 3: Culling Pipeline

### 3a. Frustum + Fog Cull Kernel

**Files:** `Culling/FrustumCullKernel.cs`

- ILGPU compute kernel, one thread per chunk
- Fog check first (cheapest), then 6-plane frustum test
- Output: compacted list of visible chunk IDs via atomic counter

### 3b. Graph-Based Occlusion (Sodium-style)

**Files:** `Culling/VisibilityGraph.cs`

- Per-section 48-bit visibility encoding (computed on block change via flood fill)
- BFS traversal from camera section (CPU-side)
- Outward-only constraint, incoming direction accumulation
- Produces candidate set for GPU culling

### 3c. Hi-Z Occlusion

**Files:** `Culling/HiZOcclusionCuller.cs`

- Mip-chain from depth buffer (compute kernel per level)
- Two-phase rendering: known-visible first, then Hi-Z test remaining
- Visibility buffer for frame-to-frame coherence

## Phase 4: LOD System

### 4a. LOD Reduction Kernel

**Files:** `LOD/LODReducer.cs`

- DelegateSpecialization: one mesh kernel, LOD level as parameter
- LOD 0: full resolution, LOD 1: 2x super-blocks, LOD 2: 4x, LOD 3: 8x
- Super-block type = most common non-air in group
- Vertical skirts at LOD boundaries

### 4b. LOD Selection

**Files:** `LOD/LODSelector.cs`

- `SelectLOD(distance, pressure)` returns LOD level per chunk
- Pressure = totalVertices / vertexBudget
- High pressure pushes LOD transitions closer to camera
- Configurable thresholds

### 4c. Streaming Budget

**Files:** `LOD/StreamingBudget.cs`

- Max vertices/uploads per frame (device-adaptive)
- Priority queue: distance + view direction + movement prediction
- Chunk state machine: UNLOADED -> LOADING -> MESHING -> GPU_READY -> VISIBLE

## Phase 5: Adaptive Quality

### 5a. Device Capability Query

**Files:** `Adaptive/DeviceCapabilities.cs`

- Query `MaxBufferSize`, derive vertex budget
- Detect Quest 3S vs desktop automatically
- All limits derived from hardware, none hardcoded

### 5b. Quality Controller

**Files:** `Adaptive/QualityController.cs`

- Frame time EMA with asymmetric rates (fast decrease, slow increase)
- Vertex budget pressure -> LOD bias
- Draw distance shrink/grow feedback loop
- Quality levels: ULTRA -> HIGH -> MEDIUM -> LOW -> MINIMAL
- Thermal throttling detection (frame time spikes without scene change)

## Phase 6: Rendering

### 6a. Reversed-Z Depth

**Files:** `Rendering/ReversedZHelper.cs`

- `depth32float` format, `depthCompare: greater`, `clearValue: 0.0`
- Infinite far plane projection matrix builder
- Zero Z-fighting at any distance

### 6b. Draw Pipeline

**Files:** `Rendering/VoxelDrawPipeline.cs`

- Vertex pulling from storage buffer (8-byte quad format)
- WGSL vertex shader unpacks position + dimensions + face direction
- Shared index buffer for quad triangulation
- Front-to-back ordering via RadixSort on distance

## Implementation Order

1. **Phase 1a** (meshing kernels) - biggest vertex reduction, unblocks everything
2. **Phase 2a+2b** (vertex pool + indirect draws) - replaces both projects' broken buffer management
3. **Phase 3a** (frustum+fog cull) - immediate culling win
4. **Phase 4b+5a** (LOD selection + device caps) - Quest 3S viability
5. **Phase 1b+3b** (face masking + graph occlusion) - further culling
6. **Phase 4a** (LOD reduction kernel) - distance-based detail
7. **Phase 5b** (quality controller) - adaptive feedback loop
8. **Phase 6** (rendering pipeline) - full indirect draw pipeline
9. **Phase 3c** (Hi-Z) - advanced occlusion
10. **Phase 4c** (streaming budget) - smooth loading

## Migration Path

1. Build and test each phase in SpawnDev.VoxelEngine with PlaywrightMultiTest
2. Add ProjectReference from AubsCraft and Lost Spawns
3. Replace project-specific implementations one phase at a time
4. Each replacement is independently testable - no big bang migration
5. When stable, publish NuGet package
