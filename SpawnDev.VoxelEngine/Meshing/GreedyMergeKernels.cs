using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// GPU kernel for greedy quad merging.
    /// Takes face masks from FaceCullKernels and produces merged quads.
    ///
    /// Dispatch: [maxMergeLayer, 6] where maxMergeLayer = max(height, chunkXZ).
    /// For faces 0-3 (+X,-X,+Z,-Z): one thread per layer along the face normal.
    /// For faces 4-5 (+Y,-Y): one thread per XZ column, iterates Y internally.
    ///   This eliminates the race condition where multiple Y-layer threads
    ///   would read/write the same faceMask column concurrently.
    ///
    /// Output: packed quads via atomic counter.
    /// </summary>
    public static class GreedyMergeKernels
    {
        /// <summary>
        /// Greedy merge kernel - processes one slice of one face direction.
        ///
        /// For faces 0-3: index.X = layer along face normal (0 to chunkXZ-1).
        /// For faces 4-5: index.X = flattened XZ column index (0 to chunkXZ*chunkXZ-1).
        ///   Each thread owns one column and iterates all Y layers sequentially.
        /// </summary>
        public static void GreedyMergeKernel(
            Index2D index,
            ArrayView<long> faceMasks,
            ArrayView<int> paddedBlocks,
            ArrayView<long> outputQuads,
            ArrayView<int> quadCounter,
            int chunkXZ,
            int height,
            int paddedXZ,
            int faceStart)
        {
            int threadIdx = index.X;
            int face = index.Y + faceStart;

            if (face >= 6) return;

            int innerCount = chunkXZ * chunkXZ;
            int stride = paddedXZ * paddedXZ;

            if (face >= 4)
            {
                // +Y or -Y face: one thread per Y layer.
                // Full XZ-plane greedy merge. Race-free because faces 4-5 are
                // dispatched SEPARATELY from faces 0-3 (second dispatch in VoxelMeshPipeline).
                // Each Y-layer thread only touches bit 'threadIdx' in the i64 masks.
                // No concurrent Y-layer threads on the same mask entry.
                if (threadIdx >= height) return;
                MergeXZPlane(faceMasks, paddedBlocks, outputQuads, quadCounter,
                    chunkXZ, height, paddedXZ, innerCount, stride, threadIdx, face);
            }
            else
            {
                // +X,-X,+Z,-Z: one thread per layer along the face normal.
                if (threadIdx >= chunkXZ) return;
                MergePerpendicularPlane(faceMasks, paddedBlocks, outputQuads, quadCounter,
                    chunkXZ, height, paddedXZ, innerCount, stride, threadIdx, face);
            }
        }

        /// <summary>
        /// Merge +Y/-Y faces in the XZ plane for a specific Y layer.
        /// This kernel is dispatched SEPARATELY from faces 0-3 (in a second dispatch)
        /// so there's no concurrent access from other Y-layer threads.
        /// One thread per Y layer, full XZ-plane greedy merge, race-free.
        /// </summary>
        private static void MergeXZPlane(
            ArrayView<long> faceMasks,
            ArrayView<int> paddedBlocks,
            ArrayView<long> outputQuads,
            ArrayView<int> quadCounter,
            int chunkXZ, int height, int paddedXZ,
            int innerCount, int stride,
            int yLayer, int face)
        {
            int faceOffset = face * innerCount;

            for (int z = 0; z < chunkXZ; z++)
            {
                int x = 0;
                while (x < chunkXZ)
                {
                    int innerIdx = x + z * chunkXZ;
                    long mask = faceMasks[faceOffset + innerIdx];

                    if ((mask & (1L << yLayer)) == 0)
                    {
                        x++;
                        continue;
                    }

                    int blockType = paddedBlocks[(x + 1) + (z + 1) * paddedXZ + yLayer * stride] & 0xFFF;

                    // Extend width (rightward along x)
                    int w = 1;
                    while (x + w < chunkXZ)
                    {
                        int nextInnerIdx = (x + w) + z * chunkXZ;
                        long nextMask = faceMasks[faceOffset + nextInnerIdx];
                        if ((nextMask & (1L << yLayer)) == 0) break;
                        int nextType = paddedBlocks[(x + w + 1) + (z + 1) * paddedXZ + yLayer * stride] & 0xFFF;
                        if (nextType != blockType) break;
                        w++;
                    }

                    // Extend height (downward along z)
                    int h = 1;
                    bool canExtend = true;
                    while (z + h < chunkXZ && canExtend)
                    {
                        for (int dx = 0; dx < w; dx++)
                        {
                            int checkInnerIdx = (x + dx) + (z + h) * chunkXZ;
                            long checkMask = faceMasks[faceOffset + checkInnerIdx];
                            if ((checkMask & (1L << yLayer)) == 0) { canExtend = false; break; }
                            int checkType = paddedBlocks[(x + dx + 1) + (z + h + 1) * paddedXZ + yLayer * stride] & 0xFFF;
                            if (checkType != blockType) { canExtend = false; break; }
                        }
                        if (canExtend) h++;
                    }

                    // Clear consumed bits atomically - multiple Y-layer threads may
                    // touch the same i64 mask (each clears a different bit).
                    // Atomic.And ensures no bit clears are lost.
                    for (int dz = 0; dz < h; dz++)
                        for (int dx = 0; dx < w; dx++)
                            Atomic.And(ref faceMasks[faceOffset + (x + dx) + (z + dz) * chunkXZ], ~(1L << yLayer));

                    int slot = Atomic.Add(ref quadCounter[0], 1);
                    if (slot >= outputQuads.IntLength)
                    {
                        Atomic.Add(ref quadCounter[0], -1);
                    }
                    else
                    {
                        outputQuads[slot] = PackedQuad.Pack(x, yLayer, z, w, h, face, blockType);
                    }

                    x += w;
                }
            }
        }

        /// <summary>
        /// Merge faces in the perpendicular plane for +X,-X,+Z,-Z faces.
        /// The 'layer' is the position along the face's normal axis.
        /// </summary>
        private static void MergePerpendicularPlane(
            ArrayView<long> faceMasks,
            ArrayView<int> paddedBlocks,
            ArrayView<long> outputQuads,
            ArrayView<int> quadCounter,
            int chunkXZ, int height, int paddedXZ,
            int innerCount, int stride,
            int layer, int face)
        {
            int faceOffset = face * innerCount;

            // For +X/-X (face 0,1): normal is X axis, layer = X position
            //   Slice is YZ plane. Iterate z (outer), scan y via bits.
            // For +Z/-Z (face 2,3): normal is Z axis, layer = Z position
            //   Slice is YX plane. Iterate x (outer), scan y via bits.

            bool isXFace = face <= 1;

            int outerSize = chunkXZ; // z for X-faces, x for Z-faces

            for (int outer = 0; outer < outerSize; outer++)
            {
                // Get the face mask column for this (layer, outer) pair
                int innerX, innerZ;
                if (isXFace)
                {
                    innerX = layer;
                    innerZ = outer;
                }
                else
                {
                    innerX = outer;
                    innerZ = layer;
                }

                int innerIdx = innerX + innerZ * chunkXZ;
                long mask = faceMasks[faceOffset + innerIdx];
                if (mask == 0) continue;

                // Scan bits (Y positions) in this column
                while (mask != 0)
                {
                    // Find lowest set bit = starting Y position
                    int y = TrailingZeros(mask);
                    if (y >= height) break;

                    // Get block type
                    int px = (isXFace ? layer : outer) + 1;
                    int pz = (isXFace ? outer : layer) + 1;
                    int blockType = paddedBlocks[px + pz * paddedXZ + y * stride] & 0xFFF;

                    // Extend height along Y (rightward in bit terms)
                    int h = 1;
                    while (y + h < height && (mask & (1L << (y + h))) != 0)
                    {
                        int nextType = paddedBlocks[px + pz * paddedXZ + (y + h) * stride] & 0xFFF;
                        if (nextType != blockType) break;
                        h++;
                    }

                    // Clear consumed bits
                    for (int dy = 0; dy < h; dy++)
                    {
                        mask &= ~(1L << (y + dy));
                    }

                    // Update the face mask in global memory
                    faceMasks[faceOffset + innerIdx] = mask;

                    // Extend width along the outer axis (z for X-faces, x for Z-faces)
                    int w = 1;
                    while (outer + w < outerSize)
                    {
                        int nextInnerX, nextInnerZ;
                        if (isXFace)
                        {
                            nextInnerX = layer;
                            nextInnerZ = outer + w;
                        }
                        else
                        {
                            nextInnerX = outer + w;
                            nextInnerZ = layer;
                        }

                        int nextInnerIdx = nextInnerX + nextInnerZ * chunkXZ;
                        long nextMask = faceMasks[faceOffset + nextInnerIdx];

                        // Check that all h bits match starting at y
                        bool allMatch = true;
                        for (int dy = 0; dy < h; dy++)
                        {
                            if ((nextMask & (1L << (y + dy))) == 0) { allMatch = false; break; }
                            int npx = (isXFace ? layer : outer + w) + 1;
                            int npz = (isXFace ? outer + w : layer) + 1;
                            int nextType = paddedBlocks[npx + npz * paddedXZ + (y + dy) * stride] & 0xFFF;
                            if (nextType != blockType) { allMatch = false; break; }
                        }

                        if (!allMatch) break;

                        // Clear consumed bits in the neighbor column
                        for (int dy = 0; dy < h; dy++)
                        {
                            nextMask &= ~(1L << (y + dy));
                        }
                        faceMasks[faceOffset + nextInnerIdx] = nextMask;
                        w++;
                    }

                    // Emit quad
                    int qx, qy, qz, qw, qh;
                    if (isXFace)
                    {
                        qx = layer; qy = y; qz = outer; qw = w; qh = h;
                    }
                    else
                    {
                        qx = outer; qy = y; qz = layer; qw = w; qh = h;
                    }

                    int slot = Atomic.Add(ref quadCounter[0], 1);
                    if (slot >= outputQuads.IntLength)
                    {
                        Atomic.Add(ref quadCounter[0], -1);
                    }
                    else
                    {
                        outputQuads[slot] = PackedQuad.Pack(qx, qy, qz, qw, qh, face, blockType);
                    }
                }
            }
        }

        /// <summary>
        /// Count trailing zeros in a long (find position of lowest set bit).
        /// </summary>
        private static int TrailingZeros(long value)
        {
            if (value == 0) return 64;
            int count = 0;
            long v = value;
            while ((v & 1) == 0)
            {
                count++;
                v >>= 1;
            }
            return count;
        }
    }
}
