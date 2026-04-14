using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// GPU kernel for greedy quad merging.
    /// Takes face masks from FaceCullKernels and produces merged quads.
    ///
    /// Dispatch: [chunkXZ, 6] - one thread per layer per face direction.
    /// Each thread runs a sequential greedy merge loop for its slice.
    /// Output: packed quads via atomic counter.
    /// </summary>
    public static class GreedyMergeKernels
    {
        /// <summary>
        /// Greedy merge kernel - processes one layer of one face direction.
        ///
        /// For faces 0-3 (+X,-X,+Z,-Z): the "layer" is the position along the face's normal axis.
        /// The 2D slice being merged is the perpendicular plane.
        ///
        /// For faces 4-5 (+Y,-Y): the "layer" is the Y position.
        /// The 2D slice being merged is the XZ plane.
        ///
        /// Each thread scans bit rows, extends runs rightward (same type),
        /// then extends downward (matching runs), producing merged quads.
        /// </summary>
        public static void GreedyMergeKernel(
            Index2D index,
            ArrayView<long> faceMasks,
            ArrayView<int> paddedBlocks,
            ArrayView<long> outputQuads,
            ArrayView<int> quadCounter,
            int chunkXZ,
            int height,
            int paddedXZ)
        {
            int layer = index.X;
            int face = index.Y;

            if (face >= 6) return;

            int innerCount = chunkXZ * chunkXZ;
            int stride = paddedXZ * paddedXZ;

            // For +Y/-Y faces, layer is Y position (0 to height-1)
            // For other faces, layer is the position along the face's axis (0 to chunkXZ-1)
            int maxLayer = (face >= 4) ? height : chunkXZ;
            if (layer >= maxLayer) return;

            // Process one row at a time within this layer's slice
            // For +Y/-Y: iterate over x,z pairs. faceMask bit = Y position.
            // For +X/-X: iterate over z,y pairs. faceMask bit = Y position.
            // For +Z/-Z: iterate over x,y pairs. faceMask bit = Y position.

            // The face masks store one uint64 per (innerX, innerZ) column.
            // Each bit represents a Y position where that face is visible.

            // For +Y/-Y faces at a specific Y layer:
            // We need to check if bit 'layer' is set in each column's face mask.
            // Then greedily merge adjacent set bits in the XZ plane.

            if (face >= 4)
            {
                // +Y or -Y face: merge in XZ plane at Y=layer
                MergeXZPlane(faceMasks, paddedBlocks, outputQuads, quadCounter,
                    chunkXZ, height, paddedXZ, innerCount, stride, layer, face);
            }
            else
            {
                // +X,-X,+Z,-Z: merge in the plane perpendicular to the face normal at position=layer
                MergePerpendicularPlane(faceMasks, paddedBlocks, outputQuads, quadCounter,
                    chunkXZ, height, paddedXZ, innerCount, stride, layer, face);
            }
        }

        /// <summary>
        /// Merge faces in the XZ plane for +Y/-Y faces at a specific Y layer.
        /// Scans each row (x-axis), extends rightward then downward.
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
            // Build a 2D visited mask for this slice
            // We use the face mask bits directly - extract bit 'yLayer' from each column
            // visited[z * chunkXZ + x] = true if already consumed by a quad
            // Since we can't allocate dynamic arrays in ILGPU, we process row by row
            // and track consumed columns via bit clearing

            int faceOffset = face * innerCount;

            for (int z = 0; z < chunkXZ; z++)
            {
                int x = 0;
                while (x < chunkXZ)
                {
                    int innerIdx = x + z * chunkXZ;
                    long mask = faceMasks[faceOffset + innerIdx];

                    // Check if this face is visible at yLayer
                    if ((mask & (1L << yLayer)) == 0)
                    {
                        x++;
                        continue;
                    }

                    // Get block type at this position (padded coords: x+1, z+1, yLayer)
                    int blockType = paddedBlocks[(x + 1) + (z + 1) * paddedXZ + yLayer * stride];

                    // Extend width (rightward along x)
                    int w = 1;
                    while (x + w < chunkXZ)
                    {
                        int nextInnerIdx = (x + w) + z * chunkXZ;
                        long nextMask = faceMasks[faceOffset + nextInnerIdx];
                        if ((nextMask & (1L << yLayer)) == 0) break;

                        int nextType = paddedBlocks[(x + w + 1) + (z + 1) * paddedXZ + yLayer * stride];
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

                            int checkType = paddedBlocks[(x + dx + 1) + (z + h + 1) * paddedXZ + yLayer * stride];
                            if (checkType != blockType) { canExtend = false; break; }
                        }
                        if (canExtend) h++;
                    }

                    // Clear consumed bits from face masks
                    for (int dz = 0; dz < h; dz++)
                    {
                        for (int dx = 0; dx < w; dx++)
                        {
                            int clearIdx = faceOffset + (x + dx) + (z + dz) * chunkXZ;
                            faceMasks[clearIdx] &= ~(1L << yLayer);
                        }
                    }

                    // Emit quad with atomic counter + rollback bounds check
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
                    int blockType = paddedBlocks[px + pz * paddedXZ + y * stride];

                    // Extend height along Y (rightward in bit terms)
                    int h = 1;
                    while (y + h < height && (mask & (1L << (y + h))) != 0)
                    {
                        int nextType = paddedBlocks[px + pz * paddedXZ + (y + h) * stride];
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
                            int nextType = paddedBlocks[npx + npz * paddedXZ + (y + dy) * stride];
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
