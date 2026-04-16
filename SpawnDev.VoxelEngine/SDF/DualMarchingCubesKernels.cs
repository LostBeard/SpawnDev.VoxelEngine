using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.SDF
{
    /// <summary>
    /// GPU kernels for Dual Marching Cubes mesh generation from SDF data.
    ///
    /// Pipeline:
    /// 1. ClassifyActiveCells - mark cells that cross the isosurface
    /// 2. (External) Prefix sum for output compaction
    /// 3. GenerateVertices - produce dual vertices and quads for active cells
    ///
    /// DMC advantages over standard MC:
    /// - 150x vertex reduction in favorable cases
    /// - Inherently crack-free at LOD boundaries
    /// - Quad output (not triangle) - better for GPU rasterization
    /// </summary>
    public static class DualMarchingCubesKernels
    {
        /// <summary>
        /// Classify cells as active (crossing the isosurface) or inactive.
        /// A cell is active if not all 8 corners have the same sign.
        ///
        /// Dispatch: one thread per cell (chunkSize-1)^3.
        /// Output: cellMask[cellIdx] = 1 if active, 0 if inactive.
        /// Also outputs the 8-bit case index for each cell.
        /// </summary>
        public static void ClassifyActiveCellsKernel(
            Index3D index,
            ArrayView<short> sdfValues,
            ArrayView<int> cellMask,
            ArrayView<int> cellCases,
            int chunkSize)
        {
            int cx = index.X;
            int cy = index.Y;
            int cz = index.Z;

            int cellCount = chunkSize - 1;
            if (cx >= cellCount || cy >= cellCount || cz >= cellCount) return;

            int cellIdx = cx + cz * cellCount + cy * cellCount * cellCount;

            // Sample 8 corners of this cell
            // Corner ordering (matches standard MC convention):
            //   0: (x, y, z)      4: (x, y, z+1)
            //   1: (x+1, y, z)    5: (x+1, y, z+1)
            //   2: (x+1, y+1, z)  6: (x+1, y+1, z+1)
            //   3: (x, y+1, z)    7: (x, y+1, z+1)
            int stride = chunkSize;
            int strideY = chunkSize * chunkSize;

            short v0 = sdfValues[cx + cz * stride + cy * strideY];
            short v1 = sdfValues[(cx + 1) + cz * stride + cy * strideY];
            short v2 = sdfValues[(cx + 1) + cz * stride + (cy + 1) * strideY];
            short v3 = sdfValues[cx + cz * stride + (cy + 1) * strideY];
            short v4 = sdfValues[cx + (cz + 1) * stride + cy * strideY];
            short v5 = sdfValues[(cx + 1) + (cz + 1) * stride + cy * strideY];
            short v6 = sdfValues[(cx + 1) + (cz + 1) * stride + (cy + 1) * strideY];
            short v7 = sdfValues[cx + (cz + 1) * stride + (cy + 1) * strideY];

            // Build 8-bit case index (bit k = 1 if corner k is inside/solid)
            int caseIndex = 0;
            if (v0 > 0) caseIndex |= 1;
            if (v1 > 0) caseIndex |= 2;
            if (v2 > 0) caseIndex |= 4;
            if (v3 > 0) caseIndex |= 8;
            if (v4 > 0) caseIndex |= 16;
            if (v5 > 0) caseIndex |= 32;
            if (v6 > 0) caseIndex |= 64;
            if (v7 > 0) caseIndex |= 128;

            // Cell is active if not all corners are the same sign (case != 0 and != 255)
            bool active = caseIndex != 0 && caseIndex != 255;

            cellMask[cellIdx] = active ? 1 : 0;
            cellCases[cellIdx] = caseIndex;
        }

        /// <summary>
        /// Generate dual vertices for active cells.
        /// Each active cell produces one vertex at the cell's "mass point"
        /// (average of edge crossing positions within the cell).
        ///
        /// Dispatch: one thread per active cell (use compacted active cell list).
        /// </summary>
        public static void GenerateDualVerticesKernel(
            Index1D index,
            ArrayView<int> activeCellIds,
            ArrayView<short> sdfValues,
            ArrayView<int> cellCases,
            ArrayView<float> vertexPositions,
            ArrayView<float> vertexNormals,
            int chunkSize,
            float voxelSize,
            float chunkWorldX,
            float chunkWorldY,
            float chunkWorldZ,
            int activeCellCount)
        {
            int i = index;
            if (i >= activeCellCount) return;

            int cellIdx = activeCellIds[i];
            int cellCount = chunkSize - 1;
            int cx = cellIdx % cellCount;
            int cz = (cellIdx / cellCount) % cellCount;
            int cy = cellIdx / (cellCount * cellCount);

            int stride = chunkSize;
            int strideY = chunkSize * chunkSize;

            // Sample all 8 corners
            float v0 = sdfValues[cx + cz * stride + cy * strideY] / SdfChunk.FixedPointScale;
            float v1 = sdfValues[(cx + 1) + cz * stride + cy * strideY] / SdfChunk.FixedPointScale;
            float v2 = sdfValues[(cx + 1) + cz * stride + (cy + 1) * strideY] / SdfChunk.FixedPointScale;
            float v3 = sdfValues[cx + cz * stride + (cy + 1) * strideY] / SdfChunk.FixedPointScale;
            float v4 = sdfValues[cx + (cz + 1) * stride + cy * strideY] / SdfChunk.FixedPointScale;
            float v5 = sdfValues[(cx + 1) + (cz + 1) * stride + cy * strideY] / SdfChunk.FixedPointScale;
            float v6 = sdfValues[(cx + 1) + (cz + 1) * stride + (cy + 1) * strideY] / SdfChunk.FixedPointScale;
            float v7 = sdfValues[cx + (cz + 1) * stride + (cy + 1) * strideY] / SdfChunk.FixedPointScale;

            // Compute dual vertex position: mass point of edge crossings.
            // For each edge that crosses the isosurface, interpolate to find crossing point.
            // Average all crossing points to get the dual vertex.
            float sumX = 0, sumY = 0, sumZ = 0;
            int crossingCount = 0;

            // 12 edges of the cell. Check each for sign change.
            // Edge (corner_a, corner_b): if sign differs, interpolate crossing point.

            // Edge 0: corner 0-1 (X axis, bottom-front)
            if ((v0 > 0) != (v1 > 0)) { float t = v0 / (v0 - v1); sumX += cx + t; sumY += cy; sumZ += cz; crossingCount++; }
            // Edge 1: corner 1-2 (Y axis, right-front)
            if ((v1 > 0) != (v2 > 0)) { float t = v1 / (v1 - v2); sumX += cx + 1; sumY += cy + t; sumZ += cz; crossingCount++; }
            // Edge 2: corner 3-2 (X axis, top-front)
            if ((v3 > 0) != (v2 > 0)) { float t = v3 / (v3 - v2); sumX += cx + t; sumY += cy + 1; sumZ += cz; crossingCount++; }
            // Edge 3: corner 0-3 (Y axis, left-front)
            if ((v0 > 0) != (v3 > 0)) { float t = v0 / (v0 - v3); sumX += cx; sumY += cy + t; sumZ += cz; crossingCount++; }
            // Edge 4: corner 4-5 (X axis, bottom-back)
            if ((v4 > 0) != (v5 > 0)) { float t = v4 / (v4 - v5); sumX += cx + t; sumY += cy; sumZ += cz + 1; crossingCount++; }
            // Edge 5: corner 5-6 (Y axis, right-back)
            if ((v5 > 0) != (v6 > 0)) { float t = v5 / (v5 - v6); sumX += cx + 1; sumY += cy + t; sumZ += cz + 1; crossingCount++; }
            // Edge 6: corner 7-6 (X axis, top-back)
            if ((v7 > 0) != (v6 > 0)) { float t = v7 / (v7 - v6); sumX += cx + t; sumY += cy + 1; sumZ += cz + 1; crossingCount++; }
            // Edge 7: corner 4-7 (Y axis, left-back)
            if ((v4 > 0) != (v7 > 0)) { float t = v4 / (v4 - v7); sumX += cx; sumY += cy + t; sumZ += cz + 1; crossingCount++; }
            // Edge 8: corner 0-4 (Z axis, bottom-left)
            if ((v0 > 0) != (v4 > 0)) { float t = v0 / (v0 - v4); sumX += cx; sumY += cy; sumZ += cz + t; crossingCount++; }
            // Edge 9: corner 1-5 (Z axis, bottom-right)
            if ((v1 > 0) != (v5 > 0)) { float t = v1 / (v1 - v5); sumX += cx + 1; sumY += cy; sumZ += cz + t; crossingCount++; }
            // Edge 10: corner 2-6 (Z axis, top-right)
            if ((v2 > 0) != (v6 > 0)) { float t = v2 / (v2 - v6); sumX += cx + 1; sumY += cy + 1; sumZ += cz + t; crossingCount++; }
            // Edge 11: corner 3-7 (Z axis, top-left)
            if ((v3 > 0) != (v7 > 0)) { float t = v3 / (v3 - v7); sumX += cx; sumY += cy + 1; sumZ += cz + t; crossingCount++; }

            // Average to get dual vertex position (in chunk-local voxel coords)
            float invCount = crossingCount > 0 ? 1f / crossingCount : 0f;
            float lx = sumX * invCount;
            float ly = sumY * invCount;
            float lz = sumZ * invCount;

            // Convert to world coordinates
            float wx = chunkWorldX + lx * voxelSize;
            float wy = chunkWorldY + ly * voxelSize;
            float wz = chunkWorldZ + lz * voxelSize;

            // Compute normal via SDF gradient (central differences)
            // Sample SDF at +-epsilon around the vertex position
            // For GPU efficiency, use the 8 corner values to approximate gradient
            float nx = (v1 + v2 + v5 + v6) - (v0 + v3 + v4 + v7); // X gradient
            float ny = (v2 + v3 + v6 + v7) - (v0 + v1 + v4 + v5); // Y gradient
            float nz = (v4 + v5 + v6 + v7) - (v0 + v1 + v2 + v3); // Z gradient

            // Normalize
            float nlen = nx * nx + ny * ny + nz * nz;
            if (nlen > 0.0001f)
            {
                // Newton-Raphson inverse sqrt (2 iterations)
                float invSqrt = nlen;
                float half = 0.5f * nlen;
                int bits = 0x5F3759DF - ((int)(nlen * 1f) >> 1); // Quake fast inverse sqrt seed
                // Simplified: just normalize manually
                float len = nlen;
                float guess = len * 0.5f;
                guess = 0.5f * (guess + len / guess);
                guess = 0.5f * (guess + len / guess);
                len = guess;
                nx /= len;
                ny /= len;
                nz /= len;
            }

            // Store vertex position and normal (6 floats per vertex)
            int outIdx = i * 6;
            vertexPositions[outIdx + 0] = wx;
            vertexPositions[outIdx + 1] = wy;
            vertexPositions[outIdx + 2] = wz;
            vertexNormals[outIdx + 0] = nx;
            vertexNormals[outIdx + 1] = ny;
            vertexNormals[outIdx + 2] = nz;
        }

        /// <summary>
        /// Generate quads connecting dual vertices of adjacent active cells.
        /// A quad is emitted for each edge that has active cells on both sides
        /// with differing signs on the edge's two endpoints.
        ///
        /// Dispatch: one thread per cell (chunkSize-1)^3.
        /// Each thread checks 3 edges (+X, +Y, +Z) from its cell and emits quads.
        /// </summary>
        public static void GenerateQuadsKernel(
            Index3D index,
            ArrayView<short> sdfValues,
            ArrayView<int> cellToVertex,
            ArrayView<int> quadOutput,
            ArrayView<int> quadCounter,
            int chunkSize,
            int maxQuads)
        {
            int cx = index.X;
            int cy = index.Y;
            int cz = index.Z;

            int cellCount = chunkSize - 1;
            if (cx >= cellCount || cy >= cellCount || cz >= cellCount) return;

            int stride = chunkSize;
            int strideY = chunkSize * chunkSize;

            // Check 3 edges from this cell: +X, +Y, +Z directions.
            // A quad is emitted when an edge crosses the isosurface
            // and all 4 cells sharing that edge are active.

            // Edge along X (between corners 0 and 1)
            if (cy > 0 && cz > 0)
            {
                short a = sdfValues[cx + cz * stride + cy * strideY];
                short b = sdfValues[(cx + 1) + cz * stride + cy * strideY];
                if ((a > 0) != (b > 0))
                {
                    // 4 cells sharing this edge
                    int c0 = cx + cz * cellCount + cy * cellCount * cellCount;
                    int c1 = cx + (cz - 1) * cellCount + cy * cellCount * cellCount;
                    int c2 = cx + (cz - 1) * cellCount + (cy - 1) * cellCount * cellCount;
                    int c3 = cx + cz * cellCount + (cy - 1) * cellCount * cellCount;

                    int v0 = cellToVertex[c0];
                    int v1 = cellToVertex[c1];
                    int v2 = cellToVertex[c2];
                    int v3 = cellToVertex[c3];

                    if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0)
                    {
                        int slot = Atomic.Add(ref quadCounter[0], 1);
                        if (slot < maxQuads)
                        {
                            int qIdx = slot * 4;
                            if (a > 0) { quadOutput[qIdx] = v0; quadOutput[qIdx + 1] = v1; quadOutput[qIdx + 2] = v2; quadOutput[qIdx + 3] = v3; }
                            else { quadOutput[qIdx] = v3; quadOutput[qIdx + 1] = v2; quadOutput[qIdx + 2] = v1; quadOutput[qIdx + 3] = v0; }
                        }
                        else { Atomic.Add(ref quadCounter[0], -1); }
                    }
                }
            }

            // Edge along Y (between corners 0 and 3)
            if (cx > 0 && cz > 0)
            {
                short a = sdfValues[cx + cz * stride + cy * strideY];
                short b = sdfValues[cx + cz * stride + (cy + 1) * strideY];
                if ((a > 0) != (b > 0))
                {
                    int c0 = cx + cz * cellCount + cy * cellCount * cellCount;
                    int c1 = (cx - 1) + cz * cellCount + cy * cellCount * cellCount;
                    int c2 = (cx - 1) + (cz - 1) * cellCount + cy * cellCount * cellCount;
                    int c3 = cx + (cz - 1) * cellCount + cy * cellCount * cellCount;

                    int v0 = cellToVertex[c0];
                    int v1 = cellToVertex[c1];
                    int v2 = cellToVertex[c2];
                    int v3 = cellToVertex[c3];

                    if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0)
                    {
                        int slot = Atomic.Add(ref quadCounter[0], 1);
                        if (slot < maxQuads)
                        {
                            int qIdx = slot * 4;
                            if (a > 0) { quadOutput[qIdx] = v0; quadOutput[qIdx + 1] = v1; quadOutput[qIdx + 2] = v2; quadOutput[qIdx + 3] = v3; }
                            else { quadOutput[qIdx] = v3; quadOutput[qIdx + 1] = v2; quadOutput[qIdx + 2] = v1; quadOutput[qIdx + 3] = v0; }
                        }
                        else { Atomic.Add(ref quadCounter[0], -1); }
                    }
                }
            }

            // Edge along Z (between corners 0 and 4)
            if (cx > 0 && cy > 0)
            {
                short a = sdfValues[cx + cz * stride + cy * strideY];
                short b = sdfValues[cx + (cz + 1) * stride + cy * strideY];
                if ((a > 0) != (b > 0))
                {
                    int c0 = cx + cz * cellCount + cy * cellCount * cellCount;
                    int c1 = (cx - 1) + cz * cellCount + cy * cellCount * cellCount;
                    int c2 = (cx - 1) + cz * cellCount + (cy - 1) * cellCount * cellCount;
                    int c3 = cx + cz * cellCount + (cy - 1) * cellCount * cellCount;

                    int v0 = cellToVertex[c0];
                    int v1 = cellToVertex[c1];
                    int v2 = cellToVertex[c2];
                    int v3 = cellToVertex[c3];

                    if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0)
                    {
                        int slot = Atomic.Add(ref quadCounter[0], 1);
                        if (slot < maxQuads)
                        {
                            int qIdx = slot * 4;
                            if (a > 0) { quadOutput[qIdx] = v0; quadOutput[qIdx + 1] = v1; quadOutput[qIdx + 2] = v2; quadOutput[qIdx + 3] = v3; }
                            else { quadOutput[qIdx] = v3; quadOutput[qIdx + 1] = v2; quadOutput[qIdx + 2] = v1; quadOutput[qIdx + 3] = v0; }
                        }
                        else { Atomic.Add(ref quadCounter[0], -1); }
                    }
                }
            }
        }
    }
}
