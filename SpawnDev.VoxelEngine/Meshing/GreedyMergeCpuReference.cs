namespace SpawnDev.VoxelEngine.Meshing
{
    /// <summary>
    /// CPU reference implementation of greedy quad merging.
    /// Results must cover the same set of faces as the face masks,
    /// and every merged quad must contain only blocks of one type.
    /// </summary>
    public static class GreedyMergeCpuReference
    {
        /// <summary>
        /// Run the full greedy merge on CPU. Returns list of packed quads.
        /// Modifies faceMasks in place (clears consumed bits).
        /// </summary>
        public static List<long> GreedyMerge(
            long[] faceMasks, int[] paddedBlocks,
            int chunkXZ, int height, int paddedXZ)
        {
            var quads = new List<long>();
            int innerCount = chunkXZ * chunkXZ;
            int stride = paddedXZ * paddedXZ;

            // Clone face masks so we can modify them
            var masks = (long[])faceMasks.Clone();

            for (int face = 0; face < 6; face++)
            {
                int faceOffset = face * innerCount;

                if (face >= 4) // +Y, -Y
                {
                    MergeXZPlane(masks, paddedBlocks, quads, chunkXZ, height, paddedXZ,
                        innerCount, stride, faceOffset, face);
                }
                else // +X, -X, +Z, -Z
                {
                    MergePerpendicularPlane(masks, paddedBlocks, quads, chunkXZ, height, paddedXZ,
                        innerCount, stride, faceOffset, face);
                }
            }

            return quads;
        }

        private static void MergeXZPlane(
            long[] masks, int[] paddedBlocks, List<long> quads,
            int chunkXZ, int height, int paddedXZ,
            int innerCount, int stride, int faceOffset, int face)
        {
            for (int yLayer = 0; yLayer < height; yLayer++)
            {
                for (int z = 0; z < chunkXZ; z++)
                {
                    int x = 0;
                    while (x < chunkXZ)
                    {
                        int innerIdx = faceOffset + x + z * chunkXZ;
                        if ((masks[innerIdx] & (1L << yLayer)) == 0) { x++; continue; }

                        int blockType = paddedBlocks[(x + 1) + (z + 1) * paddedXZ + yLayer * stride];

                        // Extend width
                        int w = 1;
                        while (x + w < chunkXZ)
                        {
                            int ni = faceOffset + (x + w) + z * chunkXZ;
                            if ((masks[ni] & (1L << yLayer)) == 0) break;
                            int nt = paddedBlocks[(x + w + 1) + (z + 1) * paddedXZ + yLayer * stride];
                            if (nt != blockType) break;
                            w++;
                        }

                        // Extend height
                        int h = 1;
                        bool canExtend = true;
                        while (z + h < chunkXZ && canExtend)
                        {
                            for (int dx = 0; dx < w; dx++)
                            {
                                int ci = faceOffset + (x + dx) + (z + h) * chunkXZ;
                                if ((masks[ci] & (1L << yLayer)) == 0) { canExtend = false; break; }
                                int ct = paddedBlocks[(x + dx + 1) + (z + h + 1) * paddedXZ + yLayer * stride];
                                if (ct != blockType) { canExtend = false; break; }
                            }
                            if (canExtend) h++;
                        }

                        // Clear
                        for (int dz = 0; dz < h; dz++)
                            for (int dx = 0; dx < w; dx++)
                                masks[faceOffset + (x + dx) + (z + dz) * chunkXZ] &= ~(1L << yLayer);

                        quads.Add(PackedQuad.Pack(x, yLayer, z, w, h, face, blockType));
                        x += w;
                    }
                }
            }
        }

        private static void MergePerpendicularPlane(
            long[] masks, int[] paddedBlocks, List<long> quads,
            int chunkXZ, int height, int paddedXZ,
            int innerCount, int stride, int faceOffset, int face)
        {
            bool isXFace = face <= 1;
            int axisSize = chunkXZ;

            for (int layer = 0; layer < axisSize; layer++)
            {
                for (int outer = 0; outer < chunkXZ; outer++)
                {
                    int innerX = isXFace ? layer : outer;
                    int innerZ = isXFace ? outer : layer;
                    int innerIdx = faceOffset + innerX + innerZ * chunkXZ;
                    long mask = masks[innerIdx];
                    if (mask == 0) continue;

                    while (mask != 0)
                    {
                        int y = TrailingZeros(mask);
                        if (y >= height) break;

                        int px = (isXFace ? layer : outer) + 1;
                        int pz = (isXFace ? outer : layer) + 1;
                        int blockType = paddedBlocks[px + pz * paddedXZ + y * stride];

                        // Extend along Y
                        int h = 1;
                        while (y + h < height && (mask & (1L << (y + h))) != 0)
                        {
                            int nt = paddedBlocks[px + pz * paddedXZ + (y + h) * stride];
                            if (nt != blockType) break;
                            h++;
                        }

                        for (int dy = 0; dy < h; dy++)
                            mask &= ~(1L << (y + dy));
                        masks[innerIdx] = mask;

                        // Extend along outer axis
                        int w = 1;
                        while (outer + w < chunkXZ)
                        {
                            int nix = isXFace ? layer : outer + w;
                            int niz = isXFace ? outer + w : layer;
                            int nii = faceOffset + nix + niz * chunkXZ;
                            long nm = masks[nii];

                            bool allMatch = true;
                            for (int dy = 0; dy < h; dy++)
                            {
                                if ((nm & (1L << (y + dy))) == 0) { allMatch = false; break; }
                                int npx = (isXFace ? layer : outer + w) + 1;
                                int npz = (isXFace ? outer + w : layer) + 1;
                                int nt = paddedBlocks[npx + npz * paddedXZ + (y + dy) * stride];
                                if (nt != blockType) { allMatch = false; break; }
                            }
                            if (!allMatch) break;

                            for (int dy = 0; dy < h; dy++)
                                nm &= ~(1L << (y + dy));
                            masks[nii] = nm;
                            w++;
                        }

                        int qx = isXFace ? layer : outer;
                        int qz = isXFace ? outer : layer;
                        quads.Add(PackedQuad.Pack(qx, y, qz, w, h, face, blockType));
                    }
                }
            }
        }

        /// <summary>
        /// Verify that a set of merged quads exactly covers the same faces as the original face masks.
        /// Expands each quad back into individual face bits and compares against the original masks.
        /// Returns null on success, error message on failure.
        /// </summary>
        public static string? VerifyQuadCoverage(
            List<long> quads, long[] originalFaceMasks,
            int chunkXZ, int height)
        {
            int innerCount = chunkXZ * chunkXZ;
            var reconstructed = new long[innerCount * 6];

            foreach (var packed in quads)
            {
                PackedQuad.Unpack(packed, out int x, out int y, out int z,
                    out int w, out int h, out int face, out int blockType);

                if (face >= 4) // +Y/-Y: w is X extent, h is Z extent
                {
                    for (int dz = 0; dz < h; dz++)
                    {
                        for (int dx = 0; dx < w; dx++)
                        {
                            int idx = face * innerCount + (x + dx) + (z + dz) * chunkXZ;
                            reconstructed[idx] |= 1L << y;
                        }
                    }
                }
                else // +X,-X,+Z,-Z: w is outer extent, h is Y extent
                {
                    bool isXFace = face <= 1;
                    for (int dw = 0; dw < w; dw++)
                    {
                        int ix = isXFace ? x : x + dw;
                        int iz = isXFace ? z + dw : z;
                        int idx = face * innerCount + ix + iz * chunkXZ;
                        for (int dy = 0; dy < h; dy++)
                        {
                            reconstructed[idx] |= 1L << (y + dy);
                        }
                    }
                }
            }

            // Compare
            for (int i = 0; i < originalFaceMasks.Length; i++)
            {
                if (reconstructed[i] != originalFaceMasks[i])
                {
                    int face = i / innerCount;
                    int rem = i % innerCount;
                    int x = rem % chunkXZ;
                    int z = rem / chunkXZ;
                    string[] faceNames = { "+X", "-X", "+Z", "-Z", "+Y", "-Y" };
                    return $"Coverage mismatch at face={faceNames[face]} x={x} z={z}: " +
                        $"original=0x{originalFaceMasks[i]:X16}, reconstructed=0x{reconstructed[i]:X16}";
                }
            }

            return null; // success
        }

        private static int TrailingZeros(long value)
        {
            if (value == 0) return 64;
            int count = 0;
            long v = value;
            while ((v & 1) == 0) { count++; v >>= 1; }
            return count;
        }
    }
}
