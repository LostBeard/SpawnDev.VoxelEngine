using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// GPU kernels for draw ordering via distance-based sorting.
    ///
    /// Opaque sections: front-to-back (early-Z rejects hidden fragments).
    /// Transparent sections: back-to-front (correct alpha blending).
    /// Same sort keys, two iteration orders. One RadixSort, two passes.
    ///
    /// The sort key is the squared distance from camera to section center,
    /// quantized to uint32 for RadixSort compatibility.
    /// Section indices are the values sorted alongside the keys.
    /// </summary>
    public static class DrawOrderKernels
    {
        /// <summary>
        /// Compute distance-squared sort keys for all sections.
        /// One thread per section.
        /// </summary>
        public static void ComputeDistanceKeysKernel(
            Index1D index,
            ArrayView<int> sectionCoordsFlat, // [cx0, sy0, cz0, cx1, sy1, cz1, ...]
            ArrayView<uint> distanceKeys,
            ArrayView<int> sectionIndices,
            float camX, float camY, float camZ,
            float voxelSize, int sectionSize, float baseY)
        {
            int i3 = index * 3;
            if (i3 + 2 >= sectionCoordsFlat.IntLength) return;

            int cx = sectionCoordsFlat[i3];
            int sy = sectionCoordsFlat[i3 + 1];
            int cz = sectionCoordsFlat[i3 + 2];

            // Section center in world space
            float halfSection = sectionSize * voxelSize * 0.5f;
            float centerX = cx * sectionSize * voxelSize + halfSection;
            float centerY = baseY + sy * sectionSize * voxelSize + halfSection;
            float centerZ = cz * sectionSize * voxelSize + halfSection;

            float dx = centerX - camX;
            float dy = centerY - camY;
            float dz = centerZ - camZ;
            float distSq = dx * dx + dy * dy + dz * dz;

            // Quantize to uint32 (preserves ordering for positive values)
            // Clamp to avoid overflow
            uint key = distSq >= 4294967295f ? uint.MaxValue : (uint)distSq;

            distanceKeys[index] = key;
            sectionIndices[index] = index;
        }

        /// <summary>
        /// CPU reference: compute distance keys and sort indices by distance.
        /// Returns sorted indices (front-to-back for opaque, reversed for transparent).
        /// </summary>
        public static int[] SortByDistance(
            SectionEntry[] entries,
            float camX, float camY, float camZ,
            float voxelSize, int sectionSize, float baseY,
            bool backToFront = false)
        {
            if (entries.Length == 0) return Array.Empty<int>();

            var indexed = new (float distSq, int index)[entries.Length];
            for (int i = 0; i < entries.Length; i++)
            {
                var coord = entries[i].Coord;
                float halfSection = sectionSize * voxelSize * 0.5f;
                float centerX = coord.Cx * sectionSize * voxelSize + halfSection;
                float centerY = baseY + coord.Sy * sectionSize * voxelSize + halfSection;
                float centerZ = coord.Cz * sectionSize * voxelSize + halfSection;

                float dx = centerX - camX;
                float dy = centerY - camY;
                float dz = centerZ - camZ;
                indexed[i] = (dx * dx + dy * dy + dz * dz, i);
            }

            if (backToFront)
                Array.Sort(indexed, (a, b) => b.distSq.CompareTo(a.distSq));
            else
                Array.Sort(indexed, (a, b) => a.distSq.CompareTo(b.distSq));

            return indexed.Select(x => x.index).ToArray();
        }
    }
}
