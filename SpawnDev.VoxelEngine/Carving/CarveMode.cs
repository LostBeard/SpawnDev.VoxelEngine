namespace SpawnDev.VoxelEngine.Carving
{
    /// <summary>
    /// The kind of CSG operation a carve tool performs on an SDF field.
    /// </summary>
    public enum CarveMode
    {
        /// <summary>
        /// Subtract the tool shape from the terrain (dig a hole).
        /// Runs <c>SdfSmoothSubtract</c> with a small blend radius so edges feel natural.
        /// </summary>
        Dig = 0,

        /// <summary>
        /// Add the tool shape into the terrain (fill in a hole or build up a mound).
        /// Runs <c>SdfSmoothUnion</c> with a small blend radius.
        /// </summary>
        Fill = 1,

        /// <summary>
        /// Same sign as <see cref="Dig"/> (material is removed), but with a wider
        /// blend radius so the crater has a softer, explosion-like rim instead of
        /// the crisper cut a normal dig leaves.
        /// </summary>
        Explode = 2,
    }
}
