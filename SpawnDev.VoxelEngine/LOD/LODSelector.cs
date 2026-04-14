namespace SpawnDev.VoxelEngine.LOD
{
    /// <summary>
    /// Selects the appropriate LOD level for a section based on distance and pressure.
    ///
    /// Hysteresis prevents oscillation: a section must move significantly past a
    /// threshold before changing LOD (increase distance = 120% of threshold,
    /// decrease distance = 80% of threshold).
    ///
    /// Quality pressure (from QualityController) shifts all thresholds closer,
    /// effectively lowering LOD quality to maintain frame rate.
    /// </summary>
    public class LODSelector
    {
        /// <summary>Distance thresholds for each LOD transition (in sections).</summary>
        public float[] LodDistances { get; set; } = { 8f, 16f, 32f };

        /// <summary>Hysteresis factor for LOD transitions. Higher = more sticky.</summary>
        public float HysteresisFactor { get; set; } = 0.2f;

        /// <summary>
        /// Select LOD level for a section.
        /// </summary>
        /// <param name="distanceSections">Distance from camera in section units.</param>
        /// <param name="currentLod">Current LOD level of this section.</param>
        /// <param name="pressureBias">LOD bias from QualityController (0 = no pressure).</param>
        /// <returns>New LOD level (0 = full detail, 1-3 = reduced).</returns>
        public int SelectLOD(float distanceSections, int currentLod, int pressureBias = 0)
        {
            int targetLod = 0;

            for (int i = 0; i < LodDistances.Length; i++)
            {
                float threshold = LodDistances[i];

                // Apply pressure: shift thresholds closer
                if (pressureBias > 0)
                    threshold *= MathF.Max(0.25f, 1f - pressureBias * 0.15f);

                // Apply hysteresis based on current direction
                float effectiveThreshold;
                if (currentLod <= i)
                {
                    // Currently at higher detail - need to go past threshold + hysteresis to reduce
                    effectiveThreshold = threshold * (1f + HysteresisFactor);
                }
                else
                {
                    // Currently at lower detail - need to come back below threshold - hysteresis to increase
                    effectiveThreshold = threshold * (1f - HysteresisFactor);
                }

                if (distanceSections > effectiveThreshold)
                    targetLod = i + 1;
            }

            return Math.Min(targetLod, 3); // max LOD 3
        }

        /// <summary>
        /// Compute geomorph blend factor for smooth LOD transitions.
        /// 0 = fully current LOD, 1 = fully next LOD.
        /// </summary>
        public float GetMorphFactor(float distanceSections, int currentLod, int pressureBias = 0)
        {
            if (currentLod >= LodDistances.Length) return 0f;

            float threshold = LodDistances[currentLod];
            if (pressureBias > 0)
                threshold *= MathF.Max(0.25f, 1f - pressureBias * 0.15f);

            // Transition zone = 20% of threshold distance before the switch point
            float transitionStart = threshold * 0.8f;
            float transitionEnd = threshold;

            if (distanceSections <= transitionStart) return 0f;
            if (distanceSections >= transitionEnd) return 1f;

            return (distanceSections - transitionStart) / (transitionEnd - transitionStart);
        }
    }
}
