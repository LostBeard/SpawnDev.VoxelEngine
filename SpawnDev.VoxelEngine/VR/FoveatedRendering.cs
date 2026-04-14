namespace SpawnDev.VoxelEngine.VR
{
    /// <summary>
    /// Fixed Foveated Rendering (FFR) integration for Quest 3S.
    ///
    /// FFR reduces fragment shader work in the peripheral vision where
    /// the user can't see detail anyway. The GPU renders the center at
    /// full resolution and the edges at reduced resolution.
    ///
    /// WebXR exposes FFR via the "foveation" session feature.
    /// Quest 3S supports 3 FFR levels: low, medium, high.
    ///
    /// Voxel rendering is typically vertex-bound (many small quads),
    /// not fragment-bound. FFR helps most when PBR lighting with
    /// multiple dynamic lights makes the fragment shader expensive.
    ///
    /// The QualityController and ThermalManager adjust FFR level:
    /// - Normal: FFR off (full quality)
    /// - Warning: FFR low
    /// - Throttling: FFR medium
    /// - Critical: FFR high
    /// </summary>
    public class FoveatedRendering
    {
        /// <summary>FFR level.</summary>
        public enum FFRLevel
        {
            /// <summary>No foveation. Full resolution everywhere.</summary>
            None = 0,

            /// <summary>Light peripheral reduction. Barely noticeable.</summary>
            Low = 1,

            /// <summary>Moderate peripheral reduction. Noticeable if looking for it.</summary>
            Medium = 2,

            /// <summary>Heavy peripheral reduction. Visible but acceptable under thermal pressure.</summary>
            High = 3,
        }

        /// <summary>Current FFR level.</summary>
        public FFRLevel Level { get; private set; } = FFRLevel.None;

        /// <summary>Whether FFR is supported by the current XR session.</summary>
        public bool IsSupported { get; set; }

        /// <summary>Whether the fragment shader is heavy enough to benefit from FFR.</summary>
        public bool FragmentHeavy { get; set; }

        /// <summary>
        /// Update FFR level based on quality controller and thermal state.
        /// Call once per frame after QualityController and ThermalManager update.
        /// </summary>
        public void Update(
            Adaptive.QualityController.QualityLevel quality,
            Adaptive.ThermalManager.ThermalState thermal)
        {
            if (!IsSupported)
            {
                Level = FFRLevel.None;
                return;
            }

            // Thermal state takes priority
            Level = thermal switch
            {
                Adaptive.ThermalManager.ThermalState.Critical => FFRLevel.High,
                Adaptive.ThermalManager.ThermalState.Throttling => FFRLevel.Medium,
                Adaptive.ThermalManager.ThermalState.Warning => FFRLevel.Low,
                Adaptive.ThermalManager.ThermalState.Recovering =>
                    quality >= Adaptive.QualityController.QualityLevel.Medium ? FFRLevel.Medium : FFRLevel.Low,
                _ => quality switch
                {
                    Adaptive.QualityController.QualityLevel.Minimal => FFRLevel.High,
                    Adaptive.QualityController.QualityLevel.Low => FFRLevel.Medium,
                    Adaptive.QualityController.QualityLevel.Medium when FragmentHeavy => FFRLevel.Low,
                    _ => FFRLevel.None,
                },
            };
        }

        /// <summary>
        /// Get the WebXR foveation level value (0.0 to 1.0) for the XR session.
        /// WebXR maps: 0.0 = none, 0.33 = low, 0.67 = medium, 1.0 = high.
        /// </summary>
        public float GetWebXRFoveationLevel()
        {
            return Level switch
            {
                FFRLevel.None => 0f,
                FFRLevel.Low => 0.33f,
                FFRLevel.Medium => 0.67f,
                FFRLevel.High => 1f,
                _ => 0f,
            };
        }
    }
}
