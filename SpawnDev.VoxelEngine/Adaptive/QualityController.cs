namespace SpawnDev.VoxelEngine.Adaptive
{
    /// <summary>
    /// Dynamic quality controller that adjusts rendering parameters based on frame timing.
    ///
    /// Monitors frame time via exponential moving average (EMA).
    /// When over budget: decrease quality quickly (4x rate).
    /// When under budget: increase quality slowly (1x rate).
    /// Thermal detection: frame time spikes without scene changes = GPU throttling.
    ///
    /// Quality levels control draw distance, LOD bias, shadow quality, particle count.
    /// Each level has a defined budget range. The controller adjusts level to stay in budget.
    /// </summary>
    public class QualityController
    {
        /// <summary>Quality levels from highest to lowest fidelity.</summary>
        public enum QualityLevel
        {
            Ultra = 0,
            High = 1,
            Medium = 2,
            Low = 3,
            Minimal = 4,
        }

        /// <summary>Current quality level.</summary>
        public QualityLevel Level { get; private set; } = QualityLevel.High;

        /// <summary>Smoothed frame time in milliseconds.</summary>
        public float SmoothedFrameTimeMs { get; private set; }

        /// <summary>Whether thermal throttling is detected.</summary>
        public bool ThermalDetected { get; private set; }

        /// <summary>Number of consecutive frames over budget.</summary>
        public int OverBudgetFrames { get; private set; }

        /// <summary>Number of consecutive frames under budget.</summary>
        public int UnderBudgetFrames { get; private set; }

        /// <summary>Target frame time in ms (e.g., 13.88 for 72Hz Quest, 16.67 for 60Hz).</summary>
        public float TargetFrameTimeMs { get; set; } = 13.88f; // 72Hz default for Quest 3S

        /// <summary>How many over-budget frames before decreasing quality.</summary>
        public int DecreaseThreshold { get; set; } = 5;

        /// <summary>How many under-budget frames before increasing quality.</summary>
        public int IncreaseThreshold { get; set; } = 60; // ~1 second at 60fps

        /// <summary>EMA smoothing factor (0-1). Higher = more responsive, noisier.</summary>
        public float SmoothingFactor { get; set; } = 0.1f;

        /// <summary>Over-budget margin. Frame time must exceed target * (1 + margin) to trigger decrease.</summary>
        public float OverBudgetMargin { get; set; } = 0.15f; // 15% over

        /// <summary>Under-budget margin. Frame time must be below target * (1 - margin) to trigger increase.</summary>
        public float UnderBudgetMargin { get; set; } = 0.25f; // 25% under

        /// <summary>Thermal spike threshold: if frame time jumps by this factor without scene change, suspect thermal.</summary>
        public float ThermalSpikeRatio { get; set; } = 2.0f;

        // Internal state
        private float _previousFrameTimeMs;
        private int _thermalSpikeCount;
        private const int ThermalSpikeThreshold = 3; // consecutive spikes to confirm thermal

        /// <summary>
        /// Update the controller with the latest frame time.
        /// Call once per frame after rendering.
        /// </summary>
        /// <param name="frameTimeMs">This frame's total time in milliseconds.</param>
        /// <param name="sceneChanged">True if the camera moved significantly or chunks loaded. False = static scene.</param>
        public void Update(float frameTimeMs, bool sceneChanged = true)
        {
            // EMA smoothing
            if (SmoothedFrameTimeMs <= 0)
                SmoothedFrameTimeMs = frameTimeMs;
            else
                SmoothedFrameTimeMs += SmoothingFactor * (frameTimeMs - SmoothedFrameTimeMs);

            // Thermal detection: frame time spike without scene change
            if (!sceneChanged && _previousFrameTimeMs > 0)
            {
                if (frameTimeMs > _previousFrameTimeMs * ThermalSpikeRatio)
                {
                    _thermalSpikeCount++;
                    if (_thermalSpikeCount >= ThermalSpikeThreshold)
                        ThermalDetected = true;
                }
                else
                {
                    _thermalSpikeCount = Math.Max(0, _thermalSpikeCount - 1);
                    if (_thermalSpikeCount == 0)
                        ThermalDetected = false;
                }
            }
            _previousFrameTimeMs = frameTimeMs;

            // Emergency: thermal detected -> drop to minimum immediately
            if (ThermalDetected && Level < QualityLevel.Minimal)
            {
                Level = QualityLevel.Minimal;
                OverBudgetFrames = 0;
                UnderBudgetFrames = 0;
                return;
            }

            float overThreshold = TargetFrameTimeMs * (1f + OverBudgetMargin);
            float underThreshold = TargetFrameTimeMs * (1f - UnderBudgetMargin);

            // Over budget: decrease quality (fast - asymmetric 4x rate)
            if (SmoothedFrameTimeMs > overThreshold)
            {
                OverBudgetFrames++;
                UnderBudgetFrames = 0;

                if (OverBudgetFrames >= DecreaseThreshold && Level < QualityLevel.Minimal)
                {
                    Level = (QualityLevel)((int)Level + 1);
                    OverBudgetFrames = 0;
                }
            }
            // Under budget: increase quality (slow - 1x rate)
            else if (SmoothedFrameTimeMs < underThreshold)
            {
                UnderBudgetFrames++;
                OverBudgetFrames = 0;

                if (UnderBudgetFrames >= IncreaseThreshold && Level > QualityLevel.Ultra)
                {
                    Level = (QualityLevel)((int)Level - 1);
                    UnderBudgetFrames = 0;
                }
            }
            else
            {
                // In budget range - reset counters
                OverBudgetFrames = 0;
                UnderBudgetFrames = 0;
            }
        }

        /// <summary>
        /// Get the draw distance multiplier for the current quality level.
        /// Ultra = 1.0, each step down reduces by 25%.
        /// </summary>
        public float DrawDistanceMultiplier => Level switch
        {
            QualityLevel.Ultra => 1.0f,
            QualityLevel.High => 0.75f,
            QualityLevel.Medium => 0.5f,
            QualityLevel.Low => 0.35f,
            QualityLevel.Minimal => 0.2f,
            _ => 0.5f,
        };

        /// <summary>
        /// Get the LOD bias for the current quality level.
        /// 0 = prefer highest detail, higher = prefer lower detail earlier.
        /// </summary>
        public int LodBias => Level switch
        {
            QualityLevel.Ultra => 0,
            QualityLevel.High => 0,
            QualityLevel.Medium => 1,
            QualityLevel.Low => 2,
            QualityLevel.Minimal => 3,
            _ => 1,
        };

        /// <summary>
        /// Get the maximum vertex budget multiplier for the current quality level.
        /// </summary>
        public float VertexBudgetMultiplier => Level switch
        {
            QualityLevel.Ultra => 1.0f,
            QualityLevel.High => 0.8f,
            QualityLevel.Medium => 0.5f,
            QualityLevel.Low => 0.3f,
            QualityLevel.Minimal => 0.15f,
            _ => 0.5f,
        };

        /// <summary>Reset controller to initial state.</summary>
        public void Reset(QualityLevel initialLevel = QualityLevel.High)
        {
            Level = initialLevel;
            SmoothedFrameTimeMs = 0;
            ThermalDetected = false;
            OverBudgetFrames = 0;
            UnderBudgetFrames = 0;
            _previousFrameTimeMs = 0;
            _thermalSpikeCount = 0;
        }
    }
}
