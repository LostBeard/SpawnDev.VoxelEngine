namespace SpawnDev.VoxelEngine.Adaptive
{
    /// <summary>
    /// Quest 3S thermal management.
    ///
    /// Mobile VR GPUs throttle when they overheat. The thermal manager detects
    /// throttling via frame time analysis and proactively reduces workload
    /// before the GPU hits critical temperature.
    ///
    /// Detection: consecutive frame time spikes without scene changes = thermal.
    /// The GPU is doing the same work but taking longer = clock speed reduced.
    ///
    /// Response levels:
    /// 1. Warning: reduce draw distance 20%, increase LOD bias +1
    /// 2. Throttling: reduce draw distance 50%, LOD bias +2, disable shadows
    /// 3. Critical: minimum draw distance, LOD bias +3, disable all effects
    /// 4. Recovery: gradually restore over 30 seconds when temps drop
    ///
    /// Quest 3S specific: Adreno 740 throttles at ~85C die temp.
    /// WebXR doesn't expose GPU temp directly, so we infer from frame timing.
    /// </summary>
    public class ThermalManager
    {
        public enum ThermalState
        {
            /// <summary>Normal operating temperature. Full quality.</summary>
            Normal,

            /// <summary>Warming up. Slight quality reduction as preventive measure.</summary>
            Warning,

            /// <summary>Active throttling detected. Significant quality reduction.</summary>
            Throttling,

            /// <summary>Severe throttling. Minimum viable quality to prevent shutdown.</summary>
            Critical,

            /// <summary>Recovering from thermal event. Gradually restoring quality.</summary>
            Recovering,
        }

        /// <summary>Current thermal state.</summary>
        public ThermalState State { get; private set; } = ThermalState.Normal;

        /// <summary>Time spent in current state (seconds).</summary>
        public float StateTime { get; private set; }

        /// <summary>Draw distance multiplier based on thermal state.</summary>
        public float DrawDistanceMultiplier => State switch
        {
            ThermalState.Normal => 1.0f,
            ThermalState.Warning => 0.8f,
            ThermalState.Throttling => 0.5f,
            ThermalState.Critical => 0.2f,
            ThermalState.Recovering => _recoveryProgress,
            _ => 1.0f,
        };

        /// <summary>LOD bias from thermal state.</summary>
        public int LodBias => State switch
        {
            ThermalState.Normal => 0,
            ThermalState.Warning => 1,
            ThermalState.Throttling => 2,
            ThermalState.Critical => 3,
            ThermalState.Recovering => Math.Max(0, 2 - (int)(_recoveryProgress * 3)),
            _ => 0,
        };

        /// <summary>Whether effects (shadows, particles, AO) should be disabled.</summary>
        public bool DisableEffects => State == ThermalState.Critical || State == ThermalState.Throttling;

        // Internal
        private float _baselineFrameTime; // established during Normal state
        private float _emaFrameTime;
        private int _spikeCount;
        private float _recoveryProgress; // 0 = just started recovering, 1 = fully recovered
        private const float RecoveryDuration = 30f; // seconds to fully recover

        // Thresholds
        private const float SpikeRatio = 1.5f; // 50% above baseline = spike
        private const int WarningSpikes = 5; // consecutive spikes to enter Warning
        private const int ThrottlingSpikes = 15;
        private const int CriticalSpikes = 30;
        private const float CooldownTime = 10f; // seconds of normal frames to exit Warning

        /// <summary>
        /// Update thermal state. Call once per frame.
        /// </summary>
        /// <param name="frameTimeMs">This frame's time in milliseconds.</param>
        /// <param name="sceneChanged">Whether the scene changed (camera move, chunk load).</param>
        /// <param name="dt">Delta time in seconds.</param>
        public void Update(float frameTimeMs, bool sceneChanged, float dt)
        {
            StateTime += dt;

            // EMA smoothing
            if (_emaFrameTime <= 0)
                _emaFrameTime = frameTimeMs;
            else
                _emaFrameTime += 0.05f * (frameTimeMs - _emaFrameTime);

            // Establish baseline during Normal state
            if (State == ThermalState.Normal && _baselineFrameTime <= 0)
                _baselineFrameTime = _emaFrameTime;
            else if (State == ThermalState.Normal)
                _baselineFrameTime += 0.01f * (_emaFrameTime - _baselineFrameTime); // slow drift

            // Detect thermal spikes (only when scene hasn't changed)
            if (!sceneChanged && _baselineFrameTime > 0)
            {
                if (_emaFrameTime > _baselineFrameTime * SpikeRatio)
                    _spikeCount++;
                else
                    _spikeCount = Math.Max(0, _spikeCount - 1);
            }

            // State transitions
            switch (State)
            {
                case ThermalState.Normal:
                    if (_spikeCount >= WarningSpikes)
                        TransitionTo(ThermalState.Warning);
                    break;

                case ThermalState.Warning:
                    if (_spikeCount >= ThrottlingSpikes)
                        TransitionTo(ThermalState.Throttling);
                    else if (_spikeCount == 0 && StateTime > CooldownTime)
                        TransitionTo(ThermalState.Normal);
                    break;

                case ThermalState.Throttling:
                    if (_spikeCount >= CriticalSpikes)
                        TransitionTo(ThermalState.Critical);
                    else if (_spikeCount < WarningSpikes && StateTime > CooldownTime)
                        TransitionTo(ThermalState.Recovering);
                    break;

                case ThermalState.Critical:
                    if (_spikeCount < ThrottlingSpikes && StateTime > CooldownTime * 2)
                        TransitionTo(ThermalState.Recovering);
                    break;

                case ThermalState.Recovering:
                    _recoveryProgress = Math.Min(1f, _recoveryProgress + dt / RecoveryDuration);
                    if (_spikeCount >= WarningSpikes)
                        TransitionTo(ThermalState.Throttling); // relapse
                    else if (_recoveryProgress >= 1f)
                        TransitionTo(ThermalState.Normal);
                    break;
            }
        }

        private void TransitionTo(ThermalState newState)
        {
            State = newState;
            StateTime = 0;
            if (newState == ThermalState.Recovering)
                _recoveryProgress = 0.2f; // start at 20% quality
            if (newState == ThermalState.Normal)
                _spikeCount = 0;
        }

        /// <summary>Reset thermal manager to normal state.</summary>
        public void Reset()
        {
            State = ThermalState.Normal;
            StateTime = 0;
            _baselineFrameTime = 0;
            _emaFrameTime = 0;
            _spikeCount = 0;
            _recoveryProgress = 0;
        }
    }
}
