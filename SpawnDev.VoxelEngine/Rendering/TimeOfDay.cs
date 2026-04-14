using System.Numerics;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Time-of-day system controlling ambient light, sun direction, fog color, and sky color.
    ///
    /// Time is normalized 0-1 where:
    ///   0.0  = midnight
    ///   0.25 = dawn (6 AM)
    ///   0.5  = noon
    ///   0.75 = dusk (6 PM)
    ///
    /// The system outputs a uniform buffer with all lighting parameters.
    /// Fragment shaders read these to compute final colors.
    ///
    /// Night in Lost Spawns is DARK - ambient drops to near-zero.
    /// Only point lights (torches, campfires) provide visibility.
    /// NVG post-process mode: green phosphor filter + noise grain + limited range.
    /// </summary>
    public class TimeOfDay
    {
        /// <summary>Current time (0-1). Set this each frame from game clock.</summary>
        public float Time { get; set; } = 0.5f; // default: noon

        /// <summary>Day cycle speed. 1.0 = real-time (24 min cycle at 60fps). 0 = frozen.</summary>
        public float CycleSpeed { get; set; } = 0f;

        /// <summary>Whether night vision mode is active.</summary>
        public bool NightVisionEnabled { get; set; }

        /// <summary>NVG effective range in world units.</summary>
        public float NightVisionRange { get; set; } = 30f;

        /// <summary>
        /// Advance time by delta. Call each frame.
        /// At CycleSpeed = 1.0, a full day takes 24 minutes (1440 seconds).
        /// </summary>
        public void Update(float dt)
        {
            if (CycleSpeed <= 0) return;
            Time += dt * CycleSpeed / 1440f;
            if (Time >= 1f) Time -= 1f;
        }

        /// <summary>Sun direction vector (normalized). Points FROM the sun TO the scene.</summary>
        public Vector3 SunDirection
        {
            get
            {
                // Sun orbits in the XY plane, Time=0.5 = directly overhead
                float angle = (Time - 0.25f) * MathF.PI * 2f; // 0.25 = horizon, 0.5 = zenith
                return Vector3.Normalize(new Vector3(
                    MathF.Cos(angle),
                    -MathF.Sin(angle), // negative = pointing down from sky
                    0.3f)); // slight Z offset for more interesting shadows
            }
        }

        /// <summary>Ambient light color and intensity based on time of day.</summary>
        public Vector3 AmbientColor
        {
            get
            {
                float sunHeight = GetSunHeight();

                if (sunHeight <= -0.1f) // Night
                    return new Vector3(0.02f, 0.02f, 0.04f); // near-black with slight blue

                if (sunHeight <= 0.1f) // Dawn/dusk transition
                {
                    float t = (sunHeight + 0.1f) / 0.2f; // 0 at -0.1, 1 at +0.1
                    var nightColor = new Vector3(0.02f, 0.02f, 0.04f);
                    var dawnColor = new Vector3(0.3f, 0.2f, 0.15f); // warm orange
                    return Vector3.Lerp(nightColor, dawnColor, t);
                }

                // Day: ramp from dawn warm to noon neutral
                float dayT = Math.Clamp((sunHeight - 0.1f) / 0.9f, 0, 1);
                var dawn = new Vector3(0.3f, 0.2f, 0.15f);
                var noon = new Vector3(0.4f, 0.45f, 0.5f); // cool neutral daylight
                return Vector3.Lerp(dawn, noon, dayT);
            }
        }

        /// <summary>Sun light color (directional light color).</summary>
        public Vector3 SunColor
        {
            get
            {
                float sunHeight = GetSunHeight();
                if (sunHeight <= 0) return Vector3.Zero; // no sun below horizon

                if (sunHeight < 0.2f)
                {
                    float t = sunHeight / 0.2f;
                    return Vector3.Lerp(new Vector3(1f, 0.5f, 0.2f), new Vector3(1f, 0.95f, 0.85f), t) * t;
                }

                return new Vector3(1f, 0.95f, 0.85f); // warm white daylight
            }
        }

        /// <summary>Sun intensity (0 at night, 1 at noon).</summary>
        public float SunIntensity
        {
            get
            {
                float sunHeight = GetSunHeight();
                if (sunHeight <= 0) return 0;
                return Math.Clamp(sunHeight * 2f, 0, 1);
            }
        }

        /// <summary>Fog color based on time of day. Matches sky color at horizon.</summary>
        public Vector3 FogColor
        {
            get
            {
                float sunHeight = GetSunHeight();
                if (sunHeight <= -0.1f) return new Vector3(0.01f, 0.01f, 0.02f); // dark blue-black
                if (sunHeight <= 0.1f)
                {
                    float t = (sunHeight + 0.1f) / 0.2f;
                    return Vector3.Lerp(new Vector3(0.01f, 0.01f, 0.02f), new Vector3(0.6f, 0.4f, 0.3f), t);
                }
                float dayT = Math.Clamp((sunHeight - 0.1f) / 0.5f, 0, 1);
                return Vector3.Lerp(new Vector3(0.6f, 0.4f, 0.3f), new Vector3(0.6f, 0.7f, 0.85f), dayT);
            }
        }

        /// <summary>Sky zenith color for clear sky rendering.</summary>
        public Vector3 SkyColor
        {
            get
            {
                float sunHeight = GetSunHeight();
                if (sunHeight <= -0.1f) return new Vector3(0.0f, 0.0f, 0.02f); // near-black
                if (sunHeight <= 0.1f)
                {
                    float t = (sunHeight + 0.1f) / 0.2f;
                    return Vector3.Lerp(new Vector3(0.0f, 0.0f, 0.02f), new Vector3(0.2f, 0.3f, 0.6f), t);
                }
                return new Vector3(0.3f, 0.5f, 0.9f); // clear blue sky
            }
        }

        /// <summary>Whether it's currently dark enough that only light sources are visible.</summary>
        public bool IsDark => GetSunHeight() <= -0.05f;

        /// <summary>Whether dawn/dusk transition is active.</summary>
        public bool IsTransition => MathF.Abs(GetSunHeight()) < 0.15f;

        /// <summary>Sun height above horizon. Negative = below horizon (night).</summary>
        private float GetSunHeight()
        {
            float angle = (Time - 0.25f) * MathF.PI * 2f;
            return MathF.Sin(angle);
        }

        /// <summary>
        /// NVG post-process parameters.
        /// Fragment shader applies: green channel boost, noise grain, range falloff.
        /// </summary>
        public (float greenBoost, float noiseIntensity, float range) GetNVGParams()
        {
            if (!NightVisionEnabled) return (0, 0, 0);
            return (2.5f, 0.05f, NightVisionRange);
        }
    }
}
