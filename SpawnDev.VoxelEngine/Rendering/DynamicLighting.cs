using System.Numerics;
using System.Runtime.InteropServices;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Dynamic point/spot/directional light for voxel scenes.
    ///
    /// Up to 16 active lights uploaded as a uniform buffer each frame.
    /// Fragment shader iterates all lights, computes attenuation, accumulates contribution.
    ///
    /// Attenuation: 1.0 / (1.0 + dist * dist * falloff)
    /// This gives physically plausible inverse-square falloff with configurable steepness.
    ///
    /// Light types:
    /// - Point: torch, campfire, glowstone. Omnidirectional.
    /// - Spot: flashlight, searchlight. Cone angle + direction.
    /// - Directional: sun, moon. No position, just direction. One per scene.
    /// </summary>
    public static class DynamicLighting
    {
        public const int MaxLights = 16;

        /// <summary>GPU-uploadable light data. 64 bytes per light, 1KB for 16 lights.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct LightData
        {
            /// <summary>World position (point/spot) or direction (directional). W = light type (0=point, 1=spot, 2=directional).</summary>
            public Vector4 PositionAndType;

            /// <summary>Light color RGB + intensity in W.</summary>
            public Vector4 ColorAndIntensity;

            /// <summary>Spot direction XYZ + spot cone angle (cosine of half-angle) in W. Unused for point/directional.</summary>
            public Vector4 DirectionAndCone;

            /// <summary>X = radius (max range), Y = falloff factor, Z = inner cone (for soft edge), W = reserved.</summary>
            public Vector4 RangeAndFalloff;

            public static LightData CreatePoint(Vector3 position, Vector3 color, float intensity, float radius, float falloff = 1f)
            {
                return new LightData
                {
                    PositionAndType = new Vector4(position, 0),
                    ColorAndIntensity = new Vector4(color, intensity),
                    DirectionAndCone = Vector4.Zero,
                    RangeAndFalloff = new Vector4(radius, falloff, 0, 0),
                };
            }

            public static LightData CreateSpot(Vector3 position, Vector3 direction, Vector3 color,
                float intensity, float radius, float coneAngleDegrees, float falloff = 1f)
            {
                float outerCos = MathF.Cos(coneAngleDegrees * MathF.PI / 180f * 0.5f);
                float innerCos = MathF.Cos(coneAngleDegrees * MathF.PI / 180f * 0.35f); // soft edge at 70% of cone
                var dir = Vector3.Normalize(direction);
                return new LightData
                {
                    PositionAndType = new Vector4(position, 1),
                    ColorAndIntensity = new Vector4(color, intensity),
                    DirectionAndCone = new Vector4(dir, outerCos),
                    RangeAndFalloff = new Vector4(radius, falloff, innerCos, 0),
                };
            }

            public static LightData CreateDirectional(Vector3 direction, Vector3 color, float intensity)
            {
                var dir = Vector3.Normalize(direction);
                return new LightData
                {
                    PositionAndType = new Vector4(dir, 2),
                    ColorAndIntensity = new Vector4(color, intensity),
                    DirectionAndCone = Vector4.Zero,
                    RangeAndFalloff = new Vector4(float.MaxValue, 0, 0, 0),
                };
            }
        }

        /// <summary>
        /// Compute light contribution at a surface point (CPU reference for testing).
        /// </summary>
        public static Vector3 ComputeLighting(
            Vector3 surfacePos, Vector3 surfaceNormal,
            ReadOnlySpan<LightData> lights, int lightCount,
            Vector3 ambientColor)
        {
            var result = ambientColor;

            for (int i = 0; i < lightCount && i < MaxLights; i++)
            {
                var light = lights[i];
                int type = (int)light.PositionAndType.W;
                var color = new Vector3(light.ColorAndIntensity.X, light.ColorAndIntensity.Y, light.ColorAndIntensity.Z);
                float intensity = light.ColorAndIntensity.W;

                Vector3 lightDir;
                float attenuation;

                if (type == 2) // Directional
                {
                    lightDir = -new Vector3(light.PositionAndType.X, light.PositionAndType.Y, light.PositionAndType.Z);
                    attenuation = 1f;
                }
                else // Point or Spot
                {
                    var lightPos = new Vector3(light.PositionAndType.X, light.PositionAndType.Y, light.PositionAndType.Z);
                    var toLight = lightPos - surfacePos;
                    float dist = toLight.Length();
                    float radius = light.RangeAndFalloff.X;
                    float falloff = light.RangeAndFalloff.Y;

                    if (dist > radius) continue;

                    lightDir = toLight / dist;
                    attenuation = 1f / (1f + dist * dist * falloff);

                    // Spot cone falloff
                    if (type == 1)
                    {
                        var spotDir = new Vector3(light.DirectionAndCone.X, light.DirectionAndCone.Y, light.DirectionAndCone.Z);
                        float outerCos = light.DirectionAndCone.W;
                        float innerCos = light.RangeAndFalloff.Z;
                        float dot = Vector3.Dot(-lightDir, spotDir);
                        if (dot < outerCos) continue; // outside cone
                        float spotFactor = Math.Clamp((dot - outerCos) / (innerCos - outerCos), 0f, 1f);
                        attenuation *= spotFactor;
                    }
                }

                // Lambertian diffuse
                float NdotL = MathF.Max(0, Vector3.Dot(surfaceNormal, lightDir));
                result += color * intensity * NdotL * attenuation;
            }

            return result;
        }
    }
}
