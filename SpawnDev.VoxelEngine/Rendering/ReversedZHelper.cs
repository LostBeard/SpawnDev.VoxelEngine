using System.Numerics;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Reversed-Z depth buffer utilities.
    ///
    /// Standard Z: near=0, far=1, precision concentrated at near plane.
    /// Reversed Z: near=1, far=0, precision distributed more evenly.
    ///
    /// With float32 depth, reversed-Z gives near-linear precision distribution,
    /// eliminating z-fighting at medium-to-far distances. Combined with infinite
    /// far plane, the only parameter needed is zNear.
    ///
    /// WebGPU config:
    ///   depthFormat: "depth32float"
    ///   depthCompare: "greater" (not "less" - reversed!)
    ///   clearValue: 0.0 (far plane, not 1.0)
    /// </summary>
    public static class ReversedZHelper
    {
        /// <summary>
        /// Create an infinite far plane perspective projection matrix with reversed Z.
        /// Near plane maps to Z=1, infinity maps to Z=0.
        ///
        /// This replaces Matrix4x4.CreatePerspectiveFieldOfView which uses standard Z.
        /// </summary>
        /// <param name="fovY">Vertical field of view in radians.</param>
        /// <param name="aspect">Aspect ratio (width / height).</param>
        /// <param name="zNear">Near clip plane distance (must be > 0).</param>
        public static Matrix4x4 CreateInfinitePerspective(float fovY, float aspect, float zNear)
        {
            float f = 1f / MathF.Tan(fovY * 0.5f);

            // Infinite reversed-Z projection:
            // Maps [zNear, inf) to [1, 0)
            // Row-major (System.Numerics convention, post-multiply: clipPos = worldPos * VP)
            return new Matrix4x4(
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, 0, -1,
                0, 0, zNear, 0);
        }

        /// <summary>
        /// Create a finite reversed-Z perspective projection matrix.
        /// Near maps to Z=1, far maps to Z=0.
        /// Use when infinite far plane is not desired (e.g., fog cutoff).
        /// </summary>
        public static Matrix4x4 CreateReversedPerspective(float fovY, float aspect, float zNear, float zFar)
        {
            float f = 1f / MathF.Tan(fovY * 0.5f);
            float rangeInv = 1f / (zNear - zFar); // note: near - far, not far - near

            return new Matrix4x4(
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, zFar * rangeInv, -1,
                0, 0, zNear * zFar * rangeInv, 0);
        }

        /// <summary>
        /// Linearize a reversed-Z depth value back to view-space distance.
        /// Useful for fog, depth-of-field, debug visualization.
        /// </summary>
        /// <param name="depth">Depth buffer value (0 = far, 1 = near).</param>
        /// <param name="zNear">Near plane distance used when creating the projection.</param>
        /// <returns>Linear distance from camera in world units.</returns>
        public static float LinearizeDepth(float depth, float zNear)
        {
            // For infinite reversed-Z: depth = zNear / viewZ
            // Therefore: viewZ = zNear / depth
            if (depth <= 0) return float.MaxValue; // at infinity
            return zNear / depth;
        }

        /// <summary>
        /// Linearize depth for finite reversed-Z projection.
        /// </summary>
        public static float LinearizeDepth(float depth, float zNear, float zFar)
        {
            // Reversed: depth = (zFar * (zNear - viewZ)) / (viewZ * (zNear - zFar))
            // Solve for viewZ:
            float num = zNear * zFar;
            float den = zFar + depth * (zNear - zFar);
            if (MathF.Abs(den) < 1e-10f) return float.MaxValue;
            return num / den;
        }

        /// <summary>WebGPU depth format for reversed-Z.</summary>
        public const string DepthFormat = "depth32float";

        /// <summary>WebGPU depth compare function for reversed-Z.</summary>
        public const string DepthCompare = "greater";

        /// <summary>WebGPU depth clear value for reversed-Z (far plane = 0).</summary>
        public const float DepthClearValue = 0f;
    }
}
