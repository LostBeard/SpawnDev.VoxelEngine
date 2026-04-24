using System.Numerics;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Matrix helpers for the WebXR -> VertexPullPipeline bridge. These are the VR-side
    /// analogue of what a desktop renderer does: take the WebXR runtime's column-major
    /// GL-style matrix data and convert to the .NET row-vector + reversed-Z form that
    /// VertexPullPipeline.UniformData.MVP expects (row-major .NET bytes reinterpreted as
    /// column-major mat4x4 by WGSL).
    ///
    /// Consumers:
    ///  - Lost Spawns `VrPrototype.razor` (the Phase C Quest 3S shipping path).
    ///  - AubsCraft VR (when Phase D ships; planned).
    ///  - SpawnScene VR (if/when VR landing is scoped).
    ///
    /// Design note: VertexPullPipeline's MVP is explicitly documented as "raw Matrix4x4
    /// in row-vector convention, bytes reinterpret as column-major mat4x4 = natural
    /// transpose" (see VertexPullPipeline.UniformData.MVP XML doc). DO NOT use
    /// GpuMatrix4x4 here - it would double-transpose and break rendering.
    /// </summary>
    public static class VrMatrixHelpers
    {
        /// <summary>
        /// Converts a 16-element GL-style column-major matrix array (as returned by
        /// WebXR's `XRView.ProjectionMatrix.ToArray()` and `XRRigidTransform.Matrix.ToArray()`)
        /// to a .NET `Matrix4x4` whose in-memory bytes are the original column-major layout.
        /// Because .NET Matrix4x4 is row-major, reinterpreting the bytes this way is
        /// semantically the transpose - which is exactly the row-vector convention needed
        /// for VertexPullPipeline's MVP input.
        /// </summary>
        /// <param name="a">16-element column-major matrix from GL/WebXR, or null.</param>
        /// <returns>.NET row-vector Matrix4x4 (== source^T).</returns>
        public static Matrix4x4 GlColumnMajorToDotNetMatrix(float[]? a)
        {
            if (a == null || a.Length < 16) return Matrix4x4.Identity;
            return new Matrix4x4(
                a[0],  a[1],  a[2],  a[3],
                a[4],  a[5],  a[6],  a[7],
                a[8],  a[9],  a[10], a[11],
                a[12], a[13], a[14], a[15]);
        }

        /// <summary>
        /// Row-vector post-multiply matrix that converts an OpenGL-convention projection
        /// (NDC z in [-1,+1], WebXR emits this for WebGL-hosted XR layers) to reversed-Z
        /// as consumed by VertexPullPipeline (NDC z in [1,0], depthCompare=greater,
        /// clearDepth=0, depth32float).
        ///
        /// Math: z_ndc_rev = (1 - z_ndc_gl) / 2. Only Z is affected:
        ///   near plane (z_ndc_gl = -1) -> z_ndc_rev = 1
        ///   far plane  (z_ndc_gl = +1) -> z_ndc_rev = 0
        ///   midpoint   (z_ndc_gl =  0) -> z_ndc_rev = 0.5
        /// In clip space: z_clip_rev = (w - z_clip_gl) / 2, w unchanged, x/y unchanged.
        ///
        /// A naive (w - z) / 1 formula (without the /2) assumes WebGPU-convention input
        /// (NDC z in [0,1]) and produces out-of-range values (up to 2) on WebXR's OpenGL
        /// projections, which clamp to 1 on the GPU and collapse the near half of the
        /// frustum into a single depth plane. The /2 keeps the full depth range usable.
        ///
        /// Use as: `vpRowReversed = vRow * pRow * ZFlipStandardToReversedZ`.
        /// </summary>
        public static readonly Matrix4x4 ZFlipStandardToReversedZ = new(
            1, 0,   0,    0,
            0, 1,   0,    0,
            0, 0,  -0.5f, 0,
            0, 0,   0.5f, 1);

        /// <summary>
        /// One-shot helper for the full WebXR -> VertexPullPipeline MVP pipeline. Given
        /// the WebXR-provided proj + view column-major 16-float arrays, returns the
        /// .NET Matrix4x4 suitable to pass as `VertexPullPipeline.UniformData.MVP`:
        /// ready for WGSL column-major reinterpretation, reversed-Z, row-vector math.
        /// Callers still supply model / sectionOffset separately via the uniform struct.
        /// </summary>
        /// <param name="projColumnMajor">XRView.ProjectionMatrix as float[16], standard-Z.</param>
        /// <param name="viewColumnMajor">XRView.Transform.Inverse.Matrix as float[16].</param>
        /// <returns>View*Projection*ZFlip in the VertexPullPipeline's expected form.</returns>
        public static Matrix4x4 BuildReversedZViewProj(float[]? projColumnMajor, float[]? viewColumnMajor)
        {
            var p = GlColumnMajorToDotNetMatrix(projColumnMajor);
            var v = GlColumnMajorToDotNetMatrix(viewColumnMajor);
            return v * p * ZFlipStandardToReversedZ;
        }

        /// <summary>
        /// Converts a thumbstick input (stickX, stickY) into a world-axis locomotion delta,
        /// integrated through the player's current Y-yaw so that pushing the stick "forward"
        /// (stickY = -1 per Quest/Oculus convention: stick up = axes[3] = -1) moves the
        /// player along their current facing direction, not the world-Z axis.
        ///
        /// Stick convention:
        ///   stickX in [-1, +1], + = right, - = left
        ///   stickY in [-1, +1], - = up / forward, + = down / backward (Quest axes[3] sign)
        ///
        /// Math:
        ///   forward = RotY(yaw) * (0, 0, -1)
        ///   right   = RotY(yaw) * (1, 0, 0)
        ///   delta   = (-stickY * forward + stickX * right) * speed * dt
        /// </summary>
        /// <param name="stickX">Right-stick X, already dead-zoned and clamped to [-1, +1].</param>
        /// <param name="stickY">Right-stick Y (Quest-sign), already dead-zoned.</param>
        /// <param name="yawRad">Current accumulated yaw, radians, right-hand rule about +Y.</param>
        /// <param name="speedMetersPerSecond">Movement speed at full stick deflection.</param>
        /// <param name="dtSec">Frame delta in seconds. Clamp upstream if hiccups are possible.</param>
        public static Vector3 ComputeLocomotionDelta(
            float stickX, float stickY, float yawRad, float speedMetersPerSecond, float dtSec)
        {
            if (dtSec <= 0) return Vector3.Zero;
            if (stickX == 0 && stickY == 0) return Vector3.Zero;

            var q = Quaternion.CreateFromAxisAngle(Vector3.UnitY, yawRad);
            var forward = Vector3.Transform(new Vector3(0, 0, -1), q);
            var right = Vector3.Transform(new Vector3(1, 0, 0), q);
            return (-stickY * forward + stickX * right) * speedMetersPerSecond * dtSec;
        }

        /// <summary>
        /// Snap-turn debounce state machine. Typical VR comfort: left thumbstick X past
        /// a snap threshold triggers a single ±angle rotation, the stick must return near
        /// center before another snap can fire (prevents continuous spin).
        ///
        /// Convention: stickX > 0 = stick-right = turn RIGHT = yaw decreases (right-hand
        /// rule about +Y means CW-from-above is negative). stickX < 0 = turn LEFT.
        /// </summary>
        /// <param name="armed">State: true when stick has been released since last snap.</param>
        /// <param name="yawRad">Accumulated yaw in radians. Mutated on snap.</param>
        /// <param name="stickX">Current stick X (already dead-zoned if desired, but not required).</param>
        /// <param name="snapThresh">Firing threshold. Default 0.7 matches VR standard.</param>
        /// <param name="releaseThresh">Re-arm threshold. Must be lower than snapThresh.</param>
        /// <param name="snapAngleRad">Snap step in radians. Default 30 degrees.</param>
        /// <returns>True if a snap fired this call; false otherwise.</returns>
        public static bool TryApplySnapTurn(
            ref bool armed, ref float yawRad,
            float stickX,
            float snapThresh = 0.7f,
            float releaseThresh = 0.3f,
            float snapAngleRad = MathF.PI / 6f)
        {
            if (!armed && MathF.Abs(stickX) < releaseThresh) armed = true;
            if (!armed) return false;
            if (stickX > snapThresh)
            {
                yawRad -= snapAngleRad;
                armed = false;
                return true;
            }
            if (stickX < -snapThresh)
            {
                yawRad += snapAngleRad;
                armed = false;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Ray-plane intersection against the Y=0 floor, bounded by a square half-size. Used by
        /// the WebXR controller target-ray -> floor hit marker lookup. Given a controller's
        /// ray-space matrix (4x4, GL column-major, stored as float[16]), the origin is at
        /// column 3 (m[12..14]) and the forward direction is -column 2 (m[8..10]).
        ///
        /// Returns false when:
        ///  - the ray is parallel to or above the floor (dy >= 0),
        ///  - the hit would be behind the controller (t &lt;= 0),
        ///  - the hit is farther than 100m (defensive cap - otherwise a near-horizontal ray
        ///    can project to a huge distance and make the out-of-bounds check meaningless),
        ///  - the hit falls outside the [-floorHalf, +floorHalf] XZ square.
        /// </summary>
        /// <param name="rayMatrix4x4">16 floats, GL column-major. If null/short, returns false.</param>
        /// <param name="floorHalfMeters">Half-size of the square floor centered at world origin.</param>
        /// <param name="hitX">Output X of the hit point on the Y=0 plane.</param>
        /// <param name="hitZ">Output Z of the hit point on the Y=0 plane.</param>
        public static bool RayFloorHit(float[]? rayMatrix4x4, float floorHalfMeters, out float hitX, out float hitZ)
        {
            hitX = 0f; hitZ = 0f;
            if (rayMatrix4x4 == null || rayMatrix4x4.Length < 16) return false;
            float ox = rayMatrix4x4[12], oy = rayMatrix4x4[13], oz = rayMatrix4x4[14];
            float dx = -rayMatrix4x4[8], dy = -rayMatrix4x4[9], dz = -rayMatrix4x4[10];
            if (dy >= -1e-5f) return false;             // pointing up or parallel
            float t = -oy / dy;
            if (t <= 0f || t > 100f) return false;
            hitX = ox + t * dx;
            hitZ = oz + t * dz;
            return MathF.Abs(hitX) <= floorHalfMeters && MathF.Abs(hitZ) <= floorHalfMeters;
        }
    }
}
