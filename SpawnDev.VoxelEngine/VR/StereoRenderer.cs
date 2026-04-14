using System.Numerics;

namespace SpawnDev.VoxelEngine.VR
{
    /// <summary>
    /// Stereo rendering support for VR headsets.
    ///
    /// Key optimization: shared mesh, two view-projection matrices.
    /// The voxel mesh is identical for both eyes - only the camera differs.
    /// Culling uses a COMBINED frustum (union of left + right) so sections
    /// visible to either eye are included. The vertex shader selects VP matrix
    /// per eye via gl_ViewID or render pass index.
    ///
    /// WebXR provides per-frame view/projection matrices via XRView.
    /// Quest 3S renders at 2064x2208 per eye (4128x2208 combined).
    /// </summary>
    public class StereoRenderer
    {
        /// <summary>Left eye view-projection matrix. Updated per frame from XRView.</summary>
        public Matrix4x4 LeftVP { get; set; } = Matrix4x4.Identity;

        /// <summary>Right eye view-projection matrix. Updated per frame from XRView.</summary>
        public Matrix4x4 RightVP { get; set; } = Matrix4x4.Identity;

        /// <summary>Combined frustum planes for culling (union of both eyes).</summary>
        public float[] CombinedFrustumPlanes { get; } = new float[24]; // 6 planes * 4 components

        /// <summary>Inter-pupillary distance in world units. Typically ~0.063m.</summary>
        public float IPD { get; set; } = 0.063f;

        /// <summary>Per-eye render target width.</summary>
        public int EyeWidth { get; set; } = 2064;

        /// <summary>Per-eye render target height.</summary>
        public int EyeHeight { get; set; } = 2208;

        /// <summary>Whether stereo rendering is active (XR session running).</summary>
        public bool IsActive { get; set; }

        /// <summary>
        /// Update view-projection matrices from WebXR frame data.
        /// Call at the start of each XR frame.
        /// </summary>
        /// <param name="leftView">Left eye view matrix (world -> eye space).</param>
        /// <param name="leftProjection">Left eye projection matrix.</param>
        /// <param name="rightView">Right eye view matrix.</param>
        /// <param name="rightProjection">Right eye projection matrix.</param>
        public void UpdateFromXRViews(
            Matrix4x4 leftView, Matrix4x4 leftProjection,
            Matrix4x4 rightView, Matrix4x4 rightProjection)
        {
            LeftVP = leftView * leftProjection;
            RightVP = rightView * rightProjection;

            // Compute combined frustum for culling (conservative union of both eyes)
            ComputeCombinedFrustum(leftView, leftProjection, rightView, rightProjection);

            IsActive = true;
        }

        /// <summary>
        /// Compute conservative combined frustum from both eye frustums.
        /// For each plane, take the one that's MORE permissive (further out)
        /// so that sections visible to either eye pass the test.
        /// </summary>
        private void ComputeCombinedFrustum(
            Matrix4x4 leftView, Matrix4x4 leftProj,
            Matrix4x4 rightView, Matrix4x4 rightProj)
        {
            var leftVP = leftView * leftProj;
            var rightVP = rightView * rightProj;

            // Extract planes from both VPs
            var leftPlanes = ExtractFrustumPlanes(leftVP);
            var rightPlanes = ExtractFrustumPlanes(rightVP);

            // For each plane, use the more permissive one
            // Left clip plane: use leftEye's left plane (it extends further left)
            // Right clip plane: use rightEye's right plane
            // Top/bottom/near/far: use whichever extends further
            for (int i = 0; i < 6; i++)
            {
                int baseIdx = i * 4;

                // Use the plane that's further from center (more permissive)
                // For left/right planes, each eye's outer plane is more permissive
                if (i == 0) // Left plane - use left eye's left plane
                {
                    CombinedFrustumPlanes[baseIdx + 0] = leftPlanes[baseIdx + 0];
                    CombinedFrustumPlanes[baseIdx + 1] = leftPlanes[baseIdx + 1];
                    CombinedFrustumPlanes[baseIdx + 2] = leftPlanes[baseIdx + 2];
                    CombinedFrustumPlanes[baseIdx + 3] = leftPlanes[baseIdx + 3];
                }
                else if (i == 1) // Right plane - use right eye's right plane
                {
                    CombinedFrustumPlanes[baseIdx + 0] = rightPlanes[baseIdx + 0];
                    CombinedFrustumPlanes[baseIdx + 1] = rightPlanes[baseIdx + 1];
                    CombinedFrustumPlanes[baseIdx + 2] = rightPlanes[baseIdx + 2];
                    CombinedFrustumPlanes[baseIdx + 3] = rightPlanes[baseIdx + 3];
                }
                else // Top, bottom, near, far - use more permissive (larger D)
                {
                    float dL = leftPlanes[baseIdx + 3];
                    float dR = rightPlanes[baseIdx + 3];
                    var src = MathF.Abs(dL) >= MathF.Abs(dR) ? leftPlanes : rightPlanes;
                    CombinedFrustumPlanes[baseIdx + 0] = src[baseIdx + 0];
                    CombinedFrustumPlanes[baseIdx + 1] = src[baseIdx + 1];
                    CombinedFrustumPlanes[baseIdx + 2] = src[baseIdx + 2];
                    CombinedFrustumPlanes[baseIdx + 3] = src[baseIdx + 3];
                }
            }
        }

        /// <summary>
        /// Extract 6 frustum planes from a VP matrix (Gribb-Hartmann method).
        /// Returns 24 floats: 6 planes * (nx, ny, nz, d).
        /// </summary>
        private static float[] ExtractFrustumPlanes(Matrix4x4 vp)
        {
            var planes = new float[24];

            // Row-major (System.Numerics post-multiply convention)
            // Left:   row3 + row0
            SetPlane(planes, 0, vp.M14 + vp.M11, vp.M24 + vp.M21, vp.M34 + vp.M31, vp.M44 + vp.M41);
            // Right:  row3 - row0
            SetPlane(planes, 1, vp.M14 - vp.M11, vp.M24 - vp.M21, vp.M34 - vp.M31, vp.M44 - vp.M41);
            // Bottom: row3 + row1
            SetPlane(planes, 2, vp.M14 + vp.M12, vp.M24 + vp.M22, vp.M34 + vp.M32, vp.M44 + vp.M42);
            // Top:    row3 - row1
            SetPlane(planes, 3, vp.M14 - vp.M12, vp.M24 - vp.M22, vp.M34 - vp.M32, vp.M44 - vp.M42);
            // Near:   row3 + row2
            SetPlane(planes, 4, vp.M14 + vp.M13, vp.M24 + vp.M23, vp.M34 + vp.M33, vp.M44 + vp.M43);
            // Far:    row3 - row2
            SetPlane(planes, 5, vp.M14 - vp.M13, vp.M24 - vp.M23, vp.M34 - vp.M33, vp.M44 - vp.M43);

            return planes;
        }

        private static void SetPlane(float[] planes, int index, float a, float b, float c, float d)
        {
            float len = MathF.Sqrt(a * a + b * b + c * c);
            if (len > 0)
            {
                float inv = 1f / len;
                a *= inv; b *= inv; c *= inv; d *= inv;
            }
            int i = index * 4;
            planes[i] = a;
            planes[i + 1] = b;
            planes[i + 2] = c;
            planes[i + 3] = d;
        }

        /// <summary>
        /// Get the center eye position (midpoint between left and right eye).
        /// Used for LOD selection and draw ordering.
        /// </summary>
        public Vector3 GetCenterEyePosition(Matrix4x4 leftView, Matrix4x4 rightView)
        {
            // Eye position = inverse of the view matrix translation column
            Matrix4x4.Invert(leftView, out var invLeft);
            Matrix4x4.Invert(rightView, out var invRight);

            var leftPos = new Vector3(invLeft.M41, invLeft.M42, invLeft.M43);
            var rightPos = new Vector3(invRight.M41, invRight.M42, invRight.M43);

            return (leftPos + rightPos) * 0.5f;
        }
    }
}
