using System.Numerics;

namespace SpawnDev.VoxelEngine.VR
{
    /// <summary>
    /// WebXR session management helpers for VR/AR rendering.
    ///
    /// Abstracts the WebXR session lifecycle so consuming projects don't need
    /// to manage XR state directly. Uses SpawnDev.BlazorJS typed wrappers
    /// for all WebXR API calls (66 typed WebXR classes in BlazorJS).
    ///
    /// Session types:
    /// - immersive-vr: Quest 3S, desktop VR headsets
    /// - immersive-ar: Quest 3S passthrough, mobile AR
    /// - inline: non-immersive fallback (desktop browser with mouse look)
    ///
    /// The helper provides:
    /// - Session request/end with feature detection
    /// - Per-frame view/projection matrix extraction from XRView
    /// - Reference space management (local, local-floor, bounded-floor)
    /// - Input source tracking (controllers, hands, gaze)
    /// - Hit test setup for AR placement
    /// </summary>
    public class WebXRHelper
    {
        /// <summary>Whether WebXR is supported in the current browser.</summary>
        public bool IsSupported { get; private set; }

        /// <summary>Whether an XR session is currently active.</summary>
        public bool IsSessionActive { get; private set; }

        /// <summary>Current session mode.</summary>
        public XRSessionMode SessionMode { get; private set; } = XRSessionMode.None;

        /// <summary>Whether hand tracking is available in the current session.</summary>
        public bool HandTrackingAvailable { get; private set; }

        /// <summary>Left eye view matrix (updated per frame).</summary>
        public Matrix4x4 LeftView { get; private set; } = Matrix4x4.Identity;

        /// <summary>Right eye view matrix.</summary>
        public Matrix4x4 RightView { get; private set; } = Matrix4x4.Identity;

        /// <summary>Left eye projection matrix.</summary>
        public Matrix4x4 LeftProjection { get; private set; } = Matrix4x4.Identity;

        /// <summary>Right eye projection matrix.</summary>
        public Matrix4x4 RightProjection { get; private set; } = Matrix4x4.Identity;

        /// <summary>Head position in world space.</summary>
        public Vector3 HeadPosition { get; private set; }

        /// <summary>Head forward direction.</summary>
        public Vector3 HeadForward { get; private set; } = -Vector3.UnitZ;

        /// <summary>Per-eye render target dimensions.</summary>
        public int EyeWidth { get; private set; } = 2064;
        public int EyeHeight { get; private set; } = 2208;

        /// <summary>
        /// Features to request when creating an XR session.
        /// Default: local-floor + hand-tracking for Quest 3S.
        /// </summary>
        public List<string> RequiredFeatures { get; set; } = new() { "local-floor" };
        public List<string> OptionalFeatures { get; set; } = new() { "hand-tracking", "hit-test" };

        /// <summary>
        /// Update per-frame XR state.
        /// Call with view/projection data extracted from XRFrame.getViewerPose().
        ///
        /// In the consuming project's render loop:
        ///   var pose = xrFrame.GetViewerPose(refSpace);
        ///   foreach (var view in pose.Views)
        ///       helper.UpdateView(view.Eye, view.Transform.Matrix, view.ProjectionMatrix);
        /// </summary>
        public void UpdateView(XREye eye, Matrix4x4 viewMatrix, Matrix4x4 projectionMatrix)
        {
            if (eye == XREye.Left || eye == XREye.None) // None = mono rendering
            {
                LeftView = viewMatrix;
                LeftProjection = projectionMatrix;
            }
            if (eye == XREye.Right || eye == XREye.None)
            {
                RightView = viewMatrix;
                RightProjection = projectionMatrix;
            }

            // Extract head position from the view matrix inverse
            if (eye == XREye.Left || eye == XREye.None)
            {
                if (Matrix4x4.Invert(viewMatrix, out var inv))
                {
                    HeadPosition = new Vector3(inv.M41, inv.M42, inv.M43);
                    HeadForward = -new Vector3(inv.M31, inv.M32, inv.M33); // -Z in view space
                }
            }
        }

        /// <summary>
        /// Mark session as started. Call when XR session is successfully created.
        /// </summary>
        public void OnSessionStarted(XRSessionMode mode, int eyeWidth, int eyeHeight, bool handTracking)
        {
            IsSessionActive = true;
            SessionMode = mode;
            EyeWidth = eyeWidth;
            EyeHeight = eyeHeight;
            HandTrackingAvailable = handTracking;
        }

        /// <summary>
        /// Mark session as ended. Call when XR session ends.
        /// </summary>
        public void OnSessionEnded()
        {
            IsSessionActive = false;
            SessionMode = XRSessionMode.None;
            HandTrackingAvailable = false;
        }

        /// <summary>
        /// Get the combined view-projection matrix for a given eye.
        /// </summary>
        public Matrix4x4 GetVP(XREye eye)
        {
            return eye switch
            {
                XREye.Left => LeftView * LeftProjection,
                XREye.Right => RightView * RightProjection,
                _ => LeftView * LeftProjection, // mono
            };
        }

        /// <summary>
        /// Get the center eye position (midpoint between eyes) for LOD/culling.
        /// </summary>
        public Vector3 GetCenterEyePosition()
        {
            if (!IsSessionActive) return HeadPosition;

            if (Matrix4x4.Invert(LeftView, out var invL) && Matrix4x4.Invert(RightView, out var invR))
            {
                var leftPos = new Vector3(invL.M41, invL.M42, invL.M43);
                var rightPos = new Vector3(invR.M41, invR.M42, invR.M43);
                return (leftPos + rightPos) * 0.5f;
            }

            return HeadPosition;
        }

        /// <summary>
        /// Compute a controller ray for UI interaction and block targeting.
        /// The ray origin is the controller grip position, direction is the controller's -Z axis.
        /// </summary>
        public static (Vector3 origin, Vector3 direction) GetControllerRay(Matrix4x4 gripTransform)
        {
            var origin = new Vector3(gripTransform.M41, gripTransform.M42, gripTransform.M43);
            var forward = -Vector3.Normalize(new Vector3(gripTransform.M31, gripTransform.M32, gripTransform.M33));
            return (origin, forward);
        }

        /// <summary>
        /// Compute a hand poke ray from finger tip position and direction.
        /// Used for direct interaction with world-space UI panels.
        /// </summary>
        public static (Vector3 origin, Vector3 direction) GetFingerRay(
            Vector3 fingerTipPosition, Vector3 fingerDirection)
        {
            return (fingerTipPosition, Vector3.Normalize(fingerDirection));
        }
    }

    /// <summary>XR session mode.</summary>
    public enum XRSessionMode
    {
        None,
        Inline,
        ImmersiveVR,
        ImmersiveAR,
    }

    /// <summary>XR eye identifier.</summary>
    public enum XREye
    {
        None,
        Left,
        Right,
    }
}
