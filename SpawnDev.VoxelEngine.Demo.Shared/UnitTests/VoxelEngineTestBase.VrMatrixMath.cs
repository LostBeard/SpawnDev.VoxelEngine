using System.Numerics;
using ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Rendering;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // VR/WebXR matrix helper tests for the hybrid bridge path
    // (SpawnDev.VoxelEngine.Rendering.VrMatrixHelpers).
    // Pure-math: don't touch the accelerator, but run once per backend because the
    // shared harness structure already iterates every backend for every [TestMethod].
    public partial class VoxelEngineTestBase
    {
        // ------------------------------------------------------------------
        // GlColumnMajorToDotNetMatrix
        // ------------------------------------------------------------------

        [TestMethod]
        public async Task VrMatrix_GlColumnMajor_IdentityInIdentityOut() => await RunTest(async accelerator =>
        {
            // GL identity, column-major stored.
            var arr = new float[16] { 1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1 };
            var m = VrMatrixHelpers.GlColumnMajorToDotNetMatrix(arr);
            if (!m.Equals(Matrix4x4.Identity))
                throw new Exception($"Expected identity, got:\n{m}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_GlColumnMajor_Null_ReturnsIdentity() => await RunTest(async accelerator =>
        {
            var m = VrMatrixHelpers.GlColumnMajorToDotNetMatrix(null);
            if (!m.Equals(Matrix4x4.Identity))
                throw new Exception("null input should produce identity matrix");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_GlColumnMajor_ShortArray_ReturnsIdentity() => await RunTest(async accelerator =>
        {
            var m = VrMatrixHelpers.GlColumnMajorToDotNetMatrix(new float[] { 1, 2, 3 });
            if (!m.Equals(Matrix4x4.Identity))
                throw new Exception("short array should produce identity fallback matrix");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_GlColumnMajor_TransposeSemantic() => await RunTest(async accelerator =>
        {
            // GL column-major P with M33=-1.222 and M43=-2.222 (standard perspective shape).
            // Column 0: (1, 0, 0, 0)
            // Column 1: (0, 1, 0, 0)
            // Column 2: (0, 0, -1.222, -1)    <- rows 0..3 of col 2
            // Column 3: (0, 0, -2.222, 0)
            var arr = new float[16]
            {
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, -1.222f, -1f,
                0f, 0f, -2.222f, 0f,
            };
            var m = VrMatrixHelpers.GlColumnMajorToDotNetMatrix(arr);

            // .NET row-major Matrix4x4: reading arr in order places arr[col*4+row] at position
            // .NET (row=col, col=row) semantically - i.e., the transpose. So .NET M11=1 (ok),
            // .NET M33 should equal GL col 2 row 2 = -1.222, and .NET M34 should equal GL col
            // 2 row 3 = -1, .NET M43 = GL col 3 row 2 = -2.222.
            if (MathF.Abs(m.M33 - (-1.222f)) > 1e-5f)
                throw new Exception($"M33 wrong: expected -1.222, got {m.M33}");
            if (MathF.Abs(m.M34 - (-1f)) > 1e-5f)
                throw new Exception($"M34 wrong (GL's M43 by transpose): expected -1, got {m.M34}");
            if (MathF.Abs(m.M43 - (-2.222f)) > 1e-5f)
                throw new Exception($"M43 wrong (GL's M34 by transpose): expected -2.222, got {m.M43}");
            if (MathF.Abs(m.M44 - 0f) > 1e-5f)
                throw new Exception($"M44 wrong: expected 0, got {m.M44}");
            await Task.CompletedTask;
        });

        // ------------------------------------------------------------------
        // ZFlipStandardToReversedZ
        // ------------------------------------------------------------------

        [TestMethod]
        public async Task VrMatrix_ZFlip_HalfOfWminusZ() => await RunTest(async accelerator =>
        {
            // Row-vector: v_out = v_in * ZFlip. Only Z changes: new_z = (w - z) / 2; new_w = w.
            // Then post-divide: z_ndc_new = (w-z)/(2w) = (1 - z/w) / 2. Maps GL z_ndc in
            // [-1,+1] to reversed-Z [1,0].
            var tests = new (Vector4 input, Vector4 expected)[]
            {
                // GL NDC sanity (w=1):
                (new Vector4(0, 0, -1, 1),      new Vector4(0, 0, 1f,    1)),   // GL near  -> rev 1
                (new Vector4(0, 0,  1, 1),      new Vector4(0, 0, 0f,    1)),   // GL far   -> rev 0
                (new Vector4(0, 0,  0, 1),      new Vector4(0, 0, 0.5f,  1)),   // GL mid   -> rev 0.5
                // Arbitrary (x, y preserved; new_z = (w-z)/2; new_w = w):
                (new Vector4(3, 4, 5, 2),       new Vector4(3, 4, -1.5f, 2)),   // (2-5)/2 = -1.5
                (new Vector4(1, -1, 2, 3),      new Vector4(1, -1, 0.5f, 3)),   // (3-2)/2 = 0.5
            };

            foreach (var (input, expected) in tests)
            {
                var got = Vector4.Transform(input, VrMatrixHelpers.ZFlipStandardToReversedZ);
                if (MathF.Abs(got.X - expected.X) > 1e-5f ||
                    MathF.Abs(got.Y - expected.Y) > 1e-5f ||
                    MathF.Abs(got.Z - expected.Z) > 1e-5f ||
                    MathF.Abs(got.W - expected.W) > 1e-5f)
                {
                    throw new Exception($"ZFlip mismatch: input={input} expected={expected} got={got}");
                }
            }
            await Task.CompletedTask;
        });

        // ------------------------------------------------------------------
        // BuildReversedZViewProj end-to-end
        // ------------------------------------------------------------------

        [TestMethod]
        public async Task VrMatrix_BuildReversedZViewProj_VertexAtMidZ() => await RunTest(async accelerator =>
        {
            // Standard-Z GL perspective: near=1, far=10, fov=90°, aspect=1.
            //   M11 = 1/tan(45°)/1 = 1
            //   M22 = 1/tan(45°)   = 1
            //   M33 = -(10+1)/(10-1) = -11/9 ~= -1.222
            //   M43 = -2*10*1/(10-1) = -20/9 ~= -2.222
            //   M34 = -1
            //   M44 = 0
            // Stored column-major for the GL API.
            var projCol = new float[16]
            {
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, -11f/9f, -1f,
                0f, 0f, -20f/9f, 0f,
            };
            // Identity view: camera at world origin looking down -Z.
            var viewCol = new float[16] { 1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1 };

            var vp = VrMatrixHelpers.BuildReversedZViewProj(projCol, viewCol);

            // World-space vertex 5 meters down -Z.
            var world = new Vector4(0f, 0f, -5f, 1f);

            // Row-vector math: clip = world * vp.
            var clipRev = Vector4.Transform(world, vp);

            // Perspective divide.
            float z_ndc_rev = clipRev.Z / clipRev.W;

            // Expected: standard GL z would give z_ndc_gl at this distance.
            // -z_view=5, standard GL z_ndc:
            //   clip_z_gl = -5 * -1.222 + 1 * -2.222 = 6.111 - 2.222 = 3.889
            //   clip_w    = -5 * -1       + 1 *  0      = 5
            //   z_ndc_gl  = 3.889 / 5 = 0.778 (in GL [-1,+1] convention, close to far)
            // Reversed: z_ndc_rev = (1 - 0.778) / 2 = 0.111 (closer to far than near).
            float expected = (1f - 0.778f) / 2f;   // 0.111
            if (MathF.Abs(z_ndc_rev - expected) > 5e-3f)
                throw new Exception($"z_ndc_rev at z=-5 expected ~{expected:F3}, got {z_ndc_rev:F5}");

            // Also verify X and Y are preserved (symmetric perspective keeps them zero
            // for an on-axis vertex).
            if (MathF.Abs(clipRev.X) > 1e-5f || MathF.Abs(clipRev.Y) > 1e-5f)
                throw new Exception($"On-axis vertex should have zero clip XY; got ({clipRev.X},{clipRev.Y})");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_BuildReversedZViewProj_NearPlaneClip1_FarPlaneClip0() => await RunTest(async accelerator =>
        {
            // Same standard-Z perspective. Reversed-Z should give:
            //   near (z_view=-1): z_ndc_rev = 1
            //   far  (z_view=-10): z_ndc_rev = 0
            var projCol = new float[16]
            {
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, -11f/9f, -1f,
                0f, 0f, -20f/9f, 0f,
            };
            var viewCol = new float[16] { 1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1 };
            var vp = VrMatrixHelpers.BuildReversedZViewProj(projCol, viewCol);

            // Near plane:
            var near = Vector4.Transform(new Vector4(0, 0, -1f, 1f), vp);
            float zNear = near.Z / near.W;
            if (MathF.Abs(zNear - 1f) > 1e-4f)
                throw new Exception($"Near plane should map to z_ndc_rev=1, got {zNear:F6}");

            // Far plane (z=-10):
            var far = Vector4.Transform(new Vector4(0, 0, -10f, 1f), vp);
            float zFar = far.Z / far.W;
            if (MathF.Abs(zFar - 0f) > 1e-4f)
                throw new Exception($"Far plane should map to z_ndc_rev=0, got {zFar:F6}");

            // Closer plane is "larger" depth in reversed-Z, as required by depthCompare=greater.
            if (zNear <= zFar)
                throw new Exception($"Reversed-Z should satisfy zNear > zFar; got zNear={zNear:F4} zFar={zFar:F4}");

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_BuildReversedZViewProj_AsymmetricFrustum_StillPreservesReversedZ() => await RunTest(async accelerator =>
        {
            // WebXR per-eye projections are ASYMMETRIC (shifted frustum for stereo).
            // This exercises the same code path with off-axis M13/M23 values to catch
            // any bug that only appears with non-zero frustum-center offsets.
            //
            // Standard GL off-axis perspective: left=-0.5, right=2.0, bottom=-1, top=1, near=1, far=10.
            //   M11 = 2n/(r-l) = 2/2.5 = 0.8
            //   M13 = (r+l)/(r-l) = 1.5/2.5 = 0.6
            //   M22 = 2n/(t-b) = 2/2 = 1.0
            //   M23 = (t+b)/(t-b) = 0
            //   M33 = -(f+n)/(f-n) = -11/9
            //   M43 = -2fn/(f-n) = -20/9
            //   M34 = -1
            //   M44 = 0
            // Column-major storage.
            var projCol = new float[16]
            {
                0.8f, 0f, 0f, 0f,        // col 0: M11, M21, M31, M41
                0f, 1.0f, 0f, 0f,        // col 1: M12, M22, M32, M42
                0.6f, 0f, -11f/9f, -1f,  // col 2: M13, M23, M33, M43
                0f, 0f, -20f/9f, 0f,     // col 3: M14, M24, M34, M44
            };
            var viewCol = new float[16] { 1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1 };
            var vp = VrMatrixHelpers.BuildReversedZViewProj(projCol, viewCol);

            // On-axis vertex still at -Z=-5. Should still map to z_ndc_rev ~ 0.111
            // (the near/far config didn't change, only the L/R frustum edges did).
            var clip = Vector4.Transform(new Vector4(0f, 0f, -5f, 1f), vp);
            float zRev = clip.Z / clip.W;
            if (MathF.Abs(zRev - ((1f - 0.778f) / 2f)) > 5e-3f)
                throw new Exception($"Asymmetric frustum still expects z_ndc_rev~=0.111 at z=-5; got {zRev:F4}");

            // Near + far plane boundaries still map to 1 and 0 respectively.
            var near = Vector4.Transform(new Vector4(0f, 0f, -1f, 1f), vp);
            var far  = Vector4.Transform(new Vector4(0f, 0f, -10f, 1f), vp);
            float zNear = near.Z / near.W;
            float zFar = far.Z / far.W;
            if (MathF.Abs(zNear - 1f) > 1e-4f)
                throw new Exception($"Asymmetric frustum near plane should map to 1, got {zNear:F6}");
            if (MathF.Abs(zFar - 0f) > 1e-4f)
                throw new Exception($"Asymmetric frustum far plane should map to 0, got {zFar:F6}");

            // Clip X must NOT be zero for an on-axis vertex (asymmetric frustum shifts it).
            // For this frustum (left=-0.5, right=2.0), the frustum-center in view-space is at
            // x_view = (right+left)/2 = 0.75, so a world vertex at x=0 is LEFT of center. With
            // M13=0.6 the clip.x contribution from z=-5 is 0.6*(-5) = -3. Verify.
            if (MathF.Abs(clip.X - (-3f)) > 1e-3f)
                throw new Exception($"Asymmetric frustum: vertex (0,0,-5,1) should have clip.x = -3, got {clip.X:F4}");

            await Task.CompletedTask;
        });

        // ------------------------------------------------------------------
        // ComputeLocomotionDelta (thumbstick -> world-axis delta integrated
        // through current yaw)
        // ------------------------------------------------------------------

        [TestMethod]
        public async Task VrMatrix_Locomotion_ZeroStick_ZeroDelta() => await RunTest(async accelerator =>
        {
            var d = VrMatrixHelpers.ComputeLocomotionDelta(0f, 0f, 0f, 3f, 1f);
            if (d != Vector3.Zero) throw new Exception($"Zero stick should give zero delta; got {d}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_Locomotion_ZeroDt_ZeroDelta() => await RunTest(async accelerator =>
        {
            var d = VrMatrixHelpers.ComputeLocomotionDelta(1f, -1f, 0f, 3f, 0f);
            if (d != Vector3.Zero) throw new Exception($"Zero dt should give zero delta; got {d}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_Locomotion_NoYaw_ForwardIsNegativeZ() => await RunTest(async accelerator =>
        {
            // Quest: stick pushed up => axes[3] = -1. We pass stickY = -1.
            // With yaw=0, "forward" should be world -Z.
            var d = VrMatrixHelpers.ComputeLocomotionDelta(0f, -1f, 0f, 3f, 1f);
            if (MathF.Abs(d.X) > 1e-5f || MathF.Abs(d.Y) > 1e-5f || MathF.Abs(d.Z - (-3f)) > 1e-5f)
                throw new Exception($"stick-up at yaw=0 should move (0,0,-3); got {d}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_Locomotion_NoYaw_RightIsPositiveX() => await RunTest(async accelerator =>
        {
            var d = VrMatrixHelpers.ComputeLocomotionDelta(1f, 0f, 0f, 3f, 1f);
            if (MathF.Abs(d.X - 3f) > 1e-5f || MathF.Abs(d.Y) > 1e-5f || MathF.Abs(d.Z) > 1e-5f)
                throw new Exception($"stick-right at yaw=0 should move (3,0,0); got {d}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_Locomotion_YawLeft90_ForwardRotatesToNegativeX() => await RunTest(async accelerator =>
        {
            // Player has turned LEFT 90 deg (yaw = +pi/2 in our convention). Their "forward"
            // direction in world coords should now be world -X. Stick forward moves them -X.
            float yaw = MathF.PI / 2f;
            var d = VrMatrixHelpers.ComputeLocomotionDelta(0f, -1f, yaw, 3f, 1f);
            if (MathF.Abs(d.X - (-3f)) > 1e-5f || MathF.Abs(d.Y) > 1e-5f || MathF.Abs(d.Z) > 1e-5f)
                throw new Exception($"stick-up at yaw=+90deg should move (-3,0,0); got {d}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_Locomotion_YawRight90_ForwardRotatesToPositiveX() => await RunTest(async accelerator =>
        {
            // Player has turned RIGHT 90 deg (yaw = -pi/2 in our convention). Forward = +X.
            float yaw = -MathF.PI / 2f;
            var d = VrMatrixHelpers.ComputeLocomotionDelta(0f, -1f, yaw, 3f, 1f);
            if (MathF.Abs(d.X - 3f) > 1e-5f || MathF.Abs(d.Y) > 1e-5f || MathF.Abs(d.Z) > 1e-5f)
                throw new Exception($"stick-up at yaw=-90deg should move (+3,0,0); got {d}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_Locomotion_SpeedAndDtScale() => await RunTest(async accelerator =>
        {
            // dt and speed both linearly scale the output.
            var a = VrMatrixHelpers.ComputeLocomotionDelta(0f, -1f, 0f, 3f, 1f);      // 3 m
            var b = VrMatrixHelpers.ComputeLocomotionDelta(0f, -1f, 0f, 3f, 0.5f);    // 1.5 m
            var c = VrMatrixHelpers.ComputeLocomotionDelta(0f, -1f, 0f, 6f, 0.5f);    // 3 m
            if (MathF.Abs(a.Z - (-3f)) > 1e-5f) throw new Exception($"a.z={a.Z} expected -3");
            if (MathF.Abs(b.Z - (-1.5f)) > 1e-5f) throw new Exception($"b.z={b.Z} expected -1.5");
            if (MathF.Abs(c.Z - (-3f)) > 1e-5f) throw new Exception($"c.z={c.Z} expected -3");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_Locomotion_DiagonalStick_NormalizedSumForNoYaw() => await RunTest(async accelerator =>
        {
            // stickX=1 and stickY=-1 (full diagonal: up-right) at yaw=0 should produce
            // +X and -Z equally. This test doesn't enforce normalization (engine doesn't
            // either - sqrt(2) m/s at full diagonal is intentional for feel), just that
            // X and Z components come out with the right magnitudes.
            var d = VrMatrixHelpers.ComputeLocomotionDelta(1f, -1f, 0f, 3f, 1f);
            if (MathF.Abs(d.X - 3f) > 1e-5f) throw new Exception($"diag +X should be 3; got {d.X}");
            if (MathF.Abs(d.Z - (-3f)) > 1e-5f) throw new Exception($"diag -Z should be -3; got {d.Z}");
            if (MathF.Abs(d.Y) > 1e-5f) throw new Exception($"no vertical component; got y={d.Y}");
            await Task.CompletedTask;
        });

        // ------------------------------------------------------------------
        // TryApplySnapTurn (snap-turn debounce state machine)
        // ------------------------------------------------------------------

        [TestMethod]
        public async Task VrMatrix_SnapTurn_CenteredStick_NoChange() => await RunTest(async accelerator =>
        {
            bool armed = true;
            float yaw = 0.5f;
            bool fired = VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0f);
            if (fired) throw new Exception("Centered stick must not fire a snap.");
            if (yaw != 0.5f) throw new Exception($"Yaw must not change; got {yaw}");
            if (!armed) throw new Exception("Armed should remain true after centered stick.");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_SnapTurn_StickRight_FiresClockwise() => await RunTest(async accelerator =>
        {
            // Convention: stick right => yaw decreases (player turns right / world rotates CCW about them).
            bool armed = true;
            float yaw = 0f;
            bool fired = VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.9f);
            if (!fired) throw new Exception("Stick past +0.7 should fire.");
            if (MathF.Abs(yaw - (-MathF.PI / 6f)) > 1e-6f) throw new Exception($"Yaw should be -30deg; got {yaw * 180f / MathF.PI:F2}deg");
            if (armed) throw new Exception("Armed should drop to false after snap.");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_SnapTurn_StickLeft_FiresCounterClockwise() => await RunTest(async accelerator =>
        {
            bool armed = true;
            float yaw = 0f;
            bool fired = VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, -0.9f);
            if (!fired) throw new Exception("Stick past -0.7 should fire.");
            if (MathF.Abs(yaw - (MathF.PI / 6f)) > 1e-6f) throw new Exception($"Yaw should be +30deg; got {yaw * 180f / MathF.PI:F2}deg");
            if (armed) throw new Exception("Armed should drop to false after snap.");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_SnapTurn_StickHeldPastThresh_FiresOnceNotRepeatedly() => await RunTest(async accelerator =>
        {
            bool armed = true;
            float yaw = 0f;
            bool first = VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.9f);
            bool second = VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.9f);
            bool third  = VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.9f);
            if (!first) throw new Exception("First call should fire.");
            if (second || third) throw new Exception("Subsequent calls while stick held should NOT fire.");
            if (MathF.Abs(yaw - (-MathF.PI / 6f)) > 1e-6f) throw new Exception($"Only one snap accumulated; yaw={yaw * 180f / MathF.PI:F2}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_SnapTurn_ReleaseThenRepress_FiresAgain() => await RunTest(async accelerator =>
        {
            bool armed = true;
            float yaw = 0f;
            VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.9f);    // first snap
            // Stick between release and snap thresh: doesn't re-arm yet.
            VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.5f);    // still disarmed
            if (armed) throw new Exception("At 0.5 (above release) must stay disarmed.");
            // Stick below release thresh: re-arms.
            VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.1f);    // armed
            if (!armed) throw new Exception("Below release thresh must re-arm.");
            // Press again fires second snap.
            bool fired = VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.9f);
            if (!fired) throw new Exception("After re-arm, press fires again.");
            if (MathF.Abs(yaw - (-2f * MathF.PI / 6f)) > 1e-6f) throw new Exception($"Two right-snaps should be -60deg; yaw={yaw * 180f / MathF.PI:F2}");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_SnapTurn_AlternatingSnaps_CancelOut() => await RunTest(async accelerator =>
        {
            bool armed = true;
            float yaw = 0f;
            // Right snap, release, left snap.
            VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0.9f);   // -30
            VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, 0f);     // rearm
            VrMatrixHelpers.TryApplySnapTurn(ref armed, ref yaw, -0.9f);  // +30 -> net 0
            if (MathF.Abs(yaw) > 1e-6f) throw new Exception($"Right-then-left snaps should cancel; yaw={yaw * 180f / MathF.PI:F2}");
            await Task.CompletedTask;
        });

        // ------------------------------------------------------------------
        // RayFloorHit (ray-plane intersection for controller floor marker)
        // ------------------------------------------------------------------

        /// <summary>
        /// Helper: builds a 16-float GL-column-major matrix for a ray that starts at
        /// `origin` and points in `forward` (forward must be normalized; function stores
        /// -forward at column 2 rows 0..2 since ray direction in the convention is -Z).
        /// </summary>
        private static float[] MakeRayMatrix(Vector3 origin, Vector3 forward)
        {
            // Column 2 (m[8..11]) holds the ray's local +Z axis; convention: ray direction = -col2.
            // For ray-direction d, set col2 = -d.
            var m = new float[16];
            // Column 0 and 1 unused for the hit math; leave as zero.
            m[8] = -forward.X; m[9] = -forward.Y; m[10] = -forward.Z;
            m[12] = origin.X; m[13] = origin.Y; m[14] = origin.Z; m[15] = 1f;
            return m;
        }

        [TestMethod]
        public async Task VrMatrix_RayFloorHit_StraightDownFromAbove_HitsAtXZ() => await RunTest(async accelerator =>
        {
            var m = MakeRayMatrix(new Vector3(1f, 2f, -3f), new Vector3(0f, -1f, 0f));
            bool hit = VrMatrixHelpers.RayFloorHit(m, 5f, out float hx, out float hz);
            if (!hit) throw new Exception("Straight-down ray from y=2 should hit the floor within bounds.");
            if (MathF.Abs(hx - 1f) > 1e-5f || MathF.Abs(hz - (-3f)) > 1e-5f)
                throw new Exception($"Hit point should be (1, -3); got ({hx}, {hz})");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_RayFloorHit_AngledDownAndForward() => await RunTest(async accelerator =>
        {
            // Origin (0, 1, 0), 45deg down-and-forward. Unit direction d = (0, -1/sqrt2, -1/sqrt2).
            // y=0 solved by 1 + t*(-1/sqrt2) = 0 -> t = sqrt2.
            // Horizontal distance = t * |horizontal component| = sqrt2 * (1/sqrt2) = 1.
            // Hit: (0, 0, -1). (Intuitive: equal horizontal and vertical distances.)
            var invRoot2 = 1f / MathF.Sqrt(2f);
            var m = MakeRayMatrix(new Vector3(0f, 1f, 0f), new Vector3(0f, -invRoot2, -invRoot2));
            bool hit = VrMatrixHelpers.RayFloorHit(m, 5f, out float hx, out float hz);
            if (!hit) throw new Exception("Angled-down ray should hit within bounds.");
            if (MathF.Abs(hx) > 1e-4f || MathF.Abs(hz - (-1f)) > 1e-4f)
                throw new Exception($"Hit should be (0, -1); got ({hx}, {hz})");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_RayFloorHit_PointingUp_Misses() => await RunTest(async accelerator =>
        {
            var m = MakeRayMatrix(new Vector3(0f, 1f, 0f), new Vector3(0f, 1f, 0f));  // pointing up
            bool hit = VrMatrixHelpers.RayFloorHit(m, 5f, out _, out _);
            if (hit) throw new Exception("Upward ray must miss the floor.");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_RayFloorHit_Horizontal_Misses() => await RunTest(async accelerator =>
        {
            var m = MakeRayMatrix(new Vector3(0f, 1f, 0f), new Vector3(0f, 0f, -1f));  // horizontal
            bool hit = VrMatrixHelpers.RayFloorHit(m, 5f, out _, out _);
            if (hit) throw new Exception("Horizontal ray must miss (never descends).");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_RayFloorHit_OriginBelowFloor_Misses() => await RunTest(async accelerator =>
        {
            // Origin below floor (y<0), pointing down: t would be negative.
            var m = MakeRayMatrix(new Vector3(0f, -1f, 0f), new Vector3(0f, -1f, 0f));
            bool hit = VrMatrixHelpers.RayFloorHit(m, 5f, out _, out _);
            if (hit) throw new Exception("Origin below floor, pointing down: no valid hit in front.");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_RayFloorHit_HitOutsideFloorBounds_Misses() => await RunTest(async accelerator =>
        {
            // 45-deg down-forward from (0, 1, 0) lands at (0, 0, -1) [see AngledDownAndForward].
            // If we shrink floorHalf to 0.5, this is outside and should miss.
            var invRoot2 = 1f / MathF.Sqrt(2f);
            var m = MakeRayMatrix(new Vector3(0f, 1f, 0f), new Vector3(0f, -invRoot2, -invRoot2));
            bool hit = VrMatrixHelpers.RayFloorHit(m, 0.5f, out _, out _);
            if (hit) throw new Exception("Hit at z=-1 outside floorHalf=0.5 must return false.");
            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_RayFloorHit_NullMatrix_Misses() => await RunTest(async accelerator =>
        {
            bool hit = VrMatrixHelpers.RayFloorHit(null, 5f, out float hx, out float hz);
            if (hit) throw new Exception("Null matrix must return false.");
            if (hx != 0 || hz != 0) throw new Exception("Null matrix must leave outputs at zero.");
            await Task.CompletedTask;
        });

        // ------------------------------------------------------------------
        // SpawnDev.BlazorJS XRRigidTransform marshaling check
        //
        // This directly verifies my Phase C 2c concern: `new XRRigidTransform(new { x, y, z, w }, null)`
        // - does the anonymous .NET object marshal through Microsoft.JSInterop's System.Text.Json
        // serialization into a JS `{x, y, z, w}` object that the XRRigidTransform constructor
        // accepts. Browser-only (WebXR API is a browser-global). First consumer test -
        // if this fails, locomotion would silently produce identity transforms and the fix
        // is at the SpawnDev.BlazorJS layer.
        // ------------------------------------------------------------------

        [TestMethod]
        public async Task VrMatrix_BlazorJS_XRRigidTransform_AnonymousObjectMarshals() => await RunTest(async accelerator =>
        {
            // WebXR API is a browser global. Browser test hosts (WebGPU/WebGL/Wasm via
            // Blazor WASM) have a live BlazorJSRuntime; desktop hosts (CPU/CUDA/OpenCL via
            // DemoConsole) don't initialize it. Defensive access + skip on anything that
            // isn't a usable browser window.
            BlazorJSRuntime? js = null;
            try { js = BlazorJSRuntime.JS; } catch { }
            if (js == null || !js.IsWindow)
                throw new UnsupportedTestException("BlazorJS runtime not in a browser window context.");
            if (js.TypeOf("XRRigidTransform") == "undefined")
                throw new UnsupportedTestException("WebXR not available in this browser.");

            // The exact shape my ApplyLocomotionFromSnaps uses in VrPrototype.razor.
            using var xform = new XRRigidTransform(
                new { x = 1.25, y = 2.5, z = -3.75, w = 1.0 },
                new { x = 0.0, y = 0.0, z = 0.0, w = 1.0 });

            using var pos = xform.Position;
            double px = pos.X, py = pos.Y, pz = pos.Z;

            if (MathF.Abs((float)px - 1.25f) > 1e-5f ||
                MathF.Abs((float)py - 2.5f) > 1e-5f ||
                MathF.Abs((float)pz - (-3.75f)) > 1e-5f)
            {
                throw new Exception(
                    $"XRRigidTransform anonymous-object position did not round-trip. " +
                    $"Expected (1.25, 2.5, -3.75); got ({px}, {py}, {pz}). " +
                    $"If near-zero: JSInterop is dropping anonymous-object properties; " +
                    $"fix at the SpawnDev.BlazorJS wrapper layer (add a typed overload " +
                    $"taking a DOMPointInit or concrete position record).");
            }

            await Task.CompletedTask;
        });

        [TestMethod]
        public async Task VrMatrix_BuildReversedZViewProj_ViewTranslation_ShiftsWorldPosition() => await RunTest(async accelerator =>
        {
            // View = inverse of "camera at world (0, 0, 5)" = translation (0, 0, -5).
            // In GL column-major: V = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,-5,1]] (column 3 has the translation).
            // viewCol stored column-by-column:
            //   col 0: (1, 0, 0, 0)
            //   col 1: (0, 1, 0, 0)
            //   col 2: (0, 0, 1, 0)
            //   col 3: (0, 0, -5, 1)
            var projCol = new float[16]
            {
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, -11f/9f, -1f,
                0f, 0f, -20f/9f, 0f,
            };
            var viewCol = new float[16]
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, -5, 1,
            };
            var vp = VrMatrixHelpers.BuildReversedZViewProj(projCol, viewCol);

            // World vertex at z = 0 (origin) should now be "5m in front of camera" -> z_view = -5.
            // Same as the MidZ test case above: z_ndc_rev ~= 0.111.
            var world = new Vector4(0f, 0f, 0f, 1f);
            var clipRev = Vector4.Transform(world, vp);
            float zRev = clipRev.Z / clipRev.W;
            if (MathF.Abs(zRev - ((1f - 0.778f) / 2f)) > 5e-3f)
                throw new Exception($"With camera at world (0,0,5) looking -Z, origin should z_ndc_rev~=0.111; got {zRev:F4}");

            await Task.CompletedTask;
        });
    }
}
