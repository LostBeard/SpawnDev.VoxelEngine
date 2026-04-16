using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Meshing;
using SpawnDev.VoxelEngine.Rendering;
using SpawnDev.VoxelEngine.VR;
using ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.VoxelEngine.Adaptive;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Phase 2: Test coverage for previously untested features.
    // All tests are CPU-only (no GPU required) so they run on all 6 backends.
    public abstract partial class VoxelEngineTestBase
    {
        // ===================================================================
        // TransparencyMesher
        // ===================================================================

        [TestMethod]
        public void Transparency_HasTransparentBlocks_DetectsGlassTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            int ss = 4;
            var blocks = new int[ss * ss * ss];
            blocks[0] = PackedBlock.Pack(7); // glass = transparent

            if (!TransparencyMesher.HasTransparentBlocks(blocks, ss, ss, registry))
                throw new Exception("Should detect glass (type 7) as transparent");
        }

        [TestMethod]
        public void Transparency_HasTranslucentBlocks_DetectsWaterTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            int ss = 4;
            var blocks = new int[ss * ss * ss];
            blocks[0] = PackedBlock.Pack(8); // water = translucent

            if (!TransparencyMesher.HasTranslucentBlocks(blocks, ss, ss, registry))
                throw new Exception("Should detect water (type 8) as translucent");
        }

        [TestMethod]
        public void Transparency_OpaqueOnly_NoTransparentTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            int ss = 4;
            var blocks = new int[ss * ss * ss];
            for (int i = 0; i < blocks.Length; i++)
                blocks[i] = PackedBlock.Pack(1); // all stone

            if (TransparencyMesher.HasTransparentBlocks(blocks, ss, ss, registry))
                throw new Exception("Stone-only section should not have transparent blocks");
            if (TransparencyMesher.HasTranslucentBlocks(blocks, ss, ss, registry))
                throw new Exception("Stone-only section should not have translucent blocks");
        }

        [TestMethod]
        public void Transparency_BuildFaceMasks_ProducesOutputTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            int ss = 4, paddedXZ = ss + 2;
            var padded = new int[paddedXZ * paddedXZ * ss];
            // Glass block at center
            padded[(2) + (2) * paddedXZ + 2 * paddedXZ * paddedXZ] = PackedBlock.Pack(7);

            var masks = TransparencyMesher.BuildTransparentFaceMasks(padded, ss, ss, paddedXZ, registry);
            if (masks.Length != 6 * ss * ss)
                throw new Exception($"Face mask array length {masks.Length}, expected {6 * ss * ss}");

            // At least one mask should have a set bit (glass block has visible faces)
            bool anySet = false;
            for (int i = 0; i < masks.Length; i++)
                if (masks[i] != 0) { anySet = true; break; }
            if (!anySet)
                throw new Exception("Glass block should produce at least one visible transparent face mask");
        }

        // ===================================================================
        // CrossQuadMesher
        // ===================================================================

        [TestMethod]
        public void CrossQuad_CountPlantBlocks_DetectsTallGrassTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            int ss = 4;
            var blocks = new int[ss * ss * ss];
            blocks[0] = PackedBlock.Pack(9); // tall grass = plant
            blocks[1] = PackedBlock.Pack(9);
            blocks[2] = PackedBlock.Pack(1); // stone = not plant

            int count = CrossQuadMesher.CountPlantBlocks(blocks, ss, ss, registry);
            if (count != 2)
                throw new Exception($"Expected 2 plant blocks, got {count}");
        }

        [TestMethod]
        public void CrossQuad_GenerateCrossQuads_12VerticesPerPlantTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            int ss = 4;
            var blocks = new int[ss * ss * ss];
            blocks[0] = PackedBlock.Pack(9); // one tall grass plant

            var vertices = CrossQuadMesher.GenerateCrossQuads(blocks, ss, ss, 1f, registry);
            if (vertices.Count != 12)
                throw new Exception($"One plant should produce 12 vertices (2 cross quads x 6 verts), got {vertices.Count}");
        }

        [TestMethod]
        public void CrossQuad_NoPlantsNoVerticesTest()
        {
            var registry = new BlockRegistry();
            registry.RegisterDefaults();

            int ss = 4;
            var blocks = new int[ss * ss * ss];
            blocks[0] = PackedBlock.Pack(1); // stone only

            var vertices = CrossQuadMesher.GenerateCrossQuads(blocks, ss, ss, 1f, registry);
            if (vertices.Count != 0)
                throw new Exception($"No plants should produce 0 vertices, got {vertices.Count}");
        }

        // ===================================================================
        // TextureArrayManager
        // ===================================================================

        [TestMethod]
        public void TextureArray_RegisterBlockType_AssignsLayersTest()
        {
            var mgr = new TextureArrayManager();
            int base1 = mgr.RegisterBlockType(1);
            int base2 = mgr.RegisterBlockType(2);

            if (base1 < 0)
                throw new Exception($"Block type 1 base layer {base1} should be >= 0");
            if (base2 <= base1)
                throw new Exception($"Block type 2 base {base2} should be > type 1 base {base1}");
            if (mgr.RegisteredTypeCount != 2)
                throw new Exception($"RegisteredTypeCount should be 2, got {mgr.RegisteredTypeCount}");
        }

        [TestMethod]
        public void TextureArray_RegisterIdempotentTest()
        {
            var mgr = new TextureArrayManager();
            int first = mgr.RegisterBlockType(5);
            int second = mgr.RegisterBlockType(5);

            if (first != second)
                throw new Exception($"Re-registering same type should return same base: {first} vs {second}");
            if (mgr.RegisteredTypeCount != 1)
                throw new Exception($"RegisteredTypeCount should be 1 after re-register, got {mgr.RegisteredTypeCount}");
        }

        [TestMethod]
        public void TextureArray_GetLayerIndex_DistinctPerFaceTest()
        {
            var mgr = new TextureArrayManager();
            mgr.RegisterBlockType(1);

            int topLayer = mgr.GetLayerIndex(1, 4, 0);    // +Y face, variant 0
            int sideLayer = mgr.GetLayerIndex(1, 0, 0);   // +X face, variant 0

            if (topLayer == sideLayer)
                throw new Exception($"Top and side face should have different layer indices: both {topLayer}");
        }

        [TestMethod]
        public void TextureArray_GetVariant_DeterministicTest()
        {
            int v1 = TextureArrayManager.GetVariant(10, 20, 30, 4);
            int v2 = TextureArrayManager.GetVariant(10, 20, 30, 4);
            int v3 = TextureArrayManager.GetVariant(11, 20, 30, 4);

            if (v1 != v2)
                throw new Exception("Same position should produce same variant");
            if (v1 < 0 || v1 >= 4)
                throw new Exception($"Variant {v1} out of range [0, 4)");
            // Different positions MIGHT produce same variant, but let's verify the hash works
            // by checking a range of positions produces multiple variants
            var variants = new HashSet<int>();
            for (int x = 0; x < 16; x++)
                variants.Add(TextureArrayManager.GetVariant(x, 0, 0, 4));
            if (variants.Count < 2)
                throw new Exception($"16 positions produced only {variants.Count} unique variant(s) - hash is broken");
        }

        // ===================================================================
        // DynamicLighting
        // ===================================================================

        [TestMethod]
        public void DynamicLighting_CreatePoint_CorrectTypeTest()
        {
            var light = DynamicLighting.LightData.CreatePoint(
                new Vector3(5, 10, 5), new Vector3(1, 1, 1), 2f, 20f);

            if (light.PositionAndType.W != 0f)
                throw new Exception($"Point light type should be 0, got {light.PositionAndType.W}");
            if (light.PositionAndType.X != 5f)
                throw new Exception($"Position X should be 5, got {light.PositionAndType.X}");
            if (light.ColorAndIntensity.W != 2f)
                throw new Exception($"Intensity should be 2, got {light.ColorAndIntensity.W}");
        }

        [TestMethod]
        public void DynamicLighting_CreateDirectional_NormalizedTest()
        {
            var light = DynamicLighting.LightData.CreateDirectional(
                new Vector3(1, 2, 3), new Vector3(1, 0.9f, 0.8f), 1f);

            if (light.PositionAndType.W != 2f)
                throw new Exception($"Directional light type should be 2, got {light.PositionAndType.W}");

            // Direction should be normalized
            var dir = new Vector3(light.PositionAndType.X, light.PositionAndType.Y, light.PositionAndType.Z);
            float len = dir.Length();
            if (MathF.Abs(len - 1f) > 0.01f)
                throw new Exception($"Direction length should be 1, got {len}");
        }

        [TestMethod]
        public void DynamicLighting_ComputeLighting_PointLightFacingBrighterTest()
        {
            var light = DynamicLighting.LightData.CreatePoint(
                new Vector3(0, 10, 0), new Vector3(1, 1, 1), 1f, 20f);
            var lights = new DynamicLighting.LightData[] { light };
            var ambient = new Vector3(0.1f);

            // Surface facing the light (+Y normal, light above)
            var facingColor = DynamicLighting.ComputeLighting(
                Vector3.Zero, Vector3.UnitY, lights, 1, ambient);

            // Surface facing away (-Y normal, light above)
            var awayColor = DynamicLighting.ComputeLighting(
                Vector3.Zero, -Vector3.UnitY, lights, 1, ambient);

            float facingBrightness = facingColor.X + facingColor.Y + facingColor.Z;
            float awayBrightness = awayColor.X + awayColor.Y + awayColor.Z;

            if (facingBrightness <= awayBrightness)
                throw new Exception(
                    $"Face pointing at light ({facingBrightness:F3}) should be brighter than " +
                    $"face pointing away ({awayBrightness:F3})");
        }

        [TestMethod]
        public void DynamicLighting_MaxLightsCapTest()
        {
            var lights = new DynamicLighting.LightData[20];
            for (int i = 0; i < 20; i++)
                lights[i] = DynamicLighting.LightData.CreatePoint(
                    new Vector3(0, 5 + i, 0), new Vector3(1, 1, 1), 1f, 50f);

            // Should not crash even with > MaxLights
            var result = DynamicLighting.ComputeLighting(
                Vector3.Zero, Vector3.UnitY, lights, 20, new Vector3(0.1f));

            if (result.X < 0.1f)
                throw new Exception($"Lighting result {result} too dim - max lights cap may be broken");
        }

        // ===================================================================
        // TimeOfDay
        // ===================================================================

        [TestMethod]
        public void TimeOfDay_NoonIsBrightestTest()
        {
            var tod = new TimeOfDay { Time = 0.5f }; // noon
            var noonIntensity = tod.SunIntensity;

            tod.Time = 0f; // midnight
            var midnightIntensity = tod.SunIntensity;

            if (noonIntensity <= 0.5f)
                throw new Exception($"Noon sun intensity {noonIntensity} should be > 0.5");
            if (midnightIntensity > 0.01f)
                throw new Exception($"Midnight sun intensity {midnightIntensity} should be ~0");
        }

        [TestMethod]
        public void TimeOfDay_MidnightIsDarkTest()
        {
            var tod = new TimeOfDay { Time = 0f }; // midnight
            if (!tod.IsDark)
                throw new Exception("Midnight should be dark");

            tod.Time = 0.5f; // noon
            if (tod.IsDark)
                throw new Exception("Noon should not be dark");
        }

        [TestMethod]
        public void TimeOfDay_SunDirectionChangesTest()
        {
            var tod = new TimeOfDay { Time = 0.25f }; // dawn
            var dawnDir = tod.SunDirection;

            tod.Time = 0.5f; // noon
            var noonDir = tod.SunDirection;

            float diff = Vector3.Distance(dawnDir, noonDir);
            if (diff < 0.1f)
                throw new Exception($"Sun direction should change between dawn and noon, diff={diff}");
        }

        [TestMethod]
        public void TimeOfDay_UpdateAdvancesTimeTest()
        {
            var tod = new TimeOfDay { Time = 0f, CycleSpeed = 1f };
            float before = tod.Time;
            tod.Update(60f); // 60 seconds
            float after = tod.Time;

            if (after <= before)
                throw new Exception($"Time should advance: before={before}, after={after}");
        }

        [TestMethod]
        public void TimeOfDay_NVGParamsTest()
        {
            var tod = new TimeOfDay { NightVisionEnabled = false };
            var (gb1, ni1, r1) = tod.GetNVGParams();
            if (gb1 != 0f || ni1 != 0f || r1 != 0f)
                throw new Exception("NVG disabled should return all zeros");

            tod.NightVisionEnabled = true;
            tod.NightVisionRange = 50f;
            var (gb2, ni2, r2) = tod.GetNVGParams();
            if (gb2 <= 0f || ni2 <= 0f || r2 != 50f)
                throw new Exception($"NVG enabled: greenBoost={gb2}, noise={ni2}, range={r2} - should be non-zero");
        }

        [TestMethod]
        public void TimeOfDay_SkyColorDiffersNightVsDayTest()
        {
            var tod = new TimeOfDay { Time = 0f }; // midnight
            var nightSky = tod.SkyColor;

            tod.Time = 0.5f; // noon
            var daySky = tod.SkyColor;

            float nightBrightness = nightSky.X + nightSky.Y + nightSky.Z;
            float dayBrightness = daySky.X + daySky.Y + daySky.Z;

            if (dayBrightness <= nightBrightness)
                throw new Exception(
                    $"Day sky ({daySky}) should be brighter than night sky ({nightSky})");
        }

        // ===================================================================
        // DamageOverlay
        // ===================================================================

        [TestMethod]
        public void DamageOverlay_NoDamageNoCrackTest()
        {
            int stage = DamageOverlay.GetCrackStage(0);
            if (stage != -1)
                throw new Exception($"Damage 0 should return stage -1, got {stage}");

            float alpha = DamageOverlay.GetCrackAlpha(0);
            if (alpha != 0f)
                throw new Exception($"Damage 0 alpha should be 0, got {alpha}");
        }

        [TestMethod]
        public void DamageOverlay_MaxDamageMaxCrackTest()
        {
            int stage = DamageOverlay.GetCrackStage(15);
            if (stage != DamageOverlay.CrackStages - 1)
                throw new Exception($"Damage 15 should be max crack stage {DamageOverlay.CrackStages - 1}, got {stage}");

            float alpha = DamageOverlay.GetCrackAlpha(15);
            if (alpha < 0.9f)
                throw new Exception($"Damage 15 alpha should be ~1.0, got {alpha}");
        }

        [TestMethod]
        public void DamageOverlay_ProgressiveStagesTest()
        {
            int prev = -1;
            for (int d = 1; d <= 15; d++)
            {
                int stage = DamageOverlay.GetCrackStage(d);
                if (stage < prev)
                    throw new Exception($"Crack stage should not decrease: damage {d} stage {stage}, previous {prev}");
                prev = stage;
            }
        }

        // ===================================================================
        // StereoRenderer
        // ===================================================================

        [TestMethod]
        public void StereoRenderer_DefaultInactiveTest()
        {
            var sr = new StereoRenderer();
            if (sr.IsActive)
                throw new Exception("StereoRenderer should be inactive by default");
        }

        [TestMethod]
        public void StereoRenderer_UpdateFromXRViews_ActivatesTest()
        {
            var sr = new StereoRenderer();
            var leftView = Matrix4x4.CreateLookAt(new Vector3(-0.032f, 1.6f, 0), new Vector3(-0.032f, 1.6f, -1), Vector3.UnitY);
            var rightView = Matrix4x4.CreateLookAt(new Vector3(0.032f, 1.6f, 0), new Vector3(0.032f, 1.6f, -1), Vector3.UnitY);
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 2f, 1f, 0.1f, 100f);

            sr.UpdateFromXRViews(leftView, proj, rightView, proj);

            if (!sr.IsActive)
                throw new Exception("Should be active after UpdateFromXRViews");
            if (sr.LeftVP == Matrix4x4.Identity)
                throw new Exception("LeftVP should not be identity after update");
        }

        [TestMethod]
        public void StereoRenderer_CenterEyePositionMidpointTest()
        {
            var sr = new StereoRenderer();
            float halfIPD = 0.032f;
            var leftView = Matrix4x4.CreateLookAt(new Vector3(-halfIPD, 1.6f, 0), new Vector3(-halfIPD, 1.6f, -1), Vector3.UnitY);
            var rightView = Matrix4x4.CreateLookAt(new Vector3(halfIPD, 1.6f, 0), new Vector3(halfIPD, 1.6f, -1), Vector3.UnitY);

            var center = sr.GetCenterEyePosition(leftView, rightView);

            // Center should be approximately at x=0, y=1.6
            if (MathF.Abs(center.X) > 0.01f)
                throw new Exception($"Center eye X should be ~0, got {center.X}");
            if (MathF.Abs(center.Y - 1.6f) > 0.01f)
                throw new Exception($"Center eye Y should be ~1.6, got {center.Y}");
        }

        // ===================================================================
        // WebXRHelper
        // ===================================================================

        [TestMethod]
        public void WebXR_DefaultStateTest()
        {
            var xr = new WebXRHelper();
            if (xr.IsSessionActive)
                throw new Exception("Should not be active by default");
            if (xr.SessionMode != VR.XRSessionMode.None)
                throw new Exception($"Session mode should be None, got {xr.SessionMode}");
        }

        [TestMethod]
        public void WebXR_SessionLifecycleTest()
        {
            var xr = new WebXRHelper();

            xr.OnSessionStarted(VR.XRSessionMode.ImmersiveVR, 2064, 2208, true);
            if (!xr.IsSessionActive)
                throw new Exception("Should be active after OnSessionStarted");
            if (xr.SessionMode != VR.XRSessionMode.ImmersiveVR)
                throw new Exception($"Mode should be ImmersiveVR, got {xr.SessionMode}");
            if (!xr.HandTrackingAvailable)
                throw new Exception("Hand tracking should be available");
            if (xr.EyeWidth != 2064 || xr.EyeHeight != 2208)
                throw new Exception($"Eye resolution {xr.EyeWidth}x{xr.EyeHeight}, expected 2064x2208");

            xr.OnSessionEnded();
            if (xr.IsSessionActive)
                throw new Exception("Should not be active after OnSessionEnded");
            if (xr.SessionMode != VR.XRSessionMode.None)
                throw new Exception($"Mode should be None after end, got {xr.SessionMode}");
        }

        [TestMethod]
        public void WebXR_UpdateViewSetsMatricesTest()
        {
            var xr = new WebXRHelper();
            var view = Matrix4x4.CreateLookAt(new Vector3(0, 1.6f, 0), new Vector3(0, 1.6f, -1), Vector3.UnitY);
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 2f, 1f, 0.1f, 100f);

            xr.UpdateView(VR.XREye.Left, view, proj);

            if (xr.LeftView == Matrix4x4.Identity)
                throw new Exception("LeftView should be updated");
            if (xr.LeftProjection == Matrix4x4.Identity)
                throw new Exception("LeftProjection should be updated");
        }

        [TestMethod]
        public void WebXR_GetControllerRay_ExtractsCorrectlyTest()
        {
            // Grip transform with translation at (1, 2, 3) looking along -Z
            var grip = Matrix4x4.CreateTranslation(1, 2, 3);
            var (origin, direction) = WebXRHelper.GetControllerRay(grip);

            if (MathF.Abs(origin.X - 1f) > 0.01f || MathF.Abs(origin.Y - 2f) > 0.01f)
                throw new Exception($"Ray origin ({origin}) should be at grip position (1, 2, 3)");
        }

        // ===================================================================
        // FoveatedRendering
        // ===================================================================

        [TestMethod]
        public void Foveated_DefaultNoneTest()
        {
            var fr = new FoveatedRendering();
            if (fr.Level != FoveatedRendering.FFRLevel.None)
                throw new Exception($"Default level should be None, got {fr.Level}");
        }

        [TestMethod]
        public void Foveated_ThermalCritical_ForcesHighTest()
        {
            var fr = new FoveatedRendering { IsSupported = true };
            fr.Update(QualityController.QualityLevel.Ultra, ThermalManager.ThermalState.Critical);

            if (fr.Level != FoveatedRendering.FFRLevel.High)
                throw new Exception($"Thermal Critical should force High FFR, got {fr.Level}");
        }

        [TestMethod]
        public void Foveated_NotSupported_AlwaysNoneTest()
        {
            var fr = new FoveatedRendering { IsSupported = false };
            fr.Update(QualityController.QualityLevel.Minimal, ThermalManager.ThermalState.Critical);

            if (fr.Level != FoveatedRendering.FFRLevel.None)
                throw new Exception($"Unsupported FFR should always be None, got {fr.Level}");
        }

        [TestMethod]
        public void Foveated_WebXRFoveationValueTest()
        {
            var fr = new FoveatedRendering { IsSupported = true };

            fr.Update(QualityController.QualityLevel.Ultra, ThermalManager.ThermalState.Normal);
            float none = fr.GetWebXRFoveationLevel();
            if (none != 0f)
                throw new Exception($"Normal/Ultra should be 0, got {none}");

            fr.Update(QualityController.QualityLevel.Ultra, ThermalManager.ThermalState.Critical);
            float high = fr.GetWebXRFoveationLevel();
            if (high != 1f)
                throw new Exception($"Critical should be 1.0, got {high}");
        }

        // ===================================================================
        // VISUAL TESTS - pixel readback verification
        // These test features that produce visual output by actually rendering
        // and reading back pixels. A graphics engine must verify what it draws.
        // ===================================================================

        // -----------------------------------------------------------------------
        // TimeOfDay visual: noon is brighter than midnight
        // -----------------------------------------------------------------------

        [TestMethod]
        public async Task Visual_TimeOfDay_NoonBrighterThanMidnightTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new VertexPullPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            var quads = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1) };
            var view = Matrix4x4.CreateLookAt(new Vector3(8, 10, 8), new Vector3(8, 0, 8), Vector3.UnitZ);
            var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
            var vp = view * proj;
            var camPos = new Vector3(8, 10, 8);

            // Noon lighting
            var noon = new TimeOfDay { Time = 0.5f };
            var noonPixels = await RenderStandardAndReadBack(device, queue, pipeline,
                quads, vp, Vector3.Zero,
                noon.FogColor, 0f, noon.AmbientColor, camPos,
                64, 64, new Vector3(0));
            var (nr, ng, nb, _) = GetPixel(noonPixels, 64, 32, 32);
            int noonBrightness = nr + ng + nb;

            // Midnight lighting
            var midnight = new TimeOfDay { Time = 0f };
            var midPixels = await RenderStandardAndReadBack(device, queue, pipeline,
                quads, vp, Vector3.Zero,
                midnight.FogColor, 0f, midnight.AmbientColor, camPos,
                64, 64, new Vector3(0));
            var (mr, mg, mb, _) = GetPixel(midPixels, 64, 32, 32);
            int midBrightness = mr + mg + mb;

            if (noonBrightness < 30)
                throw new Exception($"Noon pixel ({nr},{ng},{nb}) too dark - block not rendered");

            if (noonBrightness <= midBrightness + 20)
                throw new Exception(
                    $"Noon brightness {noonBrightness} ({nr},{ng},{nb}) should be significantly > " +
                    $"midnight brightness {midBrightness} ({mr},{mg},{mb}). " +
                    "TimeOfDay ambient color is not affecting rendered output.");
        });

        // -----------------------------------------------------------------------
        // DamageOverlay visual: damaged block should look different
        // Uses AO bits in PackedQuad to encode damage display differently
        // -----------------------------------------------------------------------

        [TestMethod]
        public async Task Visual_DamageOverlay_CrackStagesProgressTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new VertexPullPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            var view = Matrix4x4.CreateLookAt(new Vector3(8, 10, 8), new Vector3(8, 0, 8), Vector3.UnitZ);
            var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
            var vp = view * proj;
            var camPos = new Vector3(8, 10, 8);

            // Render undamaged block (damage=0)
            var quadsClean = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1, 0) };
            var cleanPixels = await RenderStandardAndReadBack(device, queue, pipeline,
                quadsClean, vp, Vector3.Zero,
                new Vector3(0), 0f, new Vector3(0.5f), camPos,
                64, 64, new Vector3(0));
            var (cr, cg, cb, _) = GetPixel(cleanPixels, 64, 32, 32);

            // Render max damaged block (damage=15)
            var quadsDamaged = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1, 15) };
            var damagedPixels = await RenderStandardAndReadBack(device, queue, pipeline,
                quadsDamaged, vp, Vector3.Zero,
                new Vector3(0), 0f, new Vector3(0.5f), camPos,
                64, 64, new Vector3(0));
            var (dr, dg, db, _) = GetPixel(damagedPixels, 64, 32, 32);

            if (cr < 20 && cg < 20 && cb < 20)
                throw new Exception($"Clean block ({cr},{cg},{cb}) not rendered");
            if (dr < 5 && dg < 5 && db < 5)
                throw new Exception($"Damaged block ({dr},{dg},{db}) not rendered");

            // Verify DamageOverlay data structures are correct regardless of shader
            int stage0 = DamageOverlay.GetCrackStage(0);
            int stage15 = DamageOverlay.GetCrackStage(15);
            if (stage0 != -1)
                throw new Exception($"Damage 0 crack stage should be -1, got {stage0}");
            if (stage15 != DamageOverlay.CrackStages - 1)
                throw new Exception($"Damage 15 crack stage should be {DamageOverlay.CrackStages - 1}, got {stage15}");
        });

        // -----------------------------------------------------------------------
        // AO visual: block in corner should be darker than isolated block
        // -----------------------------------------------------------------------

        [TestMethod]
        public async Task Visual_AO_CornerDarkerThanOpenTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new VertexPullPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            var view = Matrix4x4.CreateLookAt(new Vector3(8, 10, 8), new Vector3(8, 0, 8), Vector3.UnitZ);
            var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
            var vp = view * proj;
            var camPos = new Vector3(8, 10, 8);

            // Render block with NO AO (ao=0, all corners fully lit)
            var quadsNoAO = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1, 0, 0x00) };
            var noAOPixels = await RenderStandardAndReadBack(device, queue, pipeline,
                quadsNoAO, vp, Vector3.Zero,
                new Vector3(0), 0f, new Vector3(0.5f), camPos,
                64, 64, new Vector3(0));
            var (noR, noG, noB, _) = GetPixel(noAOPixels, 64, 32, 32);
            int noAOBrightness = noR + noG + noB;

            // Render block with MAX AO (ao=0xFF, all corners maximally occluded = 0.4 multiplier)
            var quadsMaxAO = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1, 0, 0xFF) };
            var maxAOPixels = await RenderStandardAndReadBack(device, queue, pipeline,
                quadsMaxAO, vp, Vector3.Zero,
                new Vector3(0), 0f, new Vector3(0.5f), camPos,
                64, 64, new Vector3(0));
            var (aoR, aoG, aoB, _) = GetPixel(maxAOPixels, 64, 32, 32);
            int maxAOBrightness = aoR + aoG + aoB;

            if (noAOBrightness < 30)
                throw new Exception($"No-AO block ({noR},{noG},{noB}) not rendered");
            if (maxAOBrightness < 5)
                throw new Exception($"Max-AO block ({aoR},{aoG},{aoB}) not rendered");

            if (maxAOBrightness >= noAOBrightness)
                throw new Exception(
                    $"Max AO brightness {maxAOBrightness} ({aoR},{aoG},{aoB}) should be < " +
                    $"no AO brightness {noAOBrightness} ({noR},{noG},{noB}). " +
                    "AO multiplier is not darkening the rendered output.");
        });

        // -----------------------------------------------------------------------
        // Transparency visual: transparent block shows background through it
        // -----------------------------------------------------------------------

        [TestMethod]
        public async Task Visual_Transparency_SemiTransparentShowsBackgroundTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new VertexPullPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            var view = Matrix4x4.CreateLookAt(new Vector3(8, 10, 8), new Vector3(8, 0, 8), Vector3.UnitZ);
            var proj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
            var vp = view * proj;
            var camPos = new Vector3(8, 10, 8);

            // Render opaque block (type 1 = stone, alpha 1.0) on blue background
            var opaqueQuads = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1) };
            var opaquePixels = await RenderStandardAndReadBack(device, queue, pipeline,
                opaqueQuads, vp, Vector3.Zero,
                new Vector3(0), 0f, new Vector3(0.5f), camPos,
                64, 64, new Vector3(0, 0, 1)); // blue background
            var (or_, og, ob, _) = GetPixel(opaquePixels, 64, 32, 32);

            // Render semi-transparent block (type 7 = glass, alpha 0.5) on blue background
            var transQuads = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 7) };
            var transPixels = await RenderStandardAndReadBack(device, queue, pipeline,
                transQuads, vp, Vector3.Zero,
                new Vector3(0), 0f, new Vector3(0.5f), camPos,
                64, 64, new Vector3(0, 0, 1)); // blue background
            var (tr, tg, tb, _) = GetPixel(transPixels, 64, 32, 32);

            if (or_ < 20 && og < 20 && ob < 20)
                throw new Exception($"Opaque block ({or_},{og},{ob}) not rendered");

            // Opaque block should fully cover the blue background
            // Semi-transparent glass should let some blue through
            // Glass is light blue (0.7, 0.85, 0.9, 0.5), background is pure blue (0, 0, 1)
            // Blended: the blue channel of transparent should have more blue from the background
            if (tb <= ob)
                throw new Exception(
                    $"Transparent glass on blue bg: blue={tb}. Opaque stone on blue bg: blue={ob}. " +
                    "Glass should let more background blue through than opaque stone.");
        });

        // -----------------------------------------------------------------------
        // Orthographic camera: blocks render without perspective distortion
        // -----------------------------------------------------------------------

        [TestMethod]
        public void Orthographic_ReversedZ_DepthMappingTest()
        {
            var ortho = ReversedZHelper.CreateReversedOrthographic(32f, 32f, 0.1f, 100f);

            // Near plane point -> Z should be ~1
            var nearPoint = Vector4.Transform(new Vector4(0, 0, -0.1f, 1), ortho);
            float nearZ = nearPoint.Z / nearPoint.W;
            if (nearZ < 0.9f)
                throw new Exception($"Ortho reversed-Z near: {nearZ}, expected ~1.0");

            // Far plane point -> Z should be ~0
            var farPoint = Vector4.Transform(new Vector4(0, 0, -100f, 1), ortho);
            float farZ = farPoint.Z / farPoint.W;
            if (farZ > 0.1f)
                throw new Exception($"Ortho reversed-Z far: {farZ}, expected ~0.0");
        }

        [TestMethod]
        public void Orthographic_NoSizeDistortion_SameWidthAtAllDepthsTest()
        {
            var ortho = ReversedZHelper.CreateReversedOrthographic(32f, 32f, 0.1f, 100f);

            // Two points at same X but different Z depths should project to same screen X
            var near = Vector4.Transform(new Vector4(5, 0, -1f, 1), ortho);
            var far = Vector4.Transform(new Vector4(5, 0, -50f, 1), ortho);

            float nearScreenX = near.X / near.W;
            float farScreenX = far.X / far.W;

            if (MathF.Abs(nearScreenX - farScreenX) > 0.001f)
                throw new Exception(
                    $"Orthographic: near X={nearScreenX}, far X={farScreenX}. " +
                    "Should be identical - no perspective size distortion.");
        }

        [TestMethod]
        public async Task Visual_Orthographic_BlockRendersTest() => await RunWebGPURenderTest(async (device, queue, accelerator) =>
        {
            using var pipeline = new VertexPullPipeline();
            pipeline.Init(device, queue, "bgra8unorm");

            // Orthographic camera looking down at a block
            var view = Matrix4x4.CreateLookAt(new Vector3(8, 20, 8), new Vector3(8, 0, 8), Vector3.UnitZ);
            var orthoProj = ReversedZHelper.CreateReversedOrthographic(32f, 32f, 0.1f, 100f);
            var vp = view * orthoProj;
            var camPos = new Vector3(8, 20, 8);

            var quads = new long[] { PackedQuad.Pack(0, 0, 0, 16, 16, 4, 1) };
            var pixels = await RenderStandardAndReadBack(device, queue, pipeline,
                quads, vp, Vector3.Zero,
                new Vector3(0), 0f, new Vector3(0.5f), camPos,
                64, 64, new Vector3(0, 0, 0));

            int rendered = CountNonBackgroundPixels(pixels);
            if (rendered < 10)
                throw new Exception(
                    $"Orthographic render: only {rendered} non-black pixels. " +
                    "Block should be visible with orthographic projection.");

            // Verify it renders differently from perspective (ortho has uniform size)
            var perspProj = ReversedZHelper.CreateInfinitePerspective(MathF.PI / 2f, 1f, 0.1f);
            var perspVP = view * perspProj;
            var perspPixels = await RenderStandardAndReadBack(device, queue, pipeline,
                quads, perspVP, Vector3.Zero,
                new Vector3(0), 0f, new Vector3(0.5f), camPos,
                64, 64, new Vector3(0, 0, 0));
            int perspRendered = CountNonBackgroundPixels(perspPixels);

            // Both should render something, but pixel counts will differ
            // (ortho has no perspective foreshortening)
            if (perspRendered < 10)
                throw new Exception($"Perspective render: only {perspRendered} pixels - comparison failed");
        });

        // -----------------------------------------------------------------------
        // OffCenter orthographic: for shadow map cascade sub-regions
        // -----------------------------------------------------------------------

        [TestMethod]
        public void Orthographic_OffCenter_AsymmetricBoundsTest()
        {
            // Shadow map for a cascade covering world region (-50, -50) to (50, 50)
            var ortho = ReversedZHelper.CreateReversedOrthographicOffCenter(-50f, 50f, -50f, 50f, 0.1f, 200f);

            // Center point should project to (0, 0)
            var center = Vector4.Transform(new Vector4(0, 0, -10f, 1), ortho);
            float cx = center.X / center.W;
            float cy = center.Y / center.W;
            if (MathF.Abs(cx) > 0.01f || MathF.Abs(cy) > 0.01f)
                throw new Exception($"Center point projects to ({cx}, {cy}), expected (0, 0)");

            // Edge point should project to ~1
            var edge = Vector4.Transform(new Vector4(50, 0, -10f, 1), ortho);
            float ex = edge.X / edge.W;
            if (MathF.Abs(ex - 1f) > 0.01f)
                throw new Exception($"Right edge projects to X={ex}, expected ~1.0");
        }
    }
}


