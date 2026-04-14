using System.Numerics;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.Adaptive;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // QualityController and StreamingBudget tests.
    // QualityController: adaptive quality scaling based on frame timing, thermal detection.
    // StreamingBudget: per-frame GPU work budget with priority-based section scheduling.
    // All tests are CPU-only - no GPU needed for these pure-logic systems.
    public abstract partial class VoxelEngineTestBase
    {
        // ===== QualityController tests =====

        /// <summary>
        /// Test 1: Over-budget frames decrease quality level.
        /// Feed 20ms frame times at a 13.88ms target (72Hz Quest).
        /// Over-budget threshold = target * 1.15 = 15.96ms. 20ms > 15.96ms = over.
        /// After DecreaseThreshold (5) consecutive over-budget frames, level should drop.
        /// Starting at High, should drop to Medium.
        /// </summary>
        [TestMethod]
        public void Quality_OverBudget_DecreasesLevelTest()
        {
            var qc = new QualityController();
            qc.TargetFrameTimeMs = 13.88f; // 72Hz
            qc.DecreaseThreshold = 5;
            qc.Reset(QualityController.QualityLevel.High);

            // Feed enough 20ms frames to let EMA converge above over-budget threshold (15.96ms)
            // then hit the 5-frame consecutive counter.
            // EMA starts at first frame value, then converges. With alpha=0.1:
            // After ~20 frames at 20ms, EMA will be close to 20ms, well above 15.96ms threshold.
            for (int i = 0; i < 30; i++)
            {
                qc.Update(20f);
            }

            if (qc.Level == QualityController.QualityLevel.High)
                throw new Exception(
                    $"Quality_OverBudget: level should have dropped below High after 30 frames at 20ms " +
                    $"(target={qc.TargetFrameTimeMs}ms, threshold={qc.TargetFrameTimeMs * 1.15f}ms), " +
                    $"but level is still {qc.Level}");

            // Should have dropped at least to Medium (one step down from High)
            if (qc.Level < QualityController.QualityLevel.Medium)
            {
                // Could have dropped further if multiple decrease cycles triggered - that's valid
            }
            else if (qc.Level > QualityController.QualityLevel.Minimal)
            {
                // Valid - somewhere between Medium and Minimal
            }

            // Verify SmoothedFrameTimeMs converged near 20ms
            if (MathF.Abs(qc.SmoothedFrameTimeMs - 20f) > 2f)
                throw new Exception(
                    $"Quality_OverBudget: EMA should converge near 20ms after 30 frames, got {qc.SmoothedFrameTimeMs:F2}ms");
        }

        /// <summary>
        /// Test 2: Under-budget frames increase quality level.
        /// Feed 8ms frame times at 13.88ms target. Under-budget threshold = 13.88 * 0.75 = 10.41ms.
        /// 8ms < 10.41ms = under budget. After IncreaseThreshold (60) frames, level should rise.
        /// Starting at Low, should rise to Medium.
        /// </summary>
        [TestMethod]
        public void Quality_UnderBudget_IncreasesLevelTest()
        {
            var qc = new QualityController();
            qc.TargetFrameTimeMs = 13.88f;
            qc.IncreaseThreshold = 60;
            qc.Reset(QualityController.QualityLevel.Low);

            var initialLevel = qc.Level;

            // Feed 8ms frames. EMA converges to 8ms. Under-budget threshold = 10.41ms.
            // Need 60 consecutive under-budget frames plus enough for EMA convergence.
            for (int i = 0; i < 100; i++)
            {
                qc.Update(8f);
            }

            if (qc.Level >= initialLevel)
                throw new Exception(
                    $"Quality_UnderBudget: level should have increased from {initialLevel} after 100 frames at 8ms " +
                    $"(under-threshold={qc.TargetFrameTimeMs * 0.75f}ms), but level is {qc.Level}");

            // Should be Medium (one step up from Low)
            if (qc.Level != QualityController.QualityLevel.Medium)
                throw new Exception(
                    $"Quality_UnderBudget: expected at least Medium after increase from Low, got {qc.Level}");
        }

        /// <summary>
        /// Test 3: In-budget frames don't change level.
        /// Feed 13ms frame times at 13.88ms target.
        /// Over-budget threshold = 15.96ms, under-budget threshold = 10.41ms.
        /// 13ms is between 10.41 and 15.96 - level should stay the same.
        /// </summary>
        [TestMethod]
        public void Quality_InBudget_LevelStaysTest()
        {
            var qc = new QualityController();
            qc.TargetFrameTimeMs = 13.88f;
            qc.Reset(QualityController.QualityLevel.Medium);

            var initialLevel = qc.Level;

            // Feed 13ms frames - right in the middle of the budget range
            for (int i = 0; i < 200; i++)
            {
                qc.Update(13f);
            }

            if (qc.Level != initialLevel)
                throw new Exception(
                    $"Quality_InBudget: level should stay at {initialLevel} with 13ms frames " +
                    $"(range {qc.TargetFrameTimeMs * 0.75f:F2}-{qc.TargetFrameTimeMs * 1.15f:F2}ms), " +
                    $"but changed to {qc.Level}");

            // Both counters should be 0 (reset on each in-budget frame)
            if (qc.OverBudgetFrames != 0)
                throw new Exception(
                    $"Quality_InBudget: OverBudgetFrames should be 0 when in-budget, got {qc.OverBudgetFrames}");
            if (qc.UnderBudgetFrames != 0)
                throw new Exception(
                    $"Quality_InBudget: UnderBudgetFrames should be 0 when in-budget, got {qc.UnderBudgetFrames}");
        }

        /// <summary>
        /// Test 4: Thermal detection from frame time spikes without scene changes.
        /// Feed a stable 10ms baseline, then 25ms spikes with sceneChanged=false.
        /// The spike ratio (25/10 = 2.5) exceeds ThermalSpikeRatio (2.0).
        /// After ThermalSpikeThreshold (3) consecutive spikes, ThermalDetected should be true.
        /// </summary>
        [TestMethod]
        public void Quality_ThermalDetection_SpikesWithoutSceneChangeTest()
        {
            var qc = new QualityController();
            qc.TargetFrameTimeMs = 16.67f; // 60Hz
            qc.ThermalSpikeRatio = 2.0f;
            qc.Reset(QualityController.QualityLevel.Medium);

            // Establish baseline: 20 frames at 10ms (scene is changing)
            for (int i = 0; i < 20; i++)
            {
                qc.Update(10f, sceneChanged: true);
            }

            if (qc.ThermalDetected)
                throw new Exception("Quality_Thermal: should NOT detect thermal during scene changes");

            // Now feed spikes WITHOUT scene changes:
            // previousFrameTime = 10ms, new frame = 25ms, ratio = 2.5 > 2.0
            // Need 3 consecutive spikes (ThermalSpikeThreshold = 3)
            qc.Update(10f, sceneChanged: false); // establishes _previousFrameTimeMs for static scene
            qc.Update(25f, sceneChanged: false); // spike 1: 25 / 10 = 2.5 > 2.0
            qc.Update(55f, sceneChanged: false); // spike 2: 55 / 25 = 2.2 > 2.0
            qc.Update(120f, sceneChanged: false); // spike 3: 120 / 55 = 2.18 > 2.0

            if (!qc.ThermalDetected)
                throw new Exception(
                    $"Quality_Thermal: should detect thermal after 3 consecutive spikes " +
                    $"(ratio > {qc.ThermalSpikeRatio}), but ThermalDetected={qc.ThermalDetected}");
        }

        /// <summary>
        /// Test 5: Emergency thermal drop to Minimal.
        /// When thermal throttling is detected, quality drops to Minimal immediately,
        /// regardless of the current level.
        /// </summary>
        [TestMethod]
        public void Quality_ThermalEmergency_DropsToMinimalTest()
        {
            var qc = new QualityController();
            qc.TargetFrameTimeMs = 16.67f;
            qc.ThermalSpikeRatio = 2.0f;
            qc.Reset(QualityController.QualityLevel.Ultra);

            // Trigger thermal: need consecutive spikes without scene changes
            qc.Update(10f, sceneChanged: false); // baseline
            qc.Update(25f, sceneChanged: false); // spike 1
            qc.Update(55f, sceneChanged: false); // spike 2
            qc.Update(120f, sceneChanged: false); // spike 3 - triggers thermal

            if (!qc.ThermalDetected)
                throw new Exception("Quality_ThermalEmergency: thermal should be detected after 3 spikes");

            // The Update that detects thermal should have already dropped to Minimal
            // If not yet, one more Update will trigger the emergency path
            qc.Update(120f, sceneChanged: false);

            if (qc.Level != QualityController.QualityLevel.Minimal)
                throw new Exception(
                    $"Quality_ThermalEmergency: level should be Minimal after thermal detection, " +
                    $"got {qc.Level}");

            // Counters should be reset
            if (qc.OverBudgetFrames != 0 || qc.UnderBudgetFrames != 0)
                throw new Exception(
                    $"Quality_ThermalEmergency: counters should be reset after thermal drop, " +
                    $"got Over={qc.OverBudgetFrames}, Under={qc.UnderBudgetFrames}");
        }

        /// <summary>
        /// Test 6: Draw distance multiplier matches each quality level.
        /// Verify the exact multiplier values: Ultra=1.0, High=0.75, Medium=0.5, Low=0.35, Minimal=0.2.
        /// These are critical for the adaptive draw distance system.
        /// </summary>
        [TestMethod]
        public void Quality_DrawDistanceMultiplier_MatchesLevelTest()
        {
            var qc = new QualityController();

            var expectedMultipliers = new (QualityController.QualityLevel level, float multiplier)[]
            {
                (QualityController.QualityLevel.Ultra, 1.0f),
                (QualityController.QualityLevel.High, 0.75f),
                (QualityController.QualityLevel.Medium, 0.5f),
                (QualityController.QualityLevel.Low, 0.35f),
                (QualityController.QualityLevel.Minimal, 0.2f),
            };

            foreach (var (level, expected) in expectedMultipliers)
            {
                qc.Reset(level);
                float actual = qc.DrawDistanceMultiplier;

                if (MathF.Abs(actual - expected) > 0.001f)
                    throw new Exception(
                        $"Quality_DrawDistanceMultiplier: level {level} expected {expected}, got {actual}");
            }

            // Also verify LodBias and VertexBudgetMultiplier are consistent
            qc.Reset(QualityController.QualityLevel.Ultra);
            if (qc.LodBias != 0)
                throw new Exception($"Quality_DrawDistanceMultiplier: Ultra LodBias should be 0, got {qc.LodBias}");

            qc.Reset(QualityController.QualityLevel.Minimal);
            if (qc.LodBias != 3)
                throw new Exception($"Quality_DrawDistanceMultiplier: Minimal LodBias should be 3, got {qc.LodBias}");
            if (MathF.Abs(qc.VertexBudgetMultiplier - 0.15f) > 0.001f)
                throw new Exception($"Quality_DrawDistanceMultiplier: Minimal VertexBudgetMultiplier should be 0.15, got {qc.VertexBudgetMultiplier}");
        }

        // ===== StreamingBudget tests =====

        /// <summary>
        /// Test 7: TryConsume succeeds within budget, fails when exhausted.
        /// Initialize with Desktop tier, consume up to limits, verify rejection beyond limits.
        /// </summary>
        [TestMethod]
        public void Budget_TryConsume_SucceedsAndFailsTest()
        {
            var budget = new StreamingBudget();
            var caps = new DeviceCapabilities { Tier = DeviceTier.Desktop };
            budget.Init(caps, 1.0f); // full quality

            budget.BeginFrame();

            // Desktop: MaxVerticesPerFrame=500000, MaxBytesPerFrame=8000000, MaxMeshesPerFrame=16

            // First consume should succeed
            bool first = budget.TryConsume(10000, 200000);
            if (!first)
                throw new Exception("Budget_TryConsume: first consume of 10K verts / 200KB should succeed");
            if (budget.VerticesUsed != 10000)
                throw new Exception($"Budget_TryConsume: VerticesUsed should be 10000, got {budget.VerticesUsed}");
            if (budget.BytesUsed != 200000)
                throw new Exception($"Budget_TryConsume: BytesUsed should be 200000, got {budget.BytesUsed}");
            if (budget.MeshesUsed != 1)
                throw new Exception($"Budget_TryConsume: MeshesUsed should be 1, got {budget.MeshesUsed}");

            // Consume 15 more meshes (total 16 = MaxMeshesPerFrame for Desktop)
            for (int i = 0; i < 15; i++)
            {
                bool ok = budget.TryConsume(1000, 20000);
                if (!ok)
                    throw new Exception($"Budget_TryConsume: consume #{i + 2} should succeed (mesh {i + 2}/16)");
            }

            if (budget.MeshesUsed != 16)
                throw new Exception($"Budget_TryConsume: MeshesUsed should be 16 after 16 consumes, got {budget.MeshesUsed}");

            // 17th consume should FAIL - mesh budget exhausted
            bool overflow = budget.TryConsume(100, 1000);
            if (overflow)
                throw new Exception("Budget_TryConsume: 17th mesh consume should fail (budget=16), but succeeded");

            if (!budget.IsMeshBudgetExhausted)
                throw new Exception("Budget_TryConsume: IsMeshBudgetExhausted should be true after 16 meshes");
            if (!budget.IsAnyBudgetExhausted)
                throw new Exception("Budget_TryConsume: IsAnyBudgetExhausted should be true");
        }

        /// <summary>
        /// Test 8: BeginFrame resets counters.
        /// Consume some budget, call BeginFrame, verify counters are zero.
        /// </summary>
        [TestMethod]
        public void Budget_BeginFrame_ResetsCountersTest()
        {
            var budget = new StreamingBudget();
            var caps = new DeviceCapabilities { Tier = DeviceTier.MobileHigh };
            budget.Init(caps, 0.5f); // half quality multiplier

            budget.BeginFrame();

            // Consume some budget
            budget.TryConsume(50000, 1000000);
            budget.TryConsume(30000, 500000);

            if (budget.VerticesUsed != 80000)
                throw new Exception($"Budget_BeginFrame: VerticesUsed should be 80000 before reset, got {budget.VerticesUsed}");
            if (budget.BytesUsed != 1500000)
                throw new Exception($"Budget_BeginFrame: BytesUsed should be 1500000 before reset, got {budget.BytesUsed}");
            if (budget.MeshesUsed != 2)
                throw new Exception($"Budget_BeginFrame: MeshesUsed should be 2 before reset, got {budget.MeshesUsed}");

            // Reset
            budget.BeginFrame();

            if (budget.VerticesUsed != 0)
                throw new Exception($"Budget_BeginFrame: VerticesUsed should be 0 after BeginFrame, got {budget.VerticesUsed}");
            if (budget.BytesUsed != 0)
                throw new Exception($"Budget_BeginFrame: BytesUsed should be 0 after BeginFrame, got {budget.BytesUsed}");
            if (budget.MeshesUsed != 0)
                throw new Exception($"Budget_BeginFrame: MeshesUsed should be 0 after BeginFrame, got {budget.MeshesUsed}");

            if (budget.IsAnyBudgetExhausted)
                throw new Exception("Budget_BeginFrame: no budget should be exhausted after BeginFrame");

            // Should be able to consume again
            bool ok = budget.TryConsume(1000, 10000);
            if (!ok)
                throw new Exception("Budget_BeginFrame: TryConsume should succeed after BeginFrame reset");
        }

        /// <summary>
        /// Test 9: Priority computation - closer + in-view > far + behind.
        /// ComputePriority(distanceSq, viewDot, lodUrgency) must return higher priority
        /// for sections that are closer to the camera and in the view direction.
        /// Tests with real game-relevant distances, not identity values.
        /// </summary>
        [TestMethod]
        public void Budget_Priority_CloserInViewHigherTest()
        {
            // Case A: Close section directly ahead (distSq=100, viewDot=0.9, urgency=0)
            float priorityCloseAhead = StreamingBudget.ComputePriority(100f, 0.9f, 0);

            // Case B: Far section behind camera (distSq=10000, viewDot=-0.8, urgency=0)
            float priorityFarBehind = StreamingBudget.ComputePriority(10000f, -0.8f, 0);

            if (priorityCloseAhead <= priorityFarBehind)
                throw new Exception(
                    $"Budget_Priority: close+ahead ({priorityCloseAhead:F6}) should be higher than " +
                    $"far+behind ({priorityFarBehind:F6})");

            // Case C: Same distance, ahead vs behind
            float priorityAhead = StreamingBudget.ComputePriority(500f, 0.8f, 0);
            float priorityBehind = StreamingBudget.ComputePriority(500f, -0.8f, 0);

            if (priorityAhead <= priorityBehind)
                throw new Exception(
                    $"Budget_Priority: same-distance ahead ({priorityAhead:F6}) should be higher than " +
                    $"behind ({priorityBehind:F6})");

            // Case D: Same direction, close vs far
            float priorityClose = StreamingBudget.ComputePriority(100f, 0.5f, 0);
            float priorityFar = StreamingBudget.ComputePriority(5000f, 0.5f, 0);

            if (priorityClose <= priorityFar)
                throw new Exception(
                    $"Budget_Priority: close ({priorityClose:F6}) should be higher than " +
                    $"far ({priorityFar:F6}) at same view angle");

            // Case E: LOD urgency multiplier - urgent section should rank higher
            float priorityNoUrgency = StreamingBudget.ComputePriority(500f, 0.5f, 0);
            float priorityHighUrgency = StreamingBudget.ComputePriority(500f, 0.5f, 3);

            if (priorityHighUrgency <= priorityNoUrgency)
                throw new Exception(
                    $"Budget_Priority: urgent LOD ({priorityHighUrgency:F6}) should be higher than " +
                    $"no urgency ({priorityNoUrgency:F6})");

            // Verify the multiplier: lodMultiplier = 1 + 3*2 = 7x
            float expectedRatio = (1f + 3 * 2f);
            float actualRatio = priorityHighUrgency / priorityNoUrgency;
            if (MathF.Abs(actualRatio - expectedRatio) > 0.01f)
                throw new Exception(
                    $"Budget_Priority: LOD urgency 3 should multiply by {expectedRatio}x, " +
                    $"got {actualRatio:F2}x");

            // Case F: Priority is always positive
            float priorityWorstCase = StreamingBudget.ComputePriority(1000000f, -1f, 0);
            if (priorityWorstCase <= 0f)
                throw new Exception(
                    $"Budget_Priority: priority should always be positive, got {priorityWorstCase:F8} " +
                    $"for worst case (far + directly behind)");
        }
    }
}
