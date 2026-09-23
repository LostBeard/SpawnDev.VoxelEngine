using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS;
using SpawnDev.UnitTesting;

namespace SpawnDev.VoxelEngine.Demo.Shared.UnitTests
{
    // Harness guard, ported from SpawnDev.ILGPU (BackendTestBase.AdapterIdentity.cs). Until 2026-09-23 this
    // repo's PlaywrightMultiTest launched Playwright's BUNDLED Chromium, which only exposes the SwiftShader
    // SOFTWARE WebGPU adapter on the dev machine: every WebGPU result and timing it produced was CPU time.
    // The harness now launches installed Chrome; this probe fails the run if the browser ever lands on a
    // software adapter again, and prints the adapter identity either way.
    public abstract partial class VoxelEngineTestBase
    {
        [TestMethod]
        public async Task WebGPU_AdapterIdentity_Probe() => await RunTest(async accelerator =>
        {
            if (accelerator is not WebGPUAccelerator)
                throw new UnsupportedTestException($"{accelerator.AcceleratorType}: WebGPU-only adapter probe");
            var JS = SpawnJSRuntime.Instance;
            using var gpu = JS.Get<SpawnJSObject>("navigator.gpu");
            if (gpu == null) throw new Exception("navigator.gpu missing");
            using var adapter = await gpu.JSRef!.CallAsync<SpawnJSObject>("requestAdapter");
            if (adapter == null) throw new Exception("requestAdapter returned null");
            string vendor = "?", arch = "?", device = "?", desc = "?";
            bool fallback = false;
            using (var info = adapter.JSRef!.Get<SpawnJSObject?>("info"))
            {
                if (info != null)
                {
                    vendor = info.JSRef!.Get<string?>("vendor") ?? "?";
                    arch = info.JSRef!.Get<string?>("architecture") ?? "?";
                    device = info.JSRef!.Get<string?>("device") ?? "?";
                    desc = info.JSRef!.Get<string?>("description") ?? "?";
                }
            }
            fallback = adapter.JSRef!.Get<bool?>("isFallbackAdapter") ?? false;
            Console.WriteLine($"[AdapterProbe] vendor={vendor} arch={arch} device={device} desc='{desc}' isFallbackAdapter={fallback}");
            if (fallback || arch.Contains("swiftshader", StringComparison.OrdinalIgnoreCase))
                throw new Exception($"SOFTWARE WebGPU adapter in the harness: vendor={vendor} arch={arch} isFallbackAdapter={fallback} - " +
                    "results from this session are CPU-rasterizer time. Fix the browser launch (PMT Channel=chrome, --disable-software-rasterizer).");
        });
    }
}
