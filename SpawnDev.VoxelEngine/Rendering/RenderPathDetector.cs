using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Which render path a consumer app should take, determined at init by probing browser capabilities.
    /// Per `Lost/Lost/Plans/PLAN-VR-Render-Architecture.md`, WebGPU-only is the shipping target; the
    /// split is about whether the VR render path uses the WebXR+WebGPU binding (primary) or the
    /// WebGPU-compute + WebGL2-render hybrid fallback.
    /// </summary>
    public enum RenderPath
    {
        /// <summary>Detection has not run yet.</summary>
        Unknown = 0,

        /// <summary>No WebGPU at all. Consumer should show an unsupported-browser page.</summary>
        None,

        /// <summary>WebGPU present, no WebXR immersive-vr. Desktop browser path, non-VR.</summary>
        WebGpuOnly,

        /// <summary>
        /// WebGPU + WebXR immersive-vr, but WebXR+WebGPU binding API is NOT exposed. Consumer runs
        /// compute on WebGPU, renders via WebGL2 in the WebXR session, bridges via OffscreenCanvas
        /// + transferToImageBitmap. This is the shipping path on Meta Quest Browser as of 2026-04-22
        /// (v146.0 has compute-only WebGPU; binding spec not yet shipped).
        /// </summary>
        HybridWebGl2,

        /// <summary>
        /// WebGPU + WebXR immersive-vr + WebXR+WebGPU binding. Full WebGPU end-to-end. Primary target.
        /// Available on Chrome Canary 135+ with `WebXR Projection Layers` + `WebXR/WebGPU Bindings`
        /// flags enabled, and on any future browser that ships the binding spec (Meta Quest Browser
        /// plausibly Q3-Q4 2026).
        /// </summary>
        FullWebGpu,
    }

    /// <summary>
    /// Result of <see cref="RenderPathDetector.DetectAsync"/>. Includes the chosen path plus the
    /// underlying capability flags and adapter info, so consumers can log/display the probe result
    /// for debugging and users can see why a given path was chosen.
    /// </summary>
    public record RenderPathReport(
        RenderPath Path,
        bool HasWebGpu,
        bool HasImmersiveVr,
        bool HasImmersiveAr,
        bool HasXRGPUBinding,
        string? AdapterVendor,
        string? AdapterArchitecture,
        string? AdapterDescription,
        bool IsFallbackAdapter,
        bool HasShaderF16,
        bool HasSubgroups,
        bool HasTimestampQuery,
        int MaxStorageBuffersPerShaderStage,
        int SubgroupMinSize,
        int SubgroupMaxSize)
    {
        /// <summary>
        /// Empty report for the no-WebGPU case.
        /// </summary>
        public static RenderPathReport None() => new(
            Path: RenderPath.None,
            HasWebGpu: false, HasImmersiveVr: false, HasImmersiveAr: false, HasXRGPUBinding: false,
            AdapterVendor: null, AdapterArchitecture: null, AdapterDescription: null,
            IsFallbackAdapter: false, HasShaderF16: false, HasSubgroups: false, HasTimestampQuery: false,
            MaxStorageBuffersPerShaderStage: 0, SubgroupMinSize: 0, SubgroupMaxSize: 0);
    }

    /// <summary>
    /// Runtime capability probe. Decides which render path a SpawnDev consumer should take by
    /// checking for WebGPU availability, WebXR immersive-vr support, and whether the WebXR+WebGPU
    /// binding API (`XRGPUBinding`) is exposed by the current browser. Single branch point for the
    /// shipping architecture; every downstream subsystem reads the resulting <see cref="RenderPath"/>
    /// and takes the appropriate code path. No backend-specific consumer code beyond this point.
    /// </summary>
    public static class RenderPathDetector
    {
        /// <summary>
        /// Probe the current browser. Safe to call at consumer app init. Does not consume the
        /// adapter (only reads features/limits/info).
        /// </summary>
        public static async Task<RenderPathReport> DetectAsync(BlazorJSRuntime js)
        {
            // WebGPU presence. If navigator.gpu is undefined, nothing else matters.
            using var nav = new Navigator();
            using var gpu = nav.Gpu;
            if (gpu == null)
                return RenderPathReport.None();

            // WebXR presence + session support. Some browsers throw from isSessionSupported
            // when the platform has no XR hardware attached; treat that as "not supported."
            bool hasImmersiveVr = false;
            bool hasImmersiveAr = false;
            using var xr = nav.XR;
            if (xr != null)
            {
                try { hasImmersiveVr = await xr.IsSessionSupported(XRSessionMode.ImmersiveVR); }
                catch { hasImmersiveVr = false; }
                try { hasImmersiveAr = await xr.IsSessionSupported(XRSessionMode.ImmersiveAR); }
                catch { hasImmersiveAr = false; }
            }

            // WebXR+WebGPU binding: the XRGPUBinding class is exposed at the global scope when the
            // `WebXR/WebGPU Bindings` flag is on (Chrome Canary 135+) and eventually when the spec
            // ships stable. typeof check is the documented feature-detection pattern per the
            // spec explainer.
            bool hasXRGPUBinding = !js.IsUndefined("XRGPUBinding");

            // Probe adapter info + features + limits. Do NOT call requestDevice - that "consumes"
            // the adapter and we just want capability info here.
            using var adapter = await gpu.RequestAdapter();
            if (adapter == null)
            {
                // WebGPU exists but no adapter available. Rare, but treat as None for consumer
                // dispatch purposes (there's nothing to render with).
                return RenderPathReport.None() with { HasWebGpu = true };
            }

            using var info = adapter.Info;
            using var features = adapter.Features;
            using var limits = adapter.Limits;

            var report = new RenderPathReport(
                Path: ChoosePath(hasImmersiveVr, hasXRGPUBinding),
                HasWebGpu: true,
                HasImmersiveVr: hasImmersiveVr,
                HasImmersiveAr: hasImmersiveAr,
                HasXRGPUBinding: hasXRGPUBinding,
                AdapterVendor: info?.Vendor,
                AdapterArchitecture: info?.Architecture,
                AdapterDescription: info?.Description,
                IsFallbackAdapter: adapter.IsFallbackAdapter,
                HasShaderF16: features?.Has("shader-f16") ?? false,
                HasSubgroups: features?.Has("subgroups") ?? false,
                HasTimestampQuery: features?.Has("timestamp-query") ?? false,
                MaxStorageBuffersPerShaderStage: limits?.MaxStorageBuffersPerShaderStage ?? 0,
                SubgroupMinSize: info?.SubgroupMinSize ?? 0,
                SubgroupMaxSize: info?.SubgroupMaxSize ?? 0);

            return report;
        }

        static RenderPath ChoosePath(bool hasImmersiveVr, bool hasXRGPUBinding) =>
            (hasImmersiveVr, hasXRGPUBinding) switch
            {
                (true, true) => RenderPath.FullWebGpu,
                (true, false) => RenderPath.HybridWebGl2,
                (false, _) => RenderPath.WebGpuOnly,
            };
    }
}
