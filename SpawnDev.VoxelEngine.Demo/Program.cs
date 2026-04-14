using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.BlazorJS;
using SpawnDev.VoxelEngine.Demo;
using SpawnDev.VoxelEngine.Demo.UnitTests;

// Print build timestamp
var buildTimestamp = typeof(Program).Assembly
    .GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
    .OfType<System.Reflection.AssemblyMetadataAttribute>()
    .FirstOrDefault(a => a.Key == "BuildTimestamp")?.Value ?? "unknown";
Console.WriteLine($"SpawnDev.VoxelEngine.Demo build: {buildTimestamp}");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

// BlazorJS services
builder.Services.AddBlazorJSRuntime();
builder.Services.AddSingleton<SpawnDev.ILGPU.Services.ShaderDebugService>();

// Register test classes (one per backend)
builder.Services.AddSingleton<WebGPUTests>();
builder.Services.AddSingleton<WasmTests>();
builder.Services.AddSingleton<WebGLTests>();
builder.Services.AddSingleton<DefaultTests>();

await builder.Build().BlazorJSRunAsync();
