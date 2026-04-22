namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WGSL shader source for rendering Dual Marching Cubes (SDF) meshes.
    ///
    /// Vertex pulling from three storage buffers:
    ///   positions  : array&lt;f32&gt;  (3 floats per vertex, world-space)
    ///   normals    : array&lt;f32&gt;  (3 floats per vertex, world-space)
    ///   quadIndices: array&lt;i32&gt;  (4 ints per quad, vertex indices)
    ///
    /// The DMC kernel already bakes chunk world offset into vertex positions,
    /// so sectionOffset in the uniforms is ignored here. UniformData layout is
    /// kept identical to VertexPullPipeline's for test and pipeline-selection
    /// reuse - callers pass sectionOffset = Vector3.Zero for SDF rendering.
    /// </summary>
    public static class SdfRenderShaders
    {
        /// <summary>
        /// Lit mesh shader with Lambertian shading and exponential-squared fog.
        /// Triangulates each 4-corner quad as two triangles with CCW winding.
        ///
        /// Bindings:
        ///   @group(0) @binding(0) uniforms: MVP, fog, ambient, camera
        ///   @group(0) @binding(1) positions: vertex position storage buffer
        ///   @group(0) @binding(2) normals: vertex normal storage buffer
        ///   @group(0) @binding(3) quadIndices: 4-vertex-per-quad index storage buffer
        /// </summary>
        public const string SdfMeshShader = @"
struct Uniforms {
    mvp: mat4x4<f32>,
    sectionOffset: vec3<f32>,
    voxelSize: f32,
    fogColor: vec3<f32>,
    fogDensity: f32,
    ambientColor: vec3<f32>,
    time: f32,
    cameraWorldPos: vec3<f32>,
    _pad0: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> normals: array<f32>;
@group(0) @binding(3) var<storage, read> quadIndices: array<i32>;

// Triangulate 4-corner quad as two CCW triangles: (0,1,2) and (0,2,3).
const cornerOrder = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);

@vertex
fn vs_main(@builtin(vertex_index) vertexID: u32) -> VertexOutput {
    let quadIdx = vertexID / 6u;
    let cornerSlot = cornerOrder[vertexID % 6u];
    let vertexIdx = u32(quadIndices[quadIdx * 4u + cornerSlot]);

    let p = vec3<f32>(
        positions[vertexIdx * 3u + 0u],
        positions[vertexIdx * 3u + 1u],
        positions[vertexIdx * 3u + 2u]);
    let n = vec3<f32>(
        normals[vertexIdx * 3u + 0u],
        normals[vertexIdx * 3u + 1u],
        normals[vertexIdx * 3u + 2u]);

    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(p, 1.0);
    out.worldPos = p;
    out.normal = n;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sunDir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let nrm = normalize(in.normal);
    let NdotL = max(dot(nrm, sunDir), 0.0);
    let baseColor = vec3<f32>(0.55, 0.50, 0.42); // neutral rocky tan
    let lighting = uniforms.ambientColor + vec3<f32>(1.0, 0.95, 0.85) * NdotL;
    let litColor = baseColor * lighting;

    let dist = length(in.worldPos - uniforms.cameraWorldPos);
    let fogFactor = exp(-dist * uniforms.fogDensity * uniforms.fogDensity);
    let finalColor = mix(uniforms.fogColor, litColor, clamp(fogFactor, 0.0, 1.0));

    return vec4<f32>(finalColor, 1.0);
}
";
    }
}
