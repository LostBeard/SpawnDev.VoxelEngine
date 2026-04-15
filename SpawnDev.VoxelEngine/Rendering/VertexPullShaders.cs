namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WGSL shader source for the vertex pulling voxel renderer.
    ///
    /// Vertex pulling: no vertex buffer. The vertex shader reads PackedQuad data
    /// from a storage buffer and computes vertex positions procedurally.
    ///
    /// quadIdx = vertexID / 6  (which quad)
    /// cornerIdx = vertexID % 6  (which corner of the 2-triangle quad)
    ///
    /// This eliminates vertex buffer management entirely - the GPU reads directly
    /// from the PackedQuad buffer that the greedy merge kernel produced.
    /// One storage buffer binding, one draw call per section (or per face group).
    ///
    /// Two fragment shader modes:
    /// 1. Solid color: block type -> color lookup table (debug / AubsCraft simple)
    /// 2. Texture array: block type + face -> texture layer index (production)
    /// </summary>
    public static class VertexPullShaders
    {
        /// <summary>
        /// Main vertex pulling shader with solid color fragment.
        ///
        /// Bindings:
        ///   @group(0) @binding(0) uniforms: MVP matrix, section world offset, voxel size
        ///   @group(0) @binding(1) quads: storage buffer of packed i64 quads
        ///   @group(0) @binding(2) colors: storage buffer of per-block-type RGBA colors
        /// </summary>
        public const string SolidColorShader = @"
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
    @location(0) color: vec4<f32>,
    @location(1) worldPos: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) ao: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> quads: array<vec2<u32>>; // i64 as vec2<u32>
@group(0) @binding(2) var<storage, read> blockColors: array<vec4<f32>>;

// Face normals (matches VoxelMeshConstants face order)
const faceNormals = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0),   // +X
    vec3<f32>(-1.0, 0.0, 0.0),  // -X
    vec3<f32>(0.0, 0.0, 1.0),   // +Z
    vec3<f32>(0.0, 0.0, -1.0),  // -Z
    vec3<f32>(0.0, 1.0, 0.0),   // +Y
    vec3<f32>(0.0, -1.0, 0.0),  // -Y
);

// Corner offsets for each face (2 triangles = 6 vertices per quad)
// Each face has 4 corners; the 6 vertices index into them as: 0,1,2, 1,3,2
const cornerOrder = array<u32, 6>(0u, 1u, 2u, 1u, 3u, 2u);

// AO light multipliers
const aoMultiplier = array<f32, 4>(1.0, 0.8, 0.6, 0.4);

@vertex
fn vs_main(@builtin(vertex_index) vertexID: u32) -> VertexOutput {
    let quadIdx = vertexID / 6u;
    let cornerIdx = cornerOrder[vertexID % 6u];

    // Unpack i64 quad (stored as vec2<u32>: lo, hi)
    let lo = quads[quadIdx].x;
    let hi = quads[quadIdx].y;

    // PackedQuad layout:
    // [0:3]   x (4 bits)
    // [4:7]   y (4 bits)
    // [8:11]  z (4 bits)
    // [12:15] width-1 (4 bits)
    // [16:19] height-1 (4 bits)
    // [20:22] face (3 bits)
    // [23:34] blockType (12 bits, spans lo/hi boundary at bit 32)
    // [35:38] damage (4 bits)
    // [39:46] AO (8 bits: 4 corners x 2 bits)

    let qx = f32(lo & 0xFu);
    let qy = f32((lo >> 4u) & 0xFu);
    let qz = f32((lo >> 8u) & 0xFu);
    let qw = f32(((lo >> 12u) & 0xFu) + 1u);  // width (stored as w-1)
    let qh = f32(((lo >> 16u) & 0xFu) + 1u);  // height (stored as h-1)
    let face = (lo >> 20u) & 0x7u;

    // blockType spans bits 23-34 (9 bits in lo from bit 23, 3 bits in hi from bit 0)
    let btLo = (lo >> 23u) & 0x1FFu;  // bits 23-31 = 9 bits
    let btHi = hi & 0x7u;              // bits 32-34 = 3 bits
    let blockType = btLo | (btHi << 9u);

    // AO packed in bits 39-46 (bits 7-14 of hi)
    let aoPacked = (hi >> 7u) & 0xFFu;
    let aoCorner = (aoPacked >> (cornerIdx * 2u)) & 3u;
    let aoVal = aoMultiplier[aoCorner];

    // Compute corner position based on face direction
    var localPos: vec3<f32>;

    // Corner layout per quad: 4 corners indexed by cornerIdx (0-3).
    // Two triangles: (0,1,2) and (1,3,2) from cornerOrder.
    // For positive faces (+X,+Y,+Z): corners sweep CCW from face normal -> correct winding.
    // For negative faces (-X,-Y,-Z): reverse the secondary axis so cross product flips.
    switch (face) {
        case 0u: { // +X face (YZ plane at x+1)
            let cy = select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u);
            let cz = select(0.0, qw, cornerIdx >= 2u);
            localPos = vec3<f32>(qx + 1.0, qy + cy, qz + cz);
        }
        case 1u: { // -X face (YZ plane at x) - reversed Z for correct winding
            let cy = select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u);
            let cz = select(qw, 0.0, cornerIdx >= 2u);
            localPos = vec3<f32>(qx, qy + cy, qz + cz);
        }
        case 2u: { // +Z face (XY plane at z+1)
            let cx = select(0.0, qw, cornerIdx >= 2u);
            let cy = select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u);
            localPos = vec3<f32>(qx + cx, qy + cy, qz + 1.0);
        }
        case 3u: { // -Z face (XY plane at z) - reversed X for correct winding
            let cx = select(qw, 0.0, cornerIdx >= 2u);
            let cy = select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u);
            localPos = vec3<f32>(qx + cx, qy + cy, qz);
        }
        case 4u: { // +Y face (XZ plane at y+1)
            let cx = select(0.0, qw, cornerIdx >= 2u);
            let cz = select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u);
            localPos = vec3<f32>(qx + cx, qy + 1.0, qz + cz);
        }
        default: { // -Y face (XZ plane at y) - reversed Z for correct winding
            let cx = select(0.0, qw, cornerIdx >= 2u);
            let cz = select(qh, 0.0, cornerIdx == 1u || cornerIdx == 3u);
            localPos = vec3<f32>(qx + cx, qy, qz + cz);
        }
    }

    // World position = section offset + local position * voxel size
    let worldPos = uniforms.sectionOffset + localPos * uniforms.voxelSize;

    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(worldPos, 1.0);
    out.worldPos = worldPos;
    out.normal = faceNormals[face];
    out.ao = aoVal;

    // Color from block type lookup table
    if (blockType < arrayLength(&blockColors)) {
        out.color = blockColors[blockType];
    } else {
        out.color = vec4<f32>(1.0, 0.0, 1.0, 1.0); // magenta = missing
    }

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Basic directional lighting (sun from upper-right)
    let sunDir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let NdotL = max(dot(in.normal, sunDir), 0.0);
    let lighting = uniforms.ambientColor + vec3<f32>(1.0, 0.95, 0.85) * NdotL;

    // Apply AO
    let litColor = in.color.rgb * lighting * in.ao;

    // Fog (distance from camera)
    let dist = length(in.worldPos - uniforms.cameraWorldPos);
    let fogFactor = exp(-dist * uniforms.fogDensity * uniforms.fogDensity);
    let finalColor = mix(uniforms.fogColor, litColor, clamp(fogFactor, 0.0, 1.0));

    return vec4<f32>(finalColor, in.color.a);
}
";

        /// <summary>
        /// Reversed-Z depth-only shader for shadow maps or depth pre-pass.
        /// Same vertex pulling, no fragment output (depth only).
        /// </summary>
        public const string DepthOnlyShader = @"
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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> quads: array<vec2<u32>>;

const cornerOrder = array<u32, 6>(0u, 1u, 2u, 1u, 3u, 2u);

@vertex
fn vs_main(@builtin(vertex_index) vertexID: u32) -> @builtin(position) vec4<f32> {
    let quadIdx = vertexID / 6u;
    let cornerIdx = cornerOrder[vertexID % 6u];

    let lo = quads[quadIdx].x;

    let qx = f32(lo & 0xFu);
    let qy = f32((lo >> 4u) & 0xFu);
    let qz = f32((lo >> 8u) & 0xFu);
    let qw = f32(((lo >> 12u) & 0xFu) + 1u);
    let qh = f32(((lo >> 16u) & 0xFu) + 1u);
    let face = (lo >> 20u) & 0x7u;

    var localPos: vec3<f32>;
    switch (face) {
        case 0u: { localPos = vec3<f32>(qx + 1.0, qy + select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u), qz + select(0.0, qw, cornerIdx >= 2u)); }
        case 1u: { localPos = vec3<f32>(qx, qy + select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u), qz + select(0.0, qw, cornerIdx >= 2u)); }
        case 2u: { localPos = vec3<f32>(qx + select(0.0, qw, cornerIdx >= 2u), qy + select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u), qz + 1.0); }
        case 3u: { localPos = vec3<f32>(qx + select(0.0, qw, cornerIdx >= 2u), qy + select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u), qz); }
        case 4u: { localPos = vec3<f32>(qx + select(0.0, qw, cornerIdx >= 2u), qy + 1.0, qz + select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u)); }
        default: { localPos = vec3<f32>(qx + select(0.0, qw, cornerIdx >= 2u), qy, qz + select(0.0, qh, cornerIdx == 1u || cornerIdx == 3u)); }
    }

    let worldPos = uniforms.sectionOffset + localPos * uniforms.voxelSize;
    return uniforms.mvp * vec4<f32>(worldPos, 1.0);
}
";
    }
}
