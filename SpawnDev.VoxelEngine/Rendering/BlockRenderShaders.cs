namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WGSL for <see cref="BlockRenderPipeline"/>: textured, lit, fogged blocks drawn by vertex pulling from a
    /// <see cref="Meshing.PackedQuad"/> storage buffer.
    ///
    /// A quad is a w x h rectangle of one block type on one face. Its texture REPEATS across it (one texture tile per block), so a
    /// greedy-merged 16x16 quad of grass shows the same 16x16 tile pattern as 256 single blocks would. The textures are one 2D ATLAS
    /// (a grid of equal tiles) read with exact texel fetches: the shader wraps the coordinate inside the tile itself, so there is no
    /// sampler, no seam between tiles, and none of a texture array's layer limit (WebGPU guarantees 256 layers; a game has thousands of
    /// tiles). An ANIMATED tile is a timeline of tiles, one per tick (20 a second): the vertex shader swaps the tile by the frame's time.
    ///
    /// Faces 0..5 are the cube faces (+X -X +Z -Z +Y -Y); 6 and 7 are the two crossed planes of a plant, each a 1x1 block-high
    /// rectangle on a diagonal of the block's footprint. Every cube face is wound counter-clockwise seen from outside, and the
    /// texture is upright and unmirrored seen from outside (top and bottom faces: see the notes at each case).
    ///
    /// Bindings: group 0 = frame uniforms, block materials, per-section origins, the atlas, the animation table and timelines; group 1 = the quads.
    /// Draw (vertexCount = quads * 6, firstVertex = firstQuad * 6, firstInstance = section index): the instance index picks the
    /// section's origin, so one quad buffer holds many sections and no per-draw uniform is written.
    /// </summary>
    public static class BlockRenderShaders
    {
        /// <summary>Quad "face" code of the first plant diagonal: from (x, z) to (x+1, z+1).</summary>
        public const int FaceCrossA = 6;
        /// <summary>Quad "face" code of the second plant diagonal: from (x+1, z) to (x, z+1).</summary>
        public const int FaceCrossB = 7;

        public const string Source = @"
struct Frame {
    viewProj: mat4x4<f32>,
    cameraPos: vec4<f32>,
    sunDir: vec4<f32>,
    sunColor: vec4<f32>,   // rgb, w = sun strength
    ambient: vec4<f32>,    // rgb, w = fill light strength
    fogColor: vec4<f32>,
    fogRange: vec4<f32>,   // x = start, y = end
    timeAtlas: vec4<f32>,  // x = seconds, y = atlas tiles per row, z = tile size in pixels
};

struct BlockMaterial {
    l0: u32,
    l1: u32,
    l2: u32,
    flags: u32,
    tint: vec4<f32>,       // rgb tint, w = water alpha
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> materials: array<BlockMaterial>;
@group(0) @binding(2) var<storage, read> sectionOrigins: array<vec4<f32>>;
@group(0) @binding(3) var blockTextures: texture_2d<f32>;
// Animated tiles: animTable[tile] = (offset into timeline, length), length 0 = not animated; timeline[offset + tick % length] = the tile shown
@group(0) @binding(4) var<storage, read> animTable: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> timeline: array<u32>;
@group(1) @binding(0) var<storage, read> quads: array<vec2<u32>>;   // PackedQuad: i64 as (lo, hi)

const NO_TEXTURE: u32 = 0xFFFFu;
const FLAG_TINTED: u32 = 1u;
const FLAG_LOWER_TOP: u32 = 2u;
const FLAG_FLAT_SHADING: u32 = 4u;

// Two triangles per quad from four corners: (0,1,2) and (1,3,2). corner = a*2 + b: a runs along the quad's A axis, b along its B axis.
const cornerOrder = array<u32, 6>(0u, 1u, 2u, 1u, 3u, 2u);

// Brightness of each face before lighting: sides, top, bottom, plants. The fragment shader darkens bottoms and sides further.
const faceBase = array<f32, 8>(0.88, 0.88, 0.82, 0.82, 1.05, 0.70, 1.0, 1.0);

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) normal: vec3<f32>,
    @location(3) @interpolate(flat) vcolor: vec3<f32>,
    @location(4) @interpolate(flat) layer: u32,
    @location(5) @interpolate(flat) alpha: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexID: u32, @builtin(instance_index) section: u32) -> VOut {
    let quadIdx = vertexID / 6u;
    let corner = cornerOrder[vertexID % 6u];
    let lo = quads[quadIdx].x;
    let hi = quads[quadIdx].y;

    // PackedQuad: x 0-3, y 4-7, z 8-11, width-1 12-15, height-1 16-19, face 20-22, block type 23-34 (spans lo/hi)
    let qx = f32(lo & 0xFu);
    let qy = f32((lo >> 4u) & 0xFu);
    let qz = f32((lo >> 8u) & 0xFu);
    let qw = f32(((lo >> 12u) & 0xFu) + 1u);
    let qh = f32(((lo >> 16u) & 0xFu) + 1u);
    let face = (lo >> 20u) & 0x7u;
    let blockType = ((lo >> 23u) & 0x1FFu) | ((hi & 0x7u) << 9u);
    // Bits 47-51: the section the quad sits in, within its column (0 when the mesher did not embed it)
    let sectionY = f32((hi >> 15u) & 0x1Fu);

    let a = f32(corner >> 1u);
    let b = f32(corner & 1u);

    // Each case: the rectangle's origin corner, its A and B axes (B x A = the outward normal, so the winding is
    // counter-clockwise from outside), the texture coordinate (u to the right, v DOWN, as seen from outside) and the normal.
    var origin = vec3<f32>(qx, qy, qz);
    var axisA = vec3<f32>(0.0, 0.0, 0.0);
    var axisB = vec3<f32>(0.0, 0.0, 0.0);
    var uv = vec2<f32>(0.0, 0.0);
    var normal = vec3<f32>(0.0, 0.0, 0.0);
    switch (face) {
        case 0u: { // +X: A = +Z, B = +Y. Seen from +X the right-hand side is -Z, so u counts down from the far (+Z) edge.
            origin = origin + vec3<f32>(1.0, 0.0, 0.0);
            axisA = vec3<f32>(0.0, 0.0, qw);
            axisB = vec3<f32>(0.0, qh, 0.0);
            uv = vec2<f32>((1.0 - a) * qw, (1.0 - b) * qh);
            normal = vec3<f32>(1.0, 0.0, 0.0);
        }
        case 1u: { // -X: A = -Z, B = +Y. Seen from -X the right-hand side is +Z.
            origin = origin + vec3<f32>(0.0, 0.0, qw);
            axisA = vec3<f32>(0.0, 0.0, -qw);
            axisB = vec3<f32>(0.0, qh, 0.0);
            uv = vec2<f32>((1.0 - a) * qw, (1.0 - b) * qh);
            normal = vec3<f32>(-1.0, 0.0, 0.0);
        }
        case 2u: { // +Z: A = -X, B = +Y. Seen from +Z the right-hand side is +X.
            origin = origin + vec3<f32>(qw, 0.0, 1.0);
            axisA = vec3<f32>(-qw, 0.0, 0.0);
            axisB = vec3<f32>(0.0, qh, 0.0);
            uv = vec2<f32>((1.0 - a) * qw, (1.0 - b) * qh);
            normal = vec3<f32>(0.0, 0.0, 1.0);
        }
        case 3u: { // -Z: A = +X, B = +Y. Seen from -Z the right-hand side is -X.
            axisA = vec3<f32>(qw, 0.0, 0.0);
            axisB = vec3<f32>(0.0, qh, 0.0);
            uv = vec2<f32>((1.0 - a) * qw, (1.0 - b) * qh);
            normal = vec3<f32>(0.0, 0.0, -1.0);
        }
        case 4u: { // +Y: A = +X, B = +Z. Seen from above with -Z as screen-up: u to the right (+X), v down (+Z).
            origin = origin + vec3<f32>(0.0, 1.0, 0.0);
            axisA = vec3<f32>(qw, 0.0, 0.0);
            axisB = vec3<f32>(0.0, 0.0, qh);
            uv = vec2<f32>(a * qw, b * qh);
            normal = vec3<f32>(0.0, 1.0, 0.0);
        }
        case 5u: { // -Y: A = +X, B = -Z. Seen from below with +Z as screen-up: u to the right (+X), v down (-Z).
            origin = origin + vec3<f32>(0.0, 0.0, qh);
            axisA = vec3<f32>(qw, 0.0, 0.0);
            axisB = vec3<f32>(0.0, 0.0, -qh);
            uv = vec2<f32>(a * qw, b * qh);
            normal = vec3<f32>(0.0, -1.0, 0.0);
        }
        case 6u: { // plant diagonal from (x, z) to (x+1, z+1), one block high; drawn double-sided (the fragment flips the normal)
            axisA = vec3<f32>(1.0, 0.0, 1.0);
            axisB = vec3<f32>(0.0, 1.0, 0.0);
            uv = vec2<f32>(a, 1.0 - b);
            normal = vec3<f32>(0.7071, 0.0, -0.7071);
        }
        default: { // plant diagonal from (x+1, z) to (x, z+1)
            origin = origin + vec3<f32>(1.0, 0.0, 0.0);
            axisA = vec3<f32>(-1.0, 0.0, 1.0);
            axisB = vec3<f32>(0.0, 1.0, 0.0);
            uv = vec2<f32>(a, 1.0 - b);
            normal = vec3<f32>(0.7071, 0.0, 0.7071);
        }
    }

    var local = origin + axisA * a + axisB * b;
    local.y = local.y + sectionY * 16.0;
    let m = materials[blockType];
    if (face == 4u && (m.flags & FLAG_LOWER_TOP) != 0u) {
        local.y = local.y - 0.1; // a water surface sits a tenth below the block's top
    }
    let world = sectionOrigins[section].xyz + local;

    var out: VOut;
    out.position = frame.viewProj * vec4<f32>(world, 1.0);
    out.worldPos = world;
    out.uv = uv;
    out.normal = normal;

    // This face's texture layer (a plant uses face 0's)
    var f = face;
    if (f > 5u) { f = 0u; }
    var packed = m.l0;
    if ((f >> 1u) == 1u) { packed = m.l1; }
    if ((f >> 1u) == 2u) { packed = m.l2; }
    var tile = (packed >> ((f & 1u) * 16u)) & 0xFFFFu;
    // An animated tile plays through its timeline, one entry per tick (decided per vertex: all of a quad's vertices agree, it is flat)
    if (tile != NO_TEXTURE) {
        let anim = animTable[tile];
        if (anim.y > 0u) {
            tile = timeline[anim.x + (u32(frame.timeAtlas.x * 20.0) % anim.y)];
        }
    }
    out.layer = tile;

    // Colour before the texture: the tint (when tinted, or when there is no texture to show), else white; times the face brightness
    var base = vec3<f32>(1.0, 1.0, 1.0);
    if ((m.flags & FLAG_TINTED) != 0u || out.layer == NO_TEXTURE) {
        base = m.tint.xyz;
    }
    var shade = faceBase[face];
    if ((m.flags & FLAG_FLAT_SHADING) != 0u) {
        shade = select(1.0, 0.70, face == 5u);
    }
    out.vcolor = min(base * shade, vec3<f32>(1.0, 1.0, 1.0));
    out.alpha = m.tint.w;
    return out;
}

// One texel of an atlas tile, the tile repeating: uv is in blocks (0..w, 0..h across the quad), v counting DOWN the image
fn blockTexel(tile: u32, uv: vec2<f32>) -> vec4<f32> {
    let size = u32(frame.timeAtlas.z);
    let perRow = u32(frame.timeAtlas.y);
    let p = vec2<u32>(max(uv, vec2<f32>(0.0, 0.0)) * f32(size));
    let x = (tile % perRow) * size + (p.x % size);
    let y = (tile / perRow) * size + (p.y % size);
    return textureLoad(blockTextures, vec2<i32>(i32(x), i32(y)), 0);
}

fn applyFog(color: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32> {
    let dist = length(worldPos - frame.cameraPos.xyz);
    let range = max(frame.fogRange.y - frame.fogRange.x, 0.0001);
    let f = clamp((dist - frame.fogRange.x) / range, 0.0, 1.0);
    return mix(color, frame.fogColor.xyz, f * f);
}

// Opaque and plant pass: alpha cutout (leaf gaps, plant outlines), sun + ambient + a fixed cool fill light, fog.
@fragment
fn fs_main(in: VOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {
    var color = in.vcolor;
    if (in.layer != NO_TEXTURE) {
        let t = blockTexel(in.layer, in.uv);
        if (t.a < 0.3) { discard; }
        color = color * t.xyz;
    }

    // Plants are double-sided: light the side that is facing the camera
    var n = in.normal;
    if (!frontFacing) { n = -n; }
    if (n.y < -0.5) { color = color * 0.70; }
    if (abs(n.y) < 0.1) { color = color * 0.85; }

    let fillDir = normalize(vec3<f32>(-0.3, 0.2, -0.5));
    let sunI = max(dot(n, frame.sunDir.xyz), 0.0);
    let fillI = max(dot(n, fillDir), 0.0);
    let light = frame.ambient.xyz
        + frame.sunColor.xyz * sunI * frame.sunColor.w
        + vec3<f32>(0.55, 0.65, 0.85) * fillI * frame.ambient.w;
    return vec4<f32>(applyFog(color * light, in.worldPos), 1.0);
}

// Water pass (alpha blended, no depth write): the block's alpha, a slightly brighter ambient, the sun, fog.
@fragment
fn fs_water(in: VOut) -> @location(0) vec4<f32> {
    var color = in.vcolor;
    if (in.layer != NO_TEXTURE) {
        let t = blockTexel(in.layer, in.uv);
        color = color * t.xyz;
    }
    let sunI = max(dot(in.normal, frame.sunDir.xyz), 0.0);
    let lit = frame.ambient.xyz * 1.2 + frame.sunColor.xyz * sunI * frame.sunColor.w;
    return vec4<f32>(applyFog(color * lit, in.worldPos), in.alpha);
}
";
    }
}
