namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WGSL for <see cref="EntityRenderPipeline"/>: a box expanded from one <see cref="EntityPartInstance"/> (36 vertices an instance, 6 faces of 2
    /// triangles), textured by Minecraft's box layout, lit and fogged like the blocks.
    ///
    /// Model space is Minecraft's: y points DOWN, so the "top" face of a box is at its minimum y. The texture of a box with origin (u, v) and
    /// size w x h x d is the cross unfolding, in pixels: top (u+d, v) w x d; bottom (u+d+w, v) w x d; then a row from (u, v+d): the box's
    /// right side (x minimum) d x h, front (z minimum) w x h, left side (x maximum) d x h, back (z maximum) w x h. A MIRRORED box swaps the
    /// left and right regions and runs every u backwards. Overlay inflation moves the faces outward but not the texture lookup.
    /// </summary>
    public static class EntityRenderShaders
    {
        public const string Source = @"
struct Frame {
    viewProj: mat4x4<f32>,
    cameraPos: vec4<f32>,
    sunDir: vec4<f32>,
    sunColor: vec4<f32>,   // rgb, w = sun strength
    ambient: vec4<f32>,    // rgb, w = fill light strength
    fogColor: vec4<f32>,
    fogRange: vec4<f32>,   // x = start, y = end
    timeAtlas: vec4<f32>,
};

struct Part {
    model: mat4x4<f32>,
    originInflate: vec4<f32>,
    sizeFlags: vec4<f32>,
    uv: vec4<f32>,
    tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var entityTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> parts: array<Part>;

// Two triangles from four corners 0..3 = (0,0) (1,0) (1,1) (0,1)
const cornerOrder = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) normal: vec3<f32>,
    @location(3) @interpolate(flat) tint: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexID: u32, @builtin(instance_index) instance: u32) -> VOut {
    let part = parts[instance];
    let face = vertexID / 6u;
    let c = cornerOrder[vertexID % 6u];
    let s = vec2<f32>(select(0.0, 1.0, c == 1u || c == 2u), select(0.0, 1.0, c >= 2u));

    let w = part.sizeFlags.x;
    let h = part.sizeFlags.y;
    let d = part.sizeFlags.z;
    let mirror = part.sizeFlags.w > 0.5;
    let u0 = part.uv.x;
    let v0 = part.uv.y;

    // The corner's place along each axis, 0 = the box's minimum, 1 = its maximum
    var sx = 0.0;
    var sy = 0.0;
    var sz = 0.0;
    var normal = vec3<f32>(0.0, 0.0, 0.0);
    switch (face) {
        case 0u: { sx = s.x; sy = s.y; sz = 0.0; normal = vec3<f32>(0.0, 0.0, -1.0); }   // front (z minimum)
        case 1u: { sx = s.x; sy = s.y; sz = 1.0; normal = vec3<f32>(0.0, 0.0, 1.0); }    // back (z maximum)
        case 2u: { sx = 0.0; sy = s.y; sz = s.x; normal = vec3<f32>(-1.0, 0.0, 0.0); }   // right side (x minimum)
        case 3u: { sx = 1.0; sy = s.y; sz = s.x; normal = vec3<f32>(1.0, 0.0, 0.0); }    // left side (x maximum)
        case 4u: { sx = s.x; sy = 0.0; sz = s.y; normal = vec3<f32>(0.0, -1.0, 0.0); }   // top (y minimum: y points down)
        default: { sx = s.x; sy = 1.0; sz = s.y; normal = vec3<f32>(0.0, 1.0, 0.0); }    // bottom
    }

    // Geometry, with the overlay inflation; the texture below uses the plain box
    let inf = part.originInflate.w;
    let o = part.originInflate.xyz;
    let p = vec3<f32>(
        mix(o.x - inf, o.x + w + inf, sx),
        mix(o.y - inf, o.y + h + inf, sy),
        mix(o.z - inf, o.z + d + inf, sz));

    let ex = sx * w;
    let ey = sy * h;
    let ez = sz * d;
    let mx = select(ex, w - ex, mirror);   // u along x, run backwards when mirrored
    var uv = vec2<f32>(0.0, 0.0);
    switch (face) {
        case 0u: { uv = vec2<f32>(u0 + d + mx, v0 + d + ey); }
        case 1u: { uv = vec2<f32>(u0 + 2.0 * d + w + select(w - ex, ex, mirror), v0 + d + ey); }
        case 2u: { uv = vec2<f32>(select(u0, u0 + d + w, mirror) + select(d - ez, ez, mirror), v0 + d + ey); }
        case 3u: { uv = vec2<f32>(select(u0 + d + w, u0, mirror) + select(ez, d - ez, mirror), v0 + d + ey); }
        case 4u: { uv = vec2<f32>(u0 + d + mx, v0 + d - ez); }
        default: { uv = vec2<f32>(u0 + d + w + mx, v0 + ez); }
    }

    let world = (part.model * vec4<f32>(p, 1.0)).xyz;
    var out: VOut;
    out.position = frame.viewProj * vec4<f32>(world, 1.0);
    out.worldPos = world;
    out.uv = uv;
    out.normal = normalize((part.model * vec4<f32>(normal, 0.0)).xyz);
    out.tint = part.tint.xyz;
    return out;
}

fn applyFog(color: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32> {
    let dist = length(worldPos - frame.cameraPos.xyz);
    let range = max(frame.fogRange.y - frame.fogRange.x, 0.0001);
    let f = clamp((dist - frame.fogRange.x) / range, 0.0, 1.0);
    return mix(color, frame.fogColor.xyz, f * f);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(entityTexture));
    let texel = clamp(vec2<i32>(floor(in.uv)), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    let t = textureLoad(entityTexture, texel, 0);
    if (t.a < 0.1) { discard; }
    var color = t.xyz * in.tint;

    let n = in.normal;
    if (n.y < -0.5) { color = color * 0.70; }   // world space: y is up here, the model's y points down
    if (abs(n.y) < 0.1) { color = color * 0.85; }
    let fillDir = normalize(vec3<f32>(-0.3, 0.2, -0.5));
    let sunI = max(dot(n, frame.sunDir.xyz), 0.0);
    let fillI = max(dot(n, fillDir), 0.0);
    let light = frame.ambient.xyz
        + frame.sunColor.xyz * sunI * frame.sunColor.w
        + vec3<f32>(0.55, 0.65, 0.85) * fillI * frame.ambient.w;
    return vec4<f32>(applyFog(color * light, in.worldPos), 1.0);
}
";
    }
}
