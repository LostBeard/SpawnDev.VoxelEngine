namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WGSL for <see cref="ParticleRenderPipeline"/>: a camera-facing quad (6 vertices) for each of an emitter's <see cref="ParticleRenderPipeline.MaxPerEmitter"/> particle slots, placed
    /// and faded entirely from the kind's numbers and the frame clock.
    ///
    /// A slot's particle is born again every <c>life</c> seconds; its phase is spread over the emitter's particles and offset by a hash of the emitter's block, so a row of campfires does not pulse
    /// together. Each birth takes its own random numbers (hashed from the block, the slot and the birth's number): where it starts, how it drifts. The camera's right and up come from the
    /// view-projection itself (the gradients of clip x, y and w), so any eye of a headset - off-axis frusta included - gets billboards that face it, with no extra uniform.
    /// </summary>
    public static class ParticleRenderShaders
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
    timeAtlas: vec4<f32>,  // x = seconds
};

struct Kind {
    counts: vec4<f32>,     // x particles, y life, z flags, w frames
    sprite: vec4<f32>,     // xy first frame px, zw frame size px
    motion: vec4<f32>,     // xyz velocity, w gravity
    scatter: vec4<f32>,    // x spread, y drift speed, z wobble, w wobble speed
    fade: vec4<f32>,       // x size0, y size1, z alpha0, w alpha1
    tint: vec4<f32>,
    origin: vec4<f32>,     // xyz from the block corner, w vertical scatter
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var particleTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> emitters: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> kinds: array<Kind>;

const MaxPerEmitter = 16u;
const cornerOrder = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);

fn hash(n: u32) -> u32 {
    var x = n;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn rnd(n: u32) -> f32 { return f32(hash(n) & 0xffffffu) / 16777216.0; }

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) rect: vec4<f32>,
    @location(3) @interpolate(flat) tint: vec4<f32>,   // rgb, w = light
    @location(4) alpha: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexID: u32, @builtin(instance_index) instance: u32) -> VOut {
    var out: VOut;
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);   // outside the clip volume: nothing drawn unless this slot is alive
    out.worldPos = vec3<f32>(0.0, 0.0, 0.0);
    out.uv = vec2<f32>(0.0, 0.0);
    out.rect = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.tint = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.alpha = 0.0;

    let e = instance / MaxPerEmitter;
    let k = instance % MaxPerEmitter;
    let em = emitters[e];
    let kind = kinds[u32(em.w)];
    let count = u32(kind.counts.x);
    if (k >= count) { return out; }

    let life = max(kind.counts.y, 0.01);
    let block = vec3<i32>(i32(em.x), i32(em.y), i32(em.z));
    let blockHash = hash((u32(block.x + 65536) * 73856093u) ^ (u32(block.y + 1024) * 19349663u) ^ (u32(block.z + 65536) * 83492791u));

    // Where in its life this slot's particle is: a new one is born each time the phase wraps
    let phase = frame.timeAtlas.x / life + f32(k) / f32(count) + rnd(blockHash);
    let birth = floor(phase);
    let age = phase - birth;
    let seed = hash(blockHash ^ (k * 2654435761u) ^ (u32(max(birth, 0.0)) * 40503u));
    let t = age * life;
    let r1 = rnd(seed);
    let r2 = rnd(seed + 1u);
    let r3 = rnd(seed + 2u);
    let r4 = rnd(seed + 3u);
    let r5 = rnd(seed + 4u);

    var p = vec3<f32>(em.xyz) + kind.origin.xyz
        + vec3<f32>((r1 - 0.5) * 2.0 * kind.scatter.x, (r4 - 0.5) * 2.0 * kind.origin.w, (r2 - 0.5) * 2.0 * kind.scatter.x);
    let drift = vec2<f32>(r3 - 0.5, r5 - 0.5) * 2.0 * kind.scatter.y;
    p = p + vec3<f32>(kind.motion.x + drift.x, kind.motion.y, kind.motion.z + drift.y) * t
          + vec3<f32>(0.0, 0.5 * kind.motion.w * t * t, 0.0);
    let ph = frame.timeAtlas.x * kind.scatter.w + r1 * 6.2831853;
    p = p + vec3<f32>(sin(ph), 0.0, cos(ph * 0.7 + 1.3)) * kind.scatter.z * age;

    // The camera's axes from the view-projection: the gradients of clip x, y and w with respect to the world position
    let m = frame.viewProj;
    let gx = vec3<f32>(m[0][0], m[1][0], m[2][0]);
    let gy = vec3<f32>(m[0][1], m[1][1], m[2][1]);
    let gw = vec3<f32>(m[0][3], m[1][3], m[2][3]);
    let fwd = normalize(gw);
    let right = normalize(gx - fwd * dot(gx, fwd));
    let up = normalize(gy - fwd * dot(gy, fwd));

    let c = cornerOrder[vertexID % 6u];
    let s = vec2<f32>(select(0.0, 1.0, c == 1u || c == 2u), select(0.0, 1.0, c >= 2u));
    let size = mix(kind.fade.x, kind.fade.y, age);
    let world = p + right * ((s.x - 0.5) * size) + up * ((0.5 - s.y) * size);

    let frames = max(u32(kind.counts.w), 1u);
    let frameIndex = min(u32(age * f32(frames)), frames - 1u);
    let origin = vec2<f32>(kind.sprite.x + f32(frameIndex) * kind.sprite.z, kind.sprite.y);

    // Daylight dims a particle that does not glow; night keeps a quarter
    let light = select(mix(0.25, 1.0, clamp(frame.sunColor.w, 0.0, 1.0)), 1.0, kind.counts.z > 0.5);

    out.position = frame.viewProj * vec4<f32>(world, 1.0);
    out.worldPos = world;
    out.uv = origin + s * kind.sprite.zw;
    out.rect = vec4<f32>(origin, origin + kind.sprite.zw);
    out.tint = vec4<f32>(kind.tint.xyz, light);
    out.alpha = mix(kind.fade.z, kind.fade.w, age);
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
    let lo = vec2<i32>(i32(in.rect.x), i32(in.rect.y));
    let hi = vec2<i32>(i32(in.rect.z) - 1, i32(in.rect.w) - 1);
    let texel = clamp(vec2<i32>(floor(in.uv)), lo, hi);
    let t = textureLoad(particleTexture, texel, 0);
    let a = t.a * in.alpha;
    if (a < 0.02) { discard; }
    let color = t.xyz * in.tint.xyz * in.tint.w;
    return vec4<f32>(applyFog(color, in.worldPos), a);
}
";
    }
}
