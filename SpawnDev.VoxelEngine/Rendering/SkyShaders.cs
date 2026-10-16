namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WGSL for <see cref="SkyPipeline"/>: one full-screen triangle whose every pixel works out where it looks (the inverse view-projection turns the pixel into a direction) and paints what is there:
    /// the gradient from the horizon's colour to the zenith's, a glow round the sun, the sun and the moon (a square of the game's textures, the moon in its phase), the stars at night, and a layer
    /// of clouds. The sky is drawn FIRST in the pass, with no depth, so everything else draws over it.
    /// </summary>
    public static class SkyShaders
    {
        public const string Source = @"
struct Sky {
    invViewProj: mat4x4<f32>,
    camera: vec4<f32>,       // xyz camera position
    sunDir: vec4<f32>,       // xyz toward the sun; w = how much daylight (0 night .. 1 day)
    zenith: vec4<f32>,       // rgb zenith colour; w = moon phase (0..7)
    horizon: vec4<f32>,      // rgb horizon colour; w = night (0 day .. 1 night)
    cloud: vec4<f32>,        // x height, y scroll offset (blocks, x), z weather 0..1 (rain: clouds everywhere, no sun), w star rotation angle (radians)
    cloudColor: vec4<f32>,   // rgb the colour of a cloud; w = opacity
    sunRect: vec4<f32>,      // atlas rect of the sun: x, y, w, h (pixels)
    moonRect: vec4<f32>,     // atlas rect of the first moon phase; the phases follow to the right (four a row) then down: x, y, frame w, frame h
    cloudRect: vec4<f32>,    // atlas rect of the cloud map: x, y, w, h (pixels), one pixel = 12 blocks
};

@group(0) @binding(0) var<uniform> sky: Sky;
@group(0) @binding(1) var skyTexture: texture_2d<f32>;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
    // One triangle that covers the screen
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VOut;
    out.position = vec4<f32>(p[i], 0.0, 1.0);
    out.ndc = p[i];
    return out;
}

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

// The texel of an atlas rect at (u, v) in 0..1 (nearest), alpha included
fn rectTexel(rect: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let p = vec2<i32>(i32(rect.x) + i32(clamp(floor(uv.x * rect.z), 0.0, rect.z - 1.0)), i32(rect.y) + i32(clamp(floor(uv.y * rect.w), 0.0, rect.w - 1.0)));
    return textureLoad(skyTexture, p, 0);
}

// A square image seen in direction `centre`, half a side `half` (radians): the image's texel under the pixel's direction, or alpha 0 outside
fn disc(dir: vec3<f32>, centre: vec3<f32>, half: f32, rect: vec4<f32>) -> vec4<f32> {
    let along = dot(dir, centre);
    if (along <= 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    var right = cross(vec3<f32>(0.0, 1.0, 0.0), centre);
    if (length(right) < 0.001) { right = vec3<f32>(1.0, 0.0, 0.0); }
    right = normalize(right);
    let up = cross(centre, right);
    let x = dot(dir, right) / along / half;
    let y = dot(dir, up) / along / half;
    if (abs(x) >= 1.0 || abs(y) >= 1.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    return rectTexel(rect, vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5));
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // The direction this pixel looks in: a point on its ray, from the camera
    let far = sky.invViewProj * vec4<f32>(in.ndc, 0.5, 1.0);
    let dir = normalize(far.xyz / far.w - sky.camera.xyz);

    let daylight = sky.sunDir.w;
    let night = sky.horizon.w;
    let weather = sky.cloud.z;

    // The gradient: the horizon's colour at the horizon and below, the zenith's overhead
    let elevation = clamp(dir.y, 0.0, 1.0);
    var col = mix(sky.horizon.xyz, sky.zenith.xyz, pow(elevation, 0.55));

    // A glow round the sun, strongest at dawn and dusk when the horizon is warm
    let toSun = max(dot(dir, sky.sunDir.xyz), 0.0);
    col = col + sky.horizon.xyz * pow(toSun, 12.0) * 0.35 * daylight * (1.0 - weather);

    // Stars at night, fixed to the sky: it turns with the sun
    if (night > 0.02 && dir.y > -0.02) {
        let c = cos(sky.cloud.w);
        let s = sin(sky.cloud.w);
        let turned = vec3<f32>(dir.x * c - dir.y * s, dir.x * s + dir.y * c, dir.z);
        let cell = turned * 260.0;
        let id = floor(cell);
        let h = hash3(id);
        if (h > 0.9965) {
            let at = fract(cell) - vec3<f32>(0.5, 0.5, 0.5);
            let r = length(at);
            let size = 0.18 + 0.2 * fract(h * 97.0);
            if (r < size) {
                let twinkle = 0.55 + 0.45 * fract(h * 13.0);
                col = col + vec3<f32>(1.0, 1.0, 1.0) * twinkle * night * (1.0 - weather) * clamp(dir.y * 8.0 + 0.4, 0.0, 1.0);
            }
        }
    }

    // The sun and the moon, opposite each other; the moon in its phase (frames four across, two down)
    if (weather < 0.95) {
        let sun = disc(dir, sky.sunDir.xyz, 0.17, sky.sunRect);
        col = col + sun.xyz * sun.w * (1.0 - weather);
        let phase = u32(sky.zenith.w + 0.5) % 8u;
        let moonRect = vec4<f32>(sky.moonRect.x + f32(phase % 4u) * sky.moonRect.z, sky.moonRect.y + f32(phase / 4u) * sky.moonRect.w, sky.moonRect.z, sky.moonRect.w);
        let moon = disc(dir, -sky.sunDir.xyz, 0.11, moonRect);
        col = mix(col, moon.xyz, moon.w * (1.0 - weather));
    }

    // Clouds: a flat layer at a height, one map pixel to 12 blocks, sliding along x; seen from below only
    let height = sky.cloud.x;
    if (sky.camera.y < height && dir.y > 0.012) {
        let t = (height - sky.camera.y) / dir.y;
        if (t < 4000.0) {
            let hit = sky.camera.xz + dir.xz * t;
            let px = floor((hit + vec2<f32>(sky.cloud.y, 0.0)) / 12.0);
            let size = vec2<i32>(i32(sky.cloudRect.z), i32(sky.cloudRect.w));
            let q = ((vec2<i32>(px) % size) + size) % size;
            let texel = textureLoad(skyTexture, vec2<i32>(i32(sky.cloudRect.x), i32(sky.cloudRect.y)) + q, 0);
            var density = texel.a;
            density = max(density, weather * 0.85);
            let fade = 1.0 - smoothstep(1200.0, 4000.0, t);
            col = mix(col, sky.cloudColor.xyz, density * sky.cloudColor.w * fade);
        }
    }
    return vec4<f32>(col, 1.0);
}
";
    }
}
