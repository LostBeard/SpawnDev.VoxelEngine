namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WGSL for <see cref="WeatherPipeline"/>: an instance a column of the window, each a vertical quad facing the camera from the column's ground (at least a little below the camera) to a little above the
    /// camera, wearing the rain or snow texture, which slides down it with the clock (rain fast, snow slowly and swaying).
    /// </summary>
    public static class WeatherShaders
    {
        public const string Source = @"
struct Weather {
    viewProj: mat4x4<f32>,
    camera: vec4<f32>,       // xyz camera; w clock in ticks
    params: vec4<f32>,       // x strength, y brightness, z window side, w reach (blocks above and below the camera)
    window: vec4<f32>,       // x, z of the window's first column
    rainRect: vec4<f32>,
    snowRect: vec4<f32>,
};

@group(0) @binding(0) var<uniform> w: Weather;
@group(0) @binding(1) var weatherTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> cells: array<u32>;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) kind: u32,
    @location(2) alpha: f32,
};

fn hash2(a: i32, b: i32) -> u32 {
    var h = u32(a) * 374761393u + u32(b) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var out: VOut;
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);   // out of view unless this column has something to draw
    out.uv = vec2<f32>(0.0, 0.0);
    out.kind = 0u;
    out.alpha = 0.0;

    let n = u32(w.params.z);
    let cell = cells[ii];
    let kind = cell >> 16u;
    if (kind == 0u) { return out; }
    let ground = f32(cell & 0xFFFFu) - 64.0;
    let reach = w.params.w;
    let x = i32(w.window.x) + i32(ii % n);
    let z = i32(w.window.y) + i32(ii / n);
    let cx = f32(x) + 0.5;
    let cz = f32(z) + 0.5;
    let dx = w.camera.x - cx;
    let dz = w.camera.z - cz;
    let dist = sqrt(dx * dx + dz * dz);
    let halfSide = 0.5 * f32(n);
    if (dist >= halfSide) { return out; }

    let yTop = floor(w.camera.y) + reach;
    let yBottom = max(ground, floor(w.camera.y) - reach);
    if (yTop <= yBottom) { return out; }

    // The quad's two sides: across the line from the column to the camera
    var t = vec2<f32>(0.5, 0.0);
    if (dist > 0.001) { t = vec2<f32>(-dz, dx) / dist * 0.5; }
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let c = corners[vi];
    let y = mix(yBottom, yTop, c.y);
    let world = vec3<f32>(cx + t.x * c.x, y, cz + t.y * c.x);

    let h = hash2(x, z);
    let r1 = f32(h & 255u) / 255.0;
    let r2 = f32((h >> 8u) & 255u) / 255.0;
    let ticks = w.camera.w;
    var u = c.x * 0.5 + 0.5 + r1;
    var scroll = 0.0;
    if (kind == 1u) {
        // Rain falls about nine blocks a second, each column at its own speed
        scroll = ticks * (3.0 + r2) / 32.0;
    } else {
        // Snow falls slowly and sways
        scroll = ticks * 0.016 * (0.8 + 0.4 * r2);
        u = u + sin(ticks * 0.03 + r2 * 6.28318) * 0.25;
    }
    out.position = w.viewProj * vec4<f32>(world, 1.0);
    out.uv = vec2<f32>(u, -y * 0.25 - scroll);
    out.kind = kind;
    out.alpha = clamp(1.0 - dist / halfSide, 0.0, 1.0) * w.params.x;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    var rect = w.rainRect;
    if (in.kind == 2u) { rect = w.snowRect; }
    let tw = max(rect.z, 1.0);
    let th = max(rect.w, 1.0);
    let ux = fract(in.uv.x) * tw;
    let vy = fract(in.uv.y) * th;
    let p = vec2<i32>(i32(rect.x) + i32(min(floor(ux), tw - 1.0)), i32(rect.y) + i32(min(floor(vy), th - 1.0)));
    let texel = textureLoad(weatherTexture, p, 0);
    let a = texel.a * in.alpha;
    if (a < 0.02) { discard; }
    return vec4<f32>(texel.rgb * w.params.y, a);
}
";
    }
}
