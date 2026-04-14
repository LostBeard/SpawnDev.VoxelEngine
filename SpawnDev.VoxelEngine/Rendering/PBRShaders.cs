namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// WGSL PBR fragment shader for Lost Spawns HD rendering.
    ///
    /// Per-fragment lighting using:
    /// - Diffuse texture (from texture array, per block type + face + variant)
    /// - Normal map (tangent-space normals per face)
    /// - Roughness map (controls specular highlight size)
    /// - Up to 16 dynamic point/spot/directional lights
    /// - Time-of-day sun + ambient
    /// - Per-vertex AO
    /// - Fog with distance falloff
    /// - Block damage crack overlay
    ///
    /// The vertex shader is the same vertex pulling shader from VertexPullShaders.
    /// Only the fragment shader changes for PBR mode.
    /// </summary>
    public static class PBRShaders
    {
        /// <summary>
        /// PBR fragment shader.
        ///
        /// Additional bindings beyond the solid color shader:
        ///   @group(1) @binding(0) blockTextures: texture_2d_array
        ///   @group(1) @binding(1) texSampler: sampler
        ///   @group(1) @binding(2) crackTextures: texture_2d_array (5 stages)
        ///   @group(1) @binding(3) lights: storage buffer (16 * LightData)
        ///   @group(1) @binding(4) lightParams: uniform (lightCount, sunDir, sunColor, sunIntensity)
        /// </summary>
        public const string PBRFragmentShader = @"
struct LightData {
    positionAndType: vec4<f32>,   // xyz = position/direction, w = type (0=point, 1=spot, 2=directional)
    colorAndIntensity: vec4<f32>, // rgb = color, a = intensity
    directionAndCone: vec4<f32>,  // xyz = spot direction, w = outer cone cos
    rangeAndFalloff: vec4<f32>,   // x = radius, y = falloff, z = inner cone cos, w = reserved
};

struct LightParams {
    lightCount: u32,
    sunDirection: vec3<f32>,
    sunColor: vec3<f32>,
    sunIntensity: f32,
    ambientColor: vec3<f32>,
    timeOfDay: f32,
};

struct PBRInput {
    @location(0) color: vec4<f32>,        // fallback color from vertex pull
    @location(1) worldPos: vec3<f32>,
    @location(2) normal: vec3<f32>,       // face normal
    @location(3) ao: f32,
    @location(4) uv: vec2<f32>,           // texture UV
    @location(5) blockType: u32,
    @location(6) face: u32,
    @location(7) damage: u32,
};

@group(1) @binding(0) var blockTextures: texture_2d_array<f32>;
@group(1) @binding(1) var texSampler: sampler;
@group(1) @binding(2) var crackTextures: texture_2d_array<f32>;
@group(1) @binding(3) var<storage, read> lights: array<LightData>;
@group(1) @binding(4) var<uniform> lightParams: LightParams;

// Texture variant from world position (deterministic hash)
fn getVariant(worldPos: vec3<f32>) -> u32 {
    let ix = i32(floor(worldPos.x));
    let iy = i32(floor(worldPos.y));
    let iz = i32(floor(worldPos.z));
    let hash = u32(ix * 73856093 ^ iy * 19349663 ^ iz * 83492791);
    return hash % 4u;
}

// Per-face tangent space for normal mapping
fn getFaceTangent(face: u32) -> mat3x3<f32> {
    switch (face) {
        case 0u: { return mat3x3<f32>(vec3(0,0,-1), vec3(0,1,0), vec3(1,0,0)); }   // +X
        case 1u: { return mat3x3<f32>(vec3(0,0,1), vec3(0,1,0), vec3(-1,0,0)); }    // -X
        case 2u: { return mat3x3<f32>(vec3(1,0,0), vec3(0,1,0), vec3(0,0,1)); }     // +Z
        case 3u: { return mat3x3<f32>(vec3(-1,0,0), vec3(0,1,0), vec3(0,0,-1)); }   // -Z
        case 4u: { return mat3x3<f32>(vec3(1,0,0), vec3(0,0,-1), vec3(0,1,0)); }    // +Y
        default: { return mat3x3<f32>(vec3(1,0,0), vec3(0,0,1), vec3(0,-1,0)); }    // -Y
    }
}

@fragment
fn fs_pbr(in: PBRInput) -> @location(0) vec4<f32> {
    let variant = getVariant(in.worldPos);
    let layersPerVariant = 3u; // diffuse + normal + roughness
    let baseLayer = in.blockType * 4u * 6u * layersPerVariant + variant * 6u * layersPerVariant + in.face * layersPerVariant;

    // Sample PBR textures
    let diffuse = textureSample(blockTextures, texSampler, in.uv, i32(baseLayer));
    let normalSample = textureSample(blockTextures, texSampler, in.uv, i32(baseLayer + 1u));
    let roughness = textureSample(blockTextures, texSampler, in.uv, i32(baseLayer + 2u)).r;

    // Normal mapping: tangent space -> world space
    let tangentNormal = normalSample.xyz * 2.0 - 1.0;
    let TBN = getFaceTangent(in.face);
    let worldNormal = normalize(TBN * tangentNormal);

    // Accumulate lighting
    var totalLight = lightParams.ambientColor;

    // Sun (directional)
    if (lightParams.sunIntensity > 0.0) {
        let NdotL = max(dot(worldNormal, -lightParams.sunDirection), 0.0);
        totalLight += lightParams.sunColor * lightParams.sunIntensity * NdotL;
    }

    // Dynamic lights
    for (var i = 0u; i < lightParams.lightCount && i < 16u; i++) {
        let light = lights[i];
        let lightType = u32(light.positionAndType.w);
        let lightColor = light.colorAndIntensity.xyz;
        let intensity = light.colorAndIntensity.w;

        var lightDir: vec3<f32>;
        var attenuation: f32;

        if (lightType == 2u) {
            // Directional
            lightDir = -light.positionAndType.xyz;
            attenuation = 1.0;
        } else {
            // Point or spot
            let lightPos = light.positionAndType.xyz;
            let toLight = lightPos - in.worldPos;
            let dist = length(toLight);
            let radius = light.rangeAndFalloff.x;
            let falloff = light.rangeAndFalloff.y;

            if (dist > radius) { continue; }

            lightDir = toLight / dist;
            attenuation = 1.0 / (1.0 + dist * dist * falloff);

            // Spot cone
            if (lightType == 1u) {
                let spotDir = light.directionAndCone.xyz;
                let outerCos = light.directionAndCone.w;
                let innerCos = light.rangeAndFalloff.z;
                let spotDot = dot(-lightDir, spotDir);
                if (spotDot < outerCos) { continue; }
                attenuation *= clamp((spotDot - outerCos) / (innerCos - outerCos), 0.0, 1.0);
            }
        }

        // Lambertian diffuse
        let NdotL = max(dot(worldNormal, lightDir), 0.0);

        // Simple Blinn-Phong specular (roughness controls exponent)
        let halfDir = normalize(lightDir + normalize(-in.worldPos)); // approximate view dir
        let spec = pow(max(dot(worldNormal, halfDir), 0.0), mix(4.0, 128.0, 1.0 - roughness));

        totalLight += lightColor * intensity * (NdotL + spec * 0.3) * attenuation;
    }

    // Apply lighting and AO
    var finalColor = diffuse.rgb * totalLight * in.ao;

    // Damage crack overlay
    if (in.damage > 0u) {
        let crackStage = min(in.damage / 3u, 4u);
        let crackAlpha = f32(in.damage) / 15.0;
        let crackUV = fract(in.worldPos.xz);
        let crackSample = textureSample(crackTextures, texSampler, crackUV, i32(crackStage));
        finalColor = mix(finalColor, crackSample.rgb * 0.15, crackSample.a * crackAlpha);
    }

    return vec4<f32>(finalColor, diffuse.a);
}
";
    }
}
