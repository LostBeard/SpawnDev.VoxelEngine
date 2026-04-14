namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// Block damage crack overlay rendering.
    ///
    /// Damage level 0 = pristine (no overlay).
    /// Damage level 1-14 = increasing crack severity.
    /// Damage level 15 = about to break (full crack pattern).
    ///
    /// The overlay is a separate texture (crack atlas with 5 stages)
    /// alpha-blended on top of the block's base texture in the fragment shader.
    ///
    /// DayZ-style: damage persists until repaired. Not Minecraft break-and-vanish.
    ///
    /// The damage level is packed in PackedBlock bits [12:15] and carried through
    /// to PackedQuad bits [35:38]. The fragment shader extracts it and samples
    /// the appropriate crack texture stage.
    /// </summary>
    public static class DamageOverlay
    {
        /// <summary>Number of crack texture stages in the atlas.</summary>
        public const int CrackStages = 5;

        /// <summary>
        /// Map a 4-bit damage level (0-15) to a crack stage index (0-4).
        /// Stage 0 = light hairline cracks, stage 4 = heavy damage with chunks missing.
        /// Damage 0 = no overlay (return -1).
        /// </summary>
        public static int GetCrackStage(int damageLevel)
        {
            if (damageLevel <= 0) return -1; // no damage
            if (damageLevel <= 3) return 0;  // light
            if (damageLevel <= 6) return 1;  // moderate
            if (damageLevel <= 9) return 2;  // heavy
            if (damageLevel <= 12) return 3; // severe
            return 4;                         // critical (about to break)
        }

        /// <summary>
        /// Get the alpha intensity for the crack overlay based on damage level.
        /// Higher damage = more opaque cracks.
        /// </summary>
        public static float GetCrackAlpha(int damageLevel)
        {
            if (damageLevel <= 0) return 0f;
            return Math.Clamp(damageLevel / 15f, 0.1f, 1f);
        }

        /// <summary>
        /// WGSL fragment shader snippet for damage overlay.
        /// Reads damage from the packed quad data and blends crack texture.
        ///
        /// Requires:
        ///   @group(1) @binding(0) crackTexture: texture_2d_array (5 layers)
        ///   @group(1) @binding(1) crackSampler: sampler
        ///   damage: u32 (extracted from packed quad bits 35-38)
        ///   uv: vec2<f32> (tiled across block face using fract(worldPos))
        /// </summary>
        public const string WGSLSnippet = @"
// Damage overlay
if (damage > 0u) {
    let crackStage = select(
        select(
            select(
                select(0u, 1u, damage > 3u),
                2u, damage > 6u),
            3u, damage > 9u),
        4u, damage > 12u);

    let crackAlpha = f32(damage) / 15.0;
    let crackUV = fract(worldPos.xz); // tile across face
    let crackColor = textureSample(crackTexture, crackSampler, crackUV, i32(crackStage));
    finalColor = mix(finalColor, crackColor.rgb * 0.2, crackColor.a * crackAlpha);
}
";
    }
}
