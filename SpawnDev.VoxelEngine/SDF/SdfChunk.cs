using System.Runtime.InteropServices;

namespace SpawnDev.VoxelEngine.SDF
{
    /// <summary>
    /// Signed Distance Field chunk - stores SDF values for smooth terrain.
    ///
    /// Each voxel holds a signed distance to the nearest surface:
    ///   Positive = solid (inside terrain)
    ///   Negative = air (outside terrain)
    ///   Zero = surface (isosurface)
    ///
    /// Uses 16-bit fixed-point for precision at low memory cost:
    ///   32x32x32 chunk = 64 KB (vs 128 KB at float32)
    ///   Uniform chunks (all solid/air) store no data (flagged as uniform)
    ///
    /// SDF enables: smooth surfaces, caves, overhangs, boolean carving,
    /// and terrain deformation - all impossible with block-only storage.
    /// </summary>
    public class SdfChunk
    {
        /// <summary>Default chunk size (voxels per axis).</summary>
        public const int DefaultSize = 32;

        /// <summary>
        /// Fixed-point scale: SDF values stored as short, divide by this to get float distance.
        /// Range: -128.0 to +127.996 world units (at scale 256).
        /// Precision: ~0.004 world units per step.
        /// </summary>
        public const float FixedPointScale = 256f;

        /// <summary>Chunk size in voxels per axis (typically 32).</summary>
        public int Size { get; }

        /// <summary>Total voxel count (Size^3).</summary>
        public int VoxelCount => Size * Size * Size;

        /// <summary>
        /// SDF values in 16-bit fixed-point. Null if chunk is uniform.
        /// Layout: [x + z * Size + y * Size * Size] (matching block storage order).
        /// </summary>
        public short[]? Values { get; set; }

        /// <summary>
        /// Uniform value when all voxels are the same distance.
        /// Short.MaxValue = fully solid. Short.MinValue = fully air.
        /// Only used when Values is null.
        /// </summary>
        public short UniformValue { get; set; }

        /// <summary>Whether this chunk is uniform (all same SDF value).</summary>
        public bool IsUniform => Values == null;

        /// <summary>Whether this chunk is fully solid (no surface to mesh).</summary>
        public bool IsFullySolid => IsUniform && UniformValue > 0;

        /// <summary>Whether this chunk is fully air (no surface to mesh).</summary>
        public bool IsFullyAir => IsUniform && UniformValue <= 0;

        /// <summary>Whether this chunk contains a surface (needs meshing).</summary>
        public bool HasSurface => !IsUniform || UniformValue == 0;

        public SdfChunk(int size = DefaultSize)
        {
            Size = size;
            UniformValue = short.MinValue; // default: fully air
        }

        /// <summary>Get SDF value at position. Returns uniform value if chunk is uniform.</summary>
        public short GetValue(int x, int y, int z)
        {
            if (Values == null) return UniformValue;
            return Values[x + z * Size + y * Size * Size];
        }

        /// <summary>Get SDF value as float distance.</summary>
        public float GetDistance(int x, int y, int z)
        {
            return GetValue(x, y, z) / FixedPointScale;
        }

        /// <summary>Set SDF value at position. Allocates storage if chunk was uniform.</summary>
        public void SetValue(int x, int y, int z, short value)
        {
            if (Values == null)
            {
                if (value == UniformValue) return; // no change needed
                // Allocate and fill with uniform value
                Values = new short[VoxelCount];
                if (UniformValue != 0)
                    System.Array.Fill(Values, UniformValue);
            }
            Values[x + z * Size + y * Size * Size] = value;
        }

        /// <summary>Set SDF value from float distance.</summary>
        public void SetDistance(int x, int y, int z, float distance)
        {
            short value = (short)Math.Clamp(distance * FixedPointScale, short.MinValue, short.MaxValue);
            SetValue(x, y, z, value);
        }

        /// <summary>
        /// Convert float distance to fixed-point short.
        /// </summary>
        public static short ToFixedPoint(float distance)
        {
            return (short)Math.Clamp(distance * FixedPointScale, short.MinValue, short.MaxValue);
        }

        /// <summary>
        /// Convert fixed-point short to float distance.
        /// </summary>
        public static float ToFloat(short fixedPoint)
        {
            return fixedPoint / FixedPointScale;
        }

        /// <summary>
        /// Try to compact this chunk to uniform if all values are the same.
        /// Returns true if compacted.
        /// </summary>
        public bool TryCompact()
        {
            if (Values == null) return true; // already uniform

            short first = Values[0];
            for (int i = 1; i < Values.Length; i++)
            {
                if (Values[i] != first) return false;
            }

            UniformValue = first;
            Values = null;
            return true;
        }

        /// <summary>
        /// Create a fully solid chunk (all positive SDF = inside terrain).
        /// </summary>
        public static SdfChunk CreateSolid(int size = DefaultSize)
        {
            return new SdfChunk(size) { UniformValue = short.MaxValue };
        }

        /// <summary>
        /// Create a fully air chunk (all negative SDF = outside terrain).
        /// </summary>
        public static SdfChunk CreateAir(int size = DefaultSize)
        {
            return new SdfChunk(size) { UniformValue = short.MinValue };
        }
    }
}
