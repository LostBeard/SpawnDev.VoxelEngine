using System.Numerics;
using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.VoxelEngine.Rendering
{
    /// <summary>
    /// The light of the world near the camera, for <see cref="BlockRenderPipeline"/> to light each pixel by: for every block of the columns inside a window of columns round the camera, one byte, the
    /// sky light in the high four bits (0..15) and the block light in the low four, in the order of the block ids (x + z * 16 + y * 256, 384 high from y = -64). A column the volume has no slot for
    /// (outside the window, or not given yet) is lit as open sky with no block light, which is how the world looked before there was any light data.
    ///
    /// Columns are given slots (<see cref="Assign"/>), their bytes written to them (<see cref="Write(int, ReadOnlySpan{byte})"/>), and the window follows the camera (<see cref="SetWindow"/>: a column that
    /// leaves it loses its slot). <see cref="Update"/> sends what changed to the GPU; call it before drawing. <see cref="Enabled"/> off turns the lighting off altogether.
    /// </summary>
    public sealed class LightVolume : IDisposable
    {
        /// <summary>The bytes of a column: 16 x 384 x 16.</summary>
        public const int ColumnBytes = 16 * 384 * 16;
        /// <summary>The widest window, in columns.</summary>
        public const int MaxSide = 64;

        private GPUDevice? _device;
        private GPUQueue? _queue;
        private readonly Dictionary<(int cx, int cz), int> _assigned = new();
        private readonly Stack<int> _free = new();
        private readonly int[] _table = new int[MaxSide * MaxSide];
        private int _originX, _originZ;
        private bool _tableDirty = true, _paramsDirty = true, _enabled = true, _windowSet;
        private float _floor = 0.3f, _blockStrength = 1f;

        /// <summary>Slots: how many columns can have light at once.</summary>
        public int SlotCount { get; private set; }
        /// <summary>The window's side, in columns.</summary>
        public int Side { get; private set; }
        public bool IsReady => Data != null;
        public GPUBuffer? Data { get; private set; }
        public GPUBuffer? Slots { get; private set; }
        public GPUBuffer? Params { get; private set; }

        /// <summary>The columns that have light now.</summary>
        public int Count => _assigned.Count;

        /// <summary>Light on or off: off, every pixel is lit as it was before light existed.</summary>
        public bool Enabled
        {
            get => _enabled;
            set { if (_enabled != value) { _enabled = value; _paramsDirty = true; } }
        }

        /// <summary>How bright the world is where no sky reaches (0 = black, the game's own darkness; the default 0.3 leaves a cave readable to a spectator).</summary>
        public float DarkFloor
        {
            get => _floor;
            set { value = Math.Clamp(value, 0f, 1f); if (_floor != value) { _floor = value; _paramsDirty = true; } }
        }

        /// <summary>How strongly block light (torches, lava) shows against the sky's.</summary>
        public float BlockStrength
        {
            get => _blockStrength;
            set { value = Math.Clamp(value, 0f, 2f); if (_blockStrength != value) { _blockStrength = value; _paramsDirty = true; } }
        }

        /// <param name="slotCount">How many columns can have light at once (98,304 bytes each: 300 slots are 29 MB). A binding of more than 128 MiB is not allowed, so at most 1,300.</param>
        /// <param name="side">The window's side in columns (at most <see cref="MaxSide"/>): the window is a square of columns round the camera's column.</param>
        public void Init(GPUDevice device, GPUQueue queue, int slotCount, int side)
        {
            if (slotCount < 1 || slotCount > 1300) throw new ArgumentOutOfRangeException(nameof(slotCount));
            if (side < 1 || side > MaxSide) throw new ArgumentOutOfRangeException(nameof(side));
            _device = device;
            _queue = queue;
            SlotCount = slotCount;
            Side = side;
            Data = device.CreateBuffer(new GPUBufferDescriptor { Size = (ulong)slotCount * ColumnBytes, Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst });
            Slots = device.CreateBuffer(new GPUBufferDescriptor { Size = (ulong)MaxSide * MaxSide * 4, Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst });
            Params = device.CreateBuffer(new GPUBufferDescriptor { Size = 32, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
            for (int i = slotCount - 1; i >= 0; i--) _free.Push(i);
            System.Array.Fill(_table, -1);
            _tableDirty = _paramsDirty = true;
        }

        /// <summary>The window round a column (the camera's): columns that fall outside it lose their slot and are returned in <paramref name="evicted"/>.</summary>
        public void SetWindow(int centerCx, int centerCz, List<(int cx, int cz)>? evicted = null)
        {
            int half = Side / 2;
            int ox = centerCx - half, oz = centerCz - half;
            if (_windowSet && ox == _originX && oz == _originZ) return;
            _windowSet = true;
            _originX = ox; _originZ = oz;
            foreach (var key in _assigned.Keys.ToList())
                if (!InWindow(key.cx, key.cz))
                {
                    _free.Push(_assigned[key]);
                    _assigned.Remove(key);
                    evicted?.Add(key);
                }
            RebuildTable();
        }

        public bool InWindow(int cx, int cz) => cx >= _originX && cz >= _originZ && cx < _originX + Side && cz < _originZ + Side;

        public bool Has(int cx, int cz) => _assigned.ContainsKey((cx, cz));

        /// <summary>The slot of a column (it gets one if it has none and one is free); -1 when it is outside the window or every slot is in use.</summary>
        public int Assign(int cx, int cz)
        {
            if (_assigned.TryGetValue((cx, cz), out int slot)) return slot;
            if (!InWindow(cx, cz) || _free.Count == 0) return -1;
            slot = _free.Pop();
            _assigned[(cx, cz)] = slot;
            _table[(cz - _originZ) * Side + (cx - _originX)] = slot;
            _tableDirty = true;
            return slot;
        }

        /// <summary>A column's slot taken back (the column is gone, or its light is going to be replaced and must not show half old, half new: <see cref="Assign"/> it again).</summary>
        public void Release(int cx, int cz)
        {
            if (!_assigned.Remove((cx, cz), out int slot)) return;
            _free.Push(slot);
            if (InWindow(cx, cz)) _table[(cz - _originZ) * Side + (cx - _originX)] = -1;
            _tableDirty = true;
        }

        /// <summary>The light of a column into its slot: <see cref="ColumnBytes"/> bytes.</summary>
        public void Write(int slot, ReadOnlySpan<byte> light)
        {
            if (Data == null || _queue == null) throw new InvalidOperationException("Call Init first");
            if (slot < 0 || slot >= SlotCount) throw new ArgumentOutOfRangeException(nameof(slot));
            if (light.Length != ColumnBytes) throw new ArgumentException($"A column's light is {ColumnBytes} bytes, not {light.Length}", nameof(light));
            _queue.WriteBuffer(Data, (ulong)slot * ColumnBytes, light.ToArray());
        }

        /// <summary>The light of a column into its slot, straight from a fetched array (it does not pass through .NET).</summary>
        public void Write(int slot, Uint8Array light)
        {
            if (Data == null || _queue == null) throw new InvalidOperationException("Call Init first");
            if (slot < 0 || slot >= SlotCount) throw new ArgumentOutOfRangeException(nameof(slot));
            if (light.Length != ColumnBytes) throw new ArgumentException($"A column's light is {ColumnBytes} bytes, not {light.Length}", nameof(light));
            _queue.WriteBuffer(Data, (ulong)slot * ColumnBytes, light);
        }

        private void RebuildTable()
        {
            System.Array.Fill(_table, -1);
            foreach (var ((cx, cz), slot) in _assigned) _table[(cz - _originZ) * Side + (cx - _originX)] = slot;
            _tableDirty = true;
        }

        /// <summary>Send what changed (the slot table, the settings) to the GPU. Call before drawing.</summary>
        public void Update()
        {
            if (_queue == null) return;
            if (_tableDirty)
            {
                _tableDirty = false;
                _paramsDirty = true;
                _queue.WriteBuffer(Slots!, 0, MemoryMarshal.AsBytes(_table.AsSpan(0, Side * Side)).ToArray());
            }
            if (_paramsDirty)
            {
                _paramsDirty = false;
                var floats = new float[] { _originX, _originZ, Side, _enabled ? 1f : 0f, _floor, _blockStrength, 0f, 0f };
                _queue.WriteBuffer(Params!, 0, MemoryMarshal.AsBytes(floats.AsSpan()).ToArray());
            }
        }

        public void Dispose()
        {
            Data?.Destroy(); Data?.Dispose(); Data = null;
            Slots?.Destroy(); Slots?.Dispose(); Slots = null;
            Params?.Destroy(); Params?.Dispose(); Params = null;
        }
    }
}
