using System.Runtime.InteropServices;

namespace SpawnDev.VoxelEngine.Buffers
{
    /// <summary>
    /// CPU-side indirect draw command buffer.
    /// Manages a packed array of DrawCommands with O(1) add/remove.
    ///
    /// All visible sections' draw commands are packed contiguously in this buffer.
    /// The consuming renderer uploads the entire command array to a single GPUBuffer
    /// and issues one drawIndirect call per command (or one multi-draw if supported).
    ///
    /// Why one buffer: Chrome's D3D12 backend validates each GPUBuffer bind.
    /// N separate buffers = N validations per frame. One buffer = 1 validation.
    /// AubsCraft measured 300x improvement from this pattern (Geordi, 2026-04-13).
    ///
    /// Removal uses end-swap: the removed slot is overwritten with the last entry,
    /// and the count decrements. O(1) removal, no fragmentation, no compaction needed.
    /// The section-to-slot mapping tracks where each section's command lives.
    /// </summary>
    public class IndirectDrawBuffer
    {
        private readonly DrawCommand[] _commands;
        private readonly Dictionary<SectionCoord, int> _sectionToSlot;
        private readonly SectionCoord[] _slotToSection;
        private int _count;

        /// <summary>Number of active draw commands.</summary>
        public int Count => _count;

        /// <summary>Maximum number of draw commands this buffer can hold.</summary>
        public int Capacity { get; }

        /// <summary>Read-only span of active draw commands for GPU upload.</summary>
        public ReadOnlySpan<DrawCommand> Commands => _commands.AsSpan(0, _count);

        /// <summary>Size in bytes of the active command region (for GPU upload).</summary>
        public int ActiveSizeBytes => _count * DrawCommandSizeBytes;

        /// <summary>Size of one DrawCommand in bytes (matches WebGPU drawIndirect layout).</summary>
        public const int DrawCommandSizeBytes = 16; // 4 x uint32

        /// <summary>
        /// Create an indirect draw buffer with the given capacity.
        /// Capacity should come from DeviceCapabilities.MaxDrawCommands.
        /// </summary>
        public IndirectDrawBuffer(int capacity)
        {
            Capacity = capacity;
            _commands = new DrawCommand[capacity];
            _sectionToSlot = new Dictionary<SectionCoord, int>(capacity);
            _slotToSection = new SectionCoord[capacity];
            _count = 0;
        }

        /// <summary>
        /// Add a draw command for a section. Returns the slot index.
        /// If the section already has a command, updates it in place.
        /// Returns -1 if the buffer is full.
        /// </summary>
        public int Add(SectionCoord coord, int quadOffset, int quadCount, int sectionIndex)
        {
            if (_sectionToSlot.TryGetValue(coord, out int existingSlot))
            {
                // Update in place
                _commands[existingSlot] = DrawCommand.FromSection(quadOffset, quadCount, sectionIndex);
                return existingSlot;
            }

            if (_count >= Capacity)
                return -1;

            int slot = _count;
            _commands[slot] = DrawCommand.FromSection(quadOffset, quadCount, sectionIndex);
            _sectionToSlot[coord] = slot;
            _slotToSection[slot] = coord;
            _count++;
            return slot;
        }

        /// <summary>
        /// Remove a section's draw command. O(1) via end-swap.
        /// Returns true if the section was found and removed.
        /// </summary>
        public bool Remove(SectionCoord coord)
        {
            if (!_sectionToSlot.TryGetValue(coord, out int slot))
                return false;

            _sectionToSlot.Remove(coord);
            int lastSlot = _count - 1;

            if (slot != lastSlot)
            {
                // Swap with last entry
                _commands[slot] = _commands[lastSlot];
                var movedSection = _slotToSection[lastSlot];
                _slotToSection[slot] = movedSection;
                _sectionToSlot[movedSection] = slot;
            }

            _count--;
            return true;
        }

        /// <summary>
        /// Update an existing section's draw command.
        /// Returns false if the section is not in the buffer.
        /// </summary>
        public bool Update(SectionCoord coord, int quadOffset, int quadCount, int sectionIndex)
        {
            if (!_sectionToSlot.TryGetValue(coord, out int slot))
                return false;

            _commands[slot] = DrawCommand.FromSection(quadOffset, quadCount, sectionIndex);
            return true;
        }

        /// <summary>Check if a section has a draw command in this buffer.</summary>
        public bool Contains(SectionCoord coord) => _sectionToSlot.ContainsKey(coord);

        /// <summary>Get the slot index for a section, or -1 if not present.</summary>
        public int GetSlot(SectionCoord coord) =>
            _sectionToSlot.TryGetValue(coord, out int slot) ? slot : -1;

        /// <summary>Get the section coordinate at a given slot.</summary>
        public SectionCoord GetSectionAt(int slot) => _slotToSection[slot];

        /// <summary>Get the draw command at a given slot.</summary>
        public DrawCommand GetCommandAt(int slot) => _commands[slot];

        /// <summary>Clear all draw commands.</summary>
        public void Clear()
        {
            _sectionToSlot.Clear();
            _count = 0;
        }

        /// <summary>
        /// Get the raw command array for bulk GPU upload.
        /// Only the first Count entries are valid.
        /// The array is contiguous in memory (StructLayout.Sequential on DrawCommand)
        /// and matches WebGPU's drawIndirect buffer layout exactly.
        /// </summary>
        public DrawCommand[] GetRawArray() => _commands;

        /// <summary>
        /// Copy active commands to a byte array for GPU upload via queue.WriteBuffer.
        /// Returns the number of bytes written.
        /// </summary>
        public int CopyToBytes(byte[] destination)
        {
            int bytes = _count * DrawCommandSizeBytes;
            Buffer.BlockCopy(GetCommandBytes(), 0, destination, 0, bytes);
            return bytes;
        }

        private byte[] GetCommandBytes()
        {
            int bytes = _count * DrawCommandSizeBytes;
            var result = new byte[bytes];
            var span = MemoryMarshal.AsBytes(_commands.AsSpan(0, _count));
            span.CopyTo(result);
            return result;
        }
    }
}
