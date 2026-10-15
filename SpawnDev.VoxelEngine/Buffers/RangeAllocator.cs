namespace SpawnDev.VoxelEngine.Buffers
{
    /// <summary>
    /// Hands out ranges of a fixed number of elements (best fit) and takes them back, merging neighbours, so many variable-sized
    /// allocations can share one big buffer. Pure bookkeeping: it touches no GPU memory, so the same logic is unit-tested on the CPU.
    /// Not thread-safe.
    /// </summary>
    public sealed class RangeAllocator
    {
        // Free ranges, sorted by start, never adjacent (they are merged on Free)
        private readonly List<(int Start, int Length)> _free = new();

        public RangeAllocator(int capacity)
        {
            if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
            Capacity = capacity;
            _free.Add((0, capacity));
            FreeTotal = capacity;
        }

        /// <summary>Elements the allocator manages.</summary>
        public int Capacity { get; }

        /// <summary>Elements not handed out.</summary>
        public int FreeTotal { get; private set; }

        /// <summary>Elements handed out.</summary>
        public int Used => Capacity - FreeTotal;

        /// <summary>The longest range <see cref="Allocate"/> can still give (fragmentation shows as FreeTotal > LargestFree).</summary>
        public int LargestFree
        {
            get
            {
                int best = 0;
                foreach (var f in _free) if (f.Length > best) best = f.Length;
                return best;
            }
        }

        /// <summary>Free ranges: 1 when nothing is fragmented.</summary>
        public int FreeRangeCount => _free.Count;

        /// <summary>The start of a free range of <paramref name="length"/> elements (the smallest range that fits), or -1 when none is long enough.</summary>
        public int Allocate(int length)
        {
            if (length < 1) throw new ArgumentOutOfRangeException(nameof(length));
            int best = -1;
            for (int i = 0; i < _free.Count; i++)
            {
                if (_free[i].Length < length) continue;
                if (best < 0 || _free[i].Length < _free[best].Length) best = i;
            }
            if (best < 0) return -1;

            var (start, len) = _free[best];
            if (len == length) _free.RemoveAt(best);
            else _free[best] = (start + length, len - length);
            FreeTotal -= length;
            return start;
        }

        /// <summary>Give back a range from <see cref="Allocate"/>. Throws when it overlaps something already free (a double free).</summary>
        public void Free(int start, int length)
        {
            if (length < 1) throw new ArgumentOutOfRangeException(nameof(length));
            if (start < 0 || start + length > Capacity) throw new ArgumentOutOfRangeException(nameof(start), "outside the allocator");

            // First free range that starts after this one
            int i = 0;
            while (i < _free.Count && _free[i].Start < start) i++;
            bool mergePrev = i > 0 && _free[i - 1].Start + _free[i - 1].Length >= start;
            bool mergeNext = i < _free.Count && start + length >= _free[i].Start;
            if (i > 0 && _free[i - 1].Start + _free[i - 1].Length > start)
                throw new InvalidOperationException($"Range [{start}, {start + length}) overlaps a free range: freed twice");
            if (i < _free.Count && start + length > _free[i].Start)
                throw new InvalidOperationException($"Range [{start}, {start + length}) overlaps a free range: freed twice");

            if (mergePrev && mergeNext)
            {
                _free[i - 1] = (_free[i - 1].Start, _free[i].Start + _free[i].Length - _free[i - 1].Start);
                _free.RemoveAt(i);
            }
            else if (mergePrev) _free[i - 1] = (_free[i - 1].Start, start + length - _free[i - 1].Start);
            else if (mergeNext) _free[i] = (start, _free[i].Start + _free[i].Length - start);
            else _free.Insert(i, (start, length));
            FreeTotal += length;
        }
    }
}
