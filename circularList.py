class CircularList(list):
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[x] for x in self._rangeify(x)]
        return super().__getitem__(x % len(self))

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)
