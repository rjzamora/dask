class FrameExpr:
    @property
    def meta(self):
        raise NotImplementedError

    @property
    def dsk(self):
        raise NotImplementedError

    @property
    def args(self):
        raise NotImplementedError

    @property
    def dependencies(self):
        raise NotImplementedError


class LeafFrame(FrameExpr):
    def __init__(self, dsk, name, meta, divisions):
        self._dsk = dsk
        self._name = name
        self._meta = meta
        self._divisions = divisions

    @property
    def meta(self):
        return self._meta

    @property
    def dsk(self):
        return self._dsk

    @property
    def args(self):
        return ()

    @property
    def dependencies(self):
        return ()
