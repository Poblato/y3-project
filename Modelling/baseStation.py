import numpy as np
from err import Err

class BaseStation:
    def __init__(self, pos, w, h):
        if (np.shape(pos) != (2, 1)):
            return Err.ValueError
        self.pos = pos
        self.w = w
        self.h = h

