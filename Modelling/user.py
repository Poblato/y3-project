from err import Err
import numpy as np

class User:
    def __init__(self, pos, behaviour):
        if (np.shape(pos) != (2, 1)):
            return Err.ValueError
        self.pos = pos
        self.behaviour = behaviour
    
    def GetDataRate(self, time):
        return self.behaviour.GetDataRate(time)