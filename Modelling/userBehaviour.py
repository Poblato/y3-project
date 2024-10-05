from err import Err
import numpy as np
from abc import ABC
import random as r

# Abstract base class, has no implementation
class UserBehaviour(ABC):
    def GetDataRate(self, time):
        pass

class ConstantBehaviour(UserBehaviour):
    def __init__(self, dataRate):
        if (dataRate < 0):
            Err.ShowError(Err.NonFatalRangeError)
            dataRate = 0
        self.dataRate = dataRate

    def GetDataRate(self, time):
        return {self.dataRate, Err.OK}
    
class GaussianBehaviour(UserBehaviour):
    def __init__(self, meanDataRate, stdDev):
        if (meanDataRate < 0):
            Err.ShowError(Err.NonFatalRangeError)
            meanDataRate = 0
        self.meanDataRate = meanDataRate
        if (stdDev < 0):
            Err.ShowError(Err.NonFatalRangeError)
            stdDev = 0
        self.stdDev = stdDev
    # FIXME is there a more deterministic way of generating this based on current sim time?
    def GetDataRate(self, time):
        rate = r.gauss(self.meanDataRate, self.stdDev)
        if (rate < 0):
            rate = 0
        return {rate, Err.OK}
    
class BurstBehaviour(UserBehaviour):
    # period is the percentage of time in which the user is receiving data
    def __init__(self, meanDataRate, period):
        if (meanDataRate < 0):
            Err.ShowError(Err.NonFatalRangeError)
            meanDataRate = 0
        self.meanDataRate = meanDataRate
        if (period < 0):
            Err.ShowError(Err.NonFatalRangeError)
            period = 0
        if (period > 1):
            Err.ShowError(Err.NonFatalRangeError)
            period = 1
        self.period = period
        self.peakDataRate = self.meanDataRate / self.period

    def GetDataRate(self, time):
        if (r.random() <= self.period):
            rate = self.peakDataRate
        else:
            rate = 0
        return {rate, Err.OK}
            