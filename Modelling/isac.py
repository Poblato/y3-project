import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import scipy

# SIM PARAMETERS
N = 10_000

# CHANNEL PARAMETERS
# https://erdc-library.erdc.dren.mil/server/api/core/bitstreams/8eca351f-51ec-46ae-8b30-60c7f3b2465e/content
# for 972 MHz in urban environments
thermalNoise = 1e-14
downSlowFading = 0.6
upSlowFading = 0.6
# Proportion of signal power that is transmitted via LOS path
downLosProportion = 0.6
upLosProportion = 0.6

# ANTENNA PARAMETERS
# Transmit antennas
Nt = 4
# Receive antennas
Nr = 4
# Uplink User antennas
Nu = 4
# Downlink User antennas
Nd = 4

def ThermalNoise():
    return thermalNoise

def RicianFading(power, losProportion, shape):
    v = np.sqrt((losProportion / (1 + losProportion)) * power)
    sig = np.sqrt(power / (2 * (1 + losProportion)))
    # not sure about the b=1
    return scipy.stats.rice.rvs(b=1, loc=v, scale=sig, size=shape)

def DownlinkChannel(power, shape):
    # Fast fading
    power = RicianFading(power, downLosProportion, shape)
    return power * downSlowFading + ThermalNoise()

def UplinkChannel(power, shape):
    # Fast fading
    power = RicianFading(power, upLosProportion, shape)
    return power * upSlowFading + ThermalNoise()

# Covers the full round trip channel for radar signals
def RadarChannel(power, shape):
    attenuationFactor = 0.5
    return power * attenuationFactor + ThermalNoise()

def SelfInterferenceChannel(power, shape):
    return 0

downlinkPower = 0
uplinkPower = 0
radarPower = 0

downlinkReceived = DownlinkChannel(downlinkPower)

uplinkReceived = UplinkChannel(uplinkPower) + RadarChannel(radarPower)

# For now assume redidual SI is 0

# Focus on maths for decoding radar and received signals

