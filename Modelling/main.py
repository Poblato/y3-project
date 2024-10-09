import random as r
import matplotlib.pyplot as plt
import numpy as np
import math
from err import Err
from scipy import constants

np.random.seed(100)

N = 10_000

numInterferers = 6
minDist = 20
cellRadius = 200
pathLossExponent = 2

bsAntennaH = 10
userAntennaH = 2
carrierFrequency = 900_000_000 # 900 MHz
carrierWavelength = constants.c / carrierFrequency

g = (4*bsAntennaH*userAntennaH) / carrierWavelength

interfererPos = np.zeros((numInterferers, N))
angles = np.zeros((numInterferers, N))
powerReceived = np.zeros((numInterferers, N))
interfererDists = np.zeros((numInterferers, N))
aseSamples = np.zeros(N)

normReuseDists = np.arange(2, 11, 1) # from 2 to 10 inclusive
ases = np.zeros(normReuseDists.size)

points = np.zeros((3, normReuseDists.size))

for i in range(3):
    additionalPathLossExponent = 2 * (i + 1) # 2, 4, 6
    for x in range(normReuseDists.size):
        reuseDist = normReuseDists[x] * cellRadius

        #desired user
        userDists = minDist + (cellRadius - minDist) * np.sqrt(np.random.random(N))
        userPowers = 1 / ((userDists**pathLossExponent) * ((1 + userDists/g)**additionalPathLossExponent))

        #interferers
        interfererPos = minDist + (cellRadius - minDist) * np.sqrt(np.random.random((numInterferers, N)))
        angles = np.random.uniform(0, 2 * np.pi, (numInterferers, N))
        interfererDists = np.sqrt((reuseDist**2) + (interfererPos**2) + 2*reuseDist*interfererPos*np.sin(angles))
        powerReceived = 1 / ((interfererDists**pathLossExponent) * ((1 + interfererDists/g)**additionalPathLossExponent))

        cirs = userPowers / sum(powerReceived)
        aseSamples = (4 * np.log2(1 + cirs)) / (np.pi * (normReuseDists[x]**2) * (cellRadius**2))
        ases[x] = np.mean(aseSamples) * 1_000_000
    points[i] = ases

# cellBandwith = 100
# ase = np.sum(cirs) / (math.pi * cellBandwith * (reuseDist/2)**2)

print(ases)

plt.figure()
for i in range(3):
    plt.plot(normReuseDists, points[i])
plt.show()


# calculate received power of interferers and user

# calculate ASE