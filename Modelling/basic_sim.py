import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import scipy

def Shadowing(power, stdDev):
    areaMeanPower = 10/np.log(10) * np.log(power)
    rand = np.random.normal(areaMeanPower, stdDev, areaMeanPower.shape)
    return np.exp(rand / (10/np.log(10)))

def MultipathFading(power, m):
    return np.random.gamma(m, power/m, power.shape)

np.random.seed(100)

N = 100_000
numPlots = 3

# https://erdc-library.erdc.dren.mil/server/api/core/bitstreams/8eca351f-51ec-46ae-8b30-60c7f3b2465e/content
# for 972 MHz in urban environments
thermalNoise = 1e-14

numInterferers = 6
minDist = 20
cellRadius = 200
pathLossExponent = 2

bsAntennaH = 10
userAntennaH = 2
carrierFrequency = 900_000_000 # 900 MHz
carrierWavelength = constants.c / carrierFrequency

g = (4*bsAntennaH*userAntennaH) / carrierWavelength

# shadowing constants
userSD = 6
interfererSD = 6
# multipath fading constants
m_D = 1
m_I = 1

interfererPos = np.zeros((numInterferers, N))
angles = np.zeros((numInterferers, N))
powerReceived = np.zeros((numInterferers, N))
interfererDists = np.zeros((numInterferers, N))
aseSamples = np.zeros(N)
rateSamples = np.zeros(N)

normReuseDists = np.arange(2, 11, 1) # from 2 to 10 inclusive
ases = np.zeros((numPlots, normReuseDists.size))
rates = np.zeros((numPlots, normReuseDists.size))
outages = np.zeros((numPlots, normReuseDists.size))
outageThreshold = 1.2

for i in range(numPlots):
    # additionalPathLossExponent = 2 * (i + 1) # 2, 4, 6
    additionalPathLossExponent = 2
    m_D = 2 * i - 1 # 1, 3
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

        # Shadowing
        if (i > 0):
            userPowers = Shadowing(userPowers, userSD)
            powerReceived = Shadowing(powerReceived, interfererSD)

        # Multipath fading
        if (i > 0):
            userPowers = MultipathFading(userPowers, m_D)
            powerReceived = MultipathFading(powerReceived, m_I)

        #interferer inactivity
        interfererActivity = 0.75
        activityArray = np.zeros((numInterferers, N)) + interfererActivity
        powerReceived *= np.random.random((numInterferers, N)) < activityArray

        cirs = userPowers / (thermalNoise + sum(powerReceived))
        rateSamples = np.log2(1 + cirs)
        aseSamples = 4 * rateSamples / (np.pi * (normReuseDists[x]**2) * (cellRadius**2))
        ases[i][x] = np.mean(aseSamples) * 1_000_000
        rates[i][x] = np.mean(rateSamples) * 1_000_000
        outages[i][x] = np.sum(cirs < outageThreshold) * 100 / N

# ASE
plt.figure()
plt.plot(normReuseDists, ases[0], 'ko-', label="No multipath/shadowing", linewidth=0.5, markerfacecolor="none", markersize=6)
for i in range(1, numPlots):
    plt.plot(normReuseDists, ases[i], 'k-', label="m_d=" + str(2 * i - 1) , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Normalised Reuse Distance Ru")
plt.ylabel("ASE [Bits/Sec/Hz/Km^2]")
plt.xlim([2, 10])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

# Rate
plt.figure()
plt.plot(normReuseDists, rates[0], 'ko-', label="No multipath/shadowing", linewidth=0.5, markerfacecolor="none", markersize=6)
for i in range(1, numPlots):
    plt.plot(normReuseDists, rates[i], 'k-', label="m_d=" + str(2 * i - 1) , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Normalised Reuse Distance Ru")
plt.ylabel("Rate [Bits/Sec/Hz]")
plt.xlim([2, 10])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

# Outage
plt.figure()
plt.plot(normReuseDists, outages[0], 'ko-', label="No multipath/shadowing", linewidth=0.5, markerfacecolor="none", markersize=6)
for i in range(1, numPlots):
    plt.plot(normReuseDists, outages[i], 'k-', label="m_d=" + str(2 * i - 1) , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Normalised Reuse Distance Ru")
plt.ylabel("Outage [%]")
plt.xlim([2, 10])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.show()