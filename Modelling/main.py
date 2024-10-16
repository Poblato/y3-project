import random as r
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from err import Err
from scipy import constants
import scipy

def Shadowing(power, stdDev):
    areaMeanPower = 4.3429 * np.log(power)
    # print(areaMeanPower.size())
    # dist = scipy.stats.lognorm(s = stdDev, loc=areaMeanPower.reshape(areaMeanPower.size()), scale=4.3429)
    # foo = dist.rvs(size=power.size())
    # np.reshape(foo, areaMeanPower.shape())

    rand = np.random.normal(areaMeanPower, stdDev, areaMeanPower.shape)

    # # Shadowing
    # # Area mean of the CIR
    # mu_gamma_d_minus = xi * np.log((((R * (R_u - 1)) / r_corners) ** a) * ((g + (R_u - 1) * R) / (g + r_corners)) ** b) - xi*np.log(N_I) + (variance_SI-(sigma_I**2))/(2*xi)
    # # Log-normally distributing the CIR
    # norm_gamma_d_minus_shadowing = np.random.normal(mu_gamma_d_minus, sigma_gamma_d, iterations)
    # gamma_d_minus_shadowing = np.exp(norm_gamma_d_minus_shadowing / xi)

    # # Averaging the ASE over all values of r
    # ASE_shadowing_minus[i] = (4 / (np.pi * (R_u ** 2) * (R ** 2))) * (R - R_o) * np.mean(np.log2(1+gamma_d_minus_shadowing) * p_r) * 1_000_000

    y = constants.e ** (rand / 4.3429)

    return y

def MultipathFading(power, scale):
    foo = np.random.gamma(scale, power/scale, power.shape)
    return foo

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

# shadowing constants
userSD = 4
interfererSD = 4
# multipath fading constants
m_D = 1
m_I = 1

# Interferer inactivity
activity = 0.8 # 80% chance to be active

interfererPos = np.zeros((numInterferers, N))
angles = np.zeros((numInterferers, N))
powerReceived = np.zeros((numInterferers, N))
interfererDists = np.zeros((numInterferers, N))
aseSamples = np.zeros(N)

normReuseDists = np.arange(2, 11, 1) # from 2 to 10 inclusive
ases = np.zeros(normReuseDists.size)

points = np.zeros((3, normReuseDists.size))

for i in range(3):
    # additionalPathLossExponent = 2 * (i + 1) # 2, 4, 6
    additionalPathLossExponent = 2
    m_D = 1 * (i + 1) # 1, 2, 3
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
        # userPowers = Shadowing(userPowers, userSD)
        # powerReceived = Shadowing(powerReceived, interfererSD)

        # Multipath fading
        userPowers = MultipathFading(userPowers, m_D)
        powerReceived = MultipathFading(powerReceived, m_I)

        #interferer inactivity
        # powerReceived *= np.random.random((numInterferers, N)) > activity

        cirs = userPowers / sum(powerReceived)
        aseSamples = (4 * np.log2(1 + cirs)) / (np.pi * (normReuseDists[x]**2) * (cellRadius**2))
        ases[x] = np.mean(aseSamples) * 1_000_000
    points[i] = ases

plt.figure()
for i in range(3):
    plt.plot(normReuseDists, points[i])
plt.show()