import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import scipy

# SIM PARAMETERS
N = 10_000 # Num iterations
L = 10 # Time domain samples (size of Delay-Doppler operator matrix)

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

def ThermalNoise(shape):
    return np.random.normal(0, thermalNoise, shape)

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
    power = RicianFading(power, upLosProportion, shape)
    return power * upSlowFading * downSlowFading + ThermalNoise()

def SelfInterferenceChannel(power, shape):
    return 0

downlinkPower = 0
uplinkPower = 0
radarPower = 0

# positions relative to the BS
downlinkPos = (0, 0)
uplinkPos = (0, 0)

# downlinkReceived = DownlinkChannel(downlinkPower)

# uplinkReceived = UplinkChannel(uplinkPower) + RadarChannel(radarPower)

# sensingReceived = RadarChannel()





# Parameters
P_t = 20
P_r = 10        # Power allocated to radar, equivalent to magnitude of s_r
P_c = P_t - P_r # Power allocated to comms

variance = 1
gamma_t = 1

num_plots = 3
max_gamma = 11

gamma_pfa = np.zeros((num_plots, max_gamma))
gamma_pd = np.zeros((num_plots, max_gamma))

for x in range(num_plots): # 0 to 2
    d_snr = -10 + 10*x # in dB
    d_snr = 10**(d_snr / 10) # linear
    sigma = np.sqrt(variance)
    sigma_r = sigma / np.sqrt(2)
    sigma_i = sigma / np.sqrt(2)
    gamma_d = np.sqrt((sigma**2) * d_snr / P_r)

    for y in range(max_gamma): # 0 to 10
        gamma = y
        for n in range(N): # N iterations
            s_r = np.random.randn(L,1) # Radar waveform s(t)
            s_r = s_r / np.linalg.norm(s_r) * np.sqrt(P_r)

            surveillance_noise = np.random.normal(0, sigma_r, (L, 1)) + 1j * np.random.normal(0, sigma_i, (L, 1))
            direct_noise = np.random.normal(0, sigma_r, (L, 1)) + 1j * np.random.normal(0, sigma_i, (L, 1))

            # received power when there is a target
            x_s_p = np.matrix(surveillance_noise + s_r * gamma_t)
            # received power when there is no target
            x_s_n = np.matrix(surveillance_noise)

            x_d = np.matrix(gamma_d * s_r + direct_noise)

            thresh = sigma**2 * gamma

            b = (x_d @ x_d.H + thresh * np.identity(L)) / (np.linalg.norm(x_d)**2 + thresh)
            temp_p = x_s_p.H @ b @ x_s_p
            temp_n = x_s_n.H @ b @ x_s_n

            gamma_pfa[x][y] += temp_n[0,0] >= thresh
            gamma_pd[x][y] += temp_p[0,0] >= thresh
        gamma_pfa[x][y] /= N
        gamma_pd[x][y] /= N

rate_count = 17
rate_range = np.arange(0.2, 3.6, 0.2)

P_t = 10
gamma_t = 1
gamma_c = 1

rate_pfa = np.zeros((num_plots, rate_count))
rate_pd = np.zeros((num_plots, rate_count))

for x in range(num_plots):
    d_snr = -10 + 10*x # in dB
    d_snr = 10**(d_snr / 10) # linear
    
    for y in range(rate_count):
        sigma = np.sqrt(variance)
        sigma_r = sigma / np.sqrt(2)
        sigma_i = sigma / np.sqrt(2)

        # gamma_d = np.sqrt((sigma**2) * d_snr / P_r)
        # gamma_c = gamma_d**2 / sigma**2
        # rate = np.log2(1 + P_c*gamma_c)
        rate = 0.2 + y*0.2
        P_c = (2**rate - 1) / gamma_c
        P_r = P_t - P_c
        gamma_d = np.sqrt(gamma_c * sigma**2)

        gamma = 5

        for n in range(N):
            s_r = np.random.randn(L,1) # Radar waveform s(t)
            s_r = s_r / np.linalg.norm(s_r) * np.sqrt(P_r)

            surveillance_noise = np.random.normal(0, sigma_r, (L, 1)) + 1j * np.random.normal(0, sigma_i, (L, 1))
            direct_noise = np.random.normal(0, sigma_r, (L, 1)) + 1j * np.random.normal(0, sigma_i, (L, 1))

            # received power when there is a target
            x_s_p = np.matrix(surveillance_noise + s_r * gamma_t)
            # received power when there is no target
            x_s_n = np.matrix(surveillance_noise)

            x_d = np.matrix(gamma_d * s_r + direct_noise)

            thresh = sigma**2 * gamma

            b = (x_d @ x_d.H + thresh * np.identity(L)) / (np.linalg.norm(x_d)**2 + thresh)
            temp_p = x_s_p.H @ b @ x_s_p
            temp_n = x_s_n.H @ b @ x_s_n

            rate_pfa[x][y] += temp_n[0,0] >= thresh
            rate_pd[x][y] += temp_p[0,0] >= thresh
        rate_pfa[x][y] /= N
        rate_pd[x][y] /= N


plt.figure()
for i in range(num_plots):
    plt.plot(range(max_gamma), gamma_pfa[i], 'ko-', label="D-SNR=" + str(-10 + i * 10) + "dB" , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Gamma")
plt.ylabel("PFA")
plt.ylim([0, 1])
plt.xlim([0, 10])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.figure()
for i in range(num_plots):
    plt.plot(range(max_gamma), gamma_pd[i], 'ko-', label="D-SNR=" + str(-10 + i * 10) + "dB" , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Gamma")
plt.ylabel("PD")
plt.ylim([0, 1])
plt.xlim([0, 10])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.figure()
for i in range(num_plots):
    plt.plot(rate_range, rate_pfa[i], 'ko-', label="D-SNR=" + str(-10 + i * 10) + "dB" , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Rate")
plt.ylabel("PFA")
plt.ylim([0, 1])
plt.xlim([0, 4])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.figure()
for i in range(num_plots):
    plt.plot(rate_range, rate_pd[i], 'ko-', label="D-SNR=" + str(-10 + i * 10) + "dB" , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Rate")
plt.ylabel("PD")
plt.ylim([0, 1])
plt.xlim([0, 4])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.show()