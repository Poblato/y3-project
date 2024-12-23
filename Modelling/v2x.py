import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import scipy

# SIM PARAMETERS
N = 10_000 # Num iterations
L = 1 # Number of links to target
NUM_CARS = 10 # Total number of cars
NUM_POINTS = 8
NUM_PLOTS = 4

target_rcs = 100

# ANTENNA PARAMETERS
Nct = 8 # Comms Transmit antennas
Ncr = 8 # Comms Receive antennas
Nst = 8 # Sensing Transmit antennas
Nsr = 8 # Sensing Receive antennas

tAntennaGain = 10**(8/10)
rAntennaGain = 10**(8/10)

# SIGNAL PARAMETERS
total_power = 16
comms_power = 4
sensing_power = 12

comms_snr_th = 10**(20/10)
radar_snr_th = 10**(20/10)

# MISC PARAMETERS
desired_pfa = 0.01

def Noise(shape, sigma):
    noise = np.random.normal(0, sigma / np.sqrt(2), shape) + 1j * np.random.normal(0, sigma / np.sqrt(2), shape)
    return noise

def RicianFading(losProportion, shape):
    v = np.sqrt((losProportion / (1 + losProportion)))
    sig = np.sqrt(1 / (2 * (1 + losProportion)))
    # not sure about the b=1
    return scipy.stats.rice.rvs(b=1, loc=v, scale=sig, size=shape)

def RayleighFading(omega, shape):
    sigma = np.sqrt(omega/2)
    return np.random.rayleigh(sigma, shape)

def P_u(u, psi):
    arr = np.zeros(u, complex)
    for i in range(u):
        arr[i] = (np.cos(constants.pi * psi) + 1j*np.sin(constants.pi * psi)) / np.sqrt(u)
    return np.matrix(arr).transpose()

# COMMS
K = 4.26
time_step = 0
c_carrier_f = 28_000_000_000
c_carrier_w = c_carrier_f / constants.c
s_carrier_f = 24_000_000_000
s_carrier_w = s_carrier_f / constants.c
relative_velocity = 0
symbol_period = 1e-9
P = 10 # N of paths simulated (1 LOS P-1 NLOS)

phi = np.zeros(P) #angles of arrival
theta = np.zeros(P) #angles of departure

noiseFigure = 4.5
noiseTemp = 290
r_Bandwidth = 500_000_000
s_noise_power = constants.k * noiseFigure * noiseTemp * r_Bandwidth
c_Bandwidth = 100_000_000
# c_noise_power = 10**((-174 + 10*np.log10(c_Bandwidth) + 10)/10)

theory = np.zeros((NUM_PLOTS, NUM_POINTS))
sim = np.zeros((NUM_PLOTS, NUM_POINTS))
dists = np.zeros((NUM_PLOTS, NUM_POINTS))

for a in range(NUM_PLOTS):
    c_noise_power = 10**((-170 + a*20 + 10*np.log10(c_Bandwidth))/10)
    R_r_max = np.pow((sensing_power * Nsr * rAntennaGain * tAntennaGain * target_rcs * s_carrier_w**2) / (np.pow(4*constants.pi, 3)*s_noise_power*radar_snr_th), 0.25)
    R_c_max = np.sqrt((comms_power * Ncr * c_carrier_w**2 * K)/(np.pow(4*constants.pi, 2) * c_noise_power*comms_snr_th*(K+1)))
    R_max = min(R_r_max, R_c_max, 150) # 150 m unless otherwise required

    dists[a] = np.arange(R_max / NUM_POINTS, R_max + 1, R_max / NUM_POINTS)

    for d in range(NUM_POINTS):
        outage_t = np.zeros(N)
        outage_count = np.zeros(L)

        link_dist = dists[a][d]
        # angle = np.random.uniform(-constants.pi/3, constants.pi/3, L)
        angle = 0

        for n in range(N):
            for l in range(L): # for each link
                # Set up sim geometry
                theta[0] = angle
                theta[1:P] = np.random.uniform(-constants.pi/2, constants.pi/2, P - 1)

                phi = -theta

                # Simulation
                H_c = np.matrix(np.zeros((Ncr, Nct), complex))
                alpha_angle = np.random.uniform(0, 2*constants.pi)
                alpha = np.cos(alpha_angle) + 1j*np.sin(alpha_angle)
                omega = 2*constants.pi*c_carrier_f*relative_velocity*symbol_period*np.sin(theta[0])/constants.c #doppler frequency shift
                H_c += np.sqrt((K * Ncr * Nct)/(K+1)) * alpha * P_u(Ncr, np.sin(phi[0])) @ P_u(Nct, np.sin(theta[0])).H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))
                for i in range(1, P):
                    alpha_angle = np.random.uniform(0, 2*constants.pi)
                    alpha = np.cos(alpha_angle) + 1j*np.sin(alpha_angle)
                    omega = 2*constants.pi*c_carrier_f*relative_velocity*symbol_period*np.sin(theta[i])/constants.c #doppler frequency shift
                    H_c += np.sqrt((Ncr * Nct)/((K+1)*P)) * alpha * P_u(Ncr, np.sin(phi[i])) @ P_u(Nct, np.sin(theta[i])).H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))

                symbol = 1
                clutterInterference = 1.2153e-11
                car_dists = 2 + np.random.random(NUM_CARS - 2) * (R_max-2) # from 2 to R_max
                z = 0.01
                jamming = np.sum((z**2*sensing_power*Nsr*tAntennaGain*rAntennaGain*target_rcs*s_carrier_w**2)/((4*constants.pi)**3*car_dists))
                radarSnr = (sensing_power * Nsr * tAntennaGain * rAntennaGain * target_rcs * s_carrier_w**2) / ((4*constants.pi)**3 * (s_noise_power + clutterInterference + jamming) * link_dist**4)

                rxSensitivity = 0.05
                angle_error = rxSensitivity / radarSnr
                f = P_u(Nct, (np.sin(phi[0]) + np.random.normal(0, angle_error)))
                receiverNoise = (np.random.normal(0, c_noise_power / np.sqrt(2), Nct) + 1j * np.random.normal(0, c_noise_power / np.sqrt(2), Nct)) @ np.identity(Nct)
                receivedSignal = np.sqrt((comms_power*c_carrier_w**2)/(Nct*(4*constants.pi*link_dist)**2))*H_c @ f * symbol + receiverNoise

                r_error = np.random.normal(0, angle_error)
                omega = P_u(Ncr, (np.sin(phi[0]) + r_error))

                # include angle error
                receivedSignal = omega.H @ receivedSignal

                # include compensation for velocity doppler shift
                complex_angle = -2*constants.pi*c_carrier_f*relative_velocity*symbol_period*(np.sin(theta[0])+r_error)*time_step/constants.c
                receivedSignal *= np.cos(complex_angle) + 1j*np.sin(complex_angle)

                comms_snr = (comms_power * c_carrier_w**2)/(Nct*(4*constants.pi*link_dist)**2) * np.abs(omega.H @ H_c @ f)**2 / c_noise_power
                if(comms_snr < comms_snr_th):
                    outage_count[l] += 1

                outage_t[n] += (comms_snr_th/comms_snr)

        theory[a][d] = np.mean(1 - np.pow(constants.e, -outage_t))
        sim[a][d] = outage_count.sum(0) / N

print("Theory = ", theory, ", Sim = ", sim)

# # RADAR
# N_d = 10 # direct path scatters
# N_p = 10 # multi-path clutter components

# H_s = np.matrix(np.zeros((Nsr, Nst), complex))
# for i in range(N_d + N_p):
#     alpha_angle = np.random.uniform(0, 2*constants.pi)
#     alpha = np.cos(alpha_angle) + 1j*np.sin(alpha_angle)
#     theta = 0.1
#     phi = 0.1
#     omega = 2*constants.pi*s_carrier_f*relative_velocity*symbol_period*np.sin(theta)/constants.c #doppler frequency shift
#     a_t = P_u(Nsr, np.sin(theta))
#     if (i < N_d):
#         H_s += alpha * P_u(Nst, np.sin(theta)) @ a_t.H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))
#     else:
#         H_s += alpha * P_u(Nst, np.sin(phi)) @ a_t.H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))

plt.figure()
for i in range(NUM_PLOTS):
    plt.plot(dists[i], sim[i], 'ko-', label="Noise = "+str(-170 + i*20) , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Link Distance (m)")
plt.ylabel("Outage")
plt.yscale('log')
plt.ylim([0, 1])
plt.xlim([0, 150])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.show()