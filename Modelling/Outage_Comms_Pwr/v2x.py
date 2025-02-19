import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import scipy

# SIM PARAMETERS
N = 150_000 # Num iterations
L = 3 # Number of links to target
NUM_CARS = 6 # Total number of interferers
NUM_POINTS = 14
NUM_PLOTS = 1

target_rcs = 100
reuse_dist = 30

# ANTENNA PARAMETERS
Nct = 8 # Comms Transmit antennas
Ncr = 8 # Comms Receive antennas
Nst = 8 # Sensing Transmit antennas
Nsr = 8 # Sensing Receive antennas

tAntennaGain = 10**(8/10) # 8 dB
rAntennaGain = 10**(8/10) # 8 dB

# SIGNAL PARAMETERS
total_power = 16
comms_power = 4
sensing_power = 12

comms_snr_th = 10**(20/10) # 20 dB
radar_snr_th = 10**(20/10) # 20 dB

def P_u(u, psi):
    arr = np.zeros(u, complex)
    for i in range(u):
        arr[i] = (np.cos(constants.pi * psi * i) + 1j*np.sin(constants.pi * psi * i)) / (2*np.sqrt(u))
    return np.matrix(arr).transpose()

# COMMS
K = 4.26 # Rician Factor
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


# DIST VARIATION
# R_r_max = np.pow((sensing_power * Nsr * rAntennaGain * tAntennaGain * target_rcs * s_carrier_w**2) / (np.pow(4*constants.pi, 3)*s_noise_power*radar_snr_th), 0.25)
# R_c_max = np.sqrt((comms_power * Ncr * c_carrier_w**2 * K)/(np.pow(4*constants.pi, 2) * c_noise_power*comms_snr_th*(K+1)))
# R_max = min(R_r_max, R_c_max, 150) # 150 m unless otherwise required
# R_min = 30
R_max = 150
# dists = np.arange(R_min, R_max, (R_max - R_min) / NUM_POINTS)

# Power variation
link_dist = 150
link_angles = np.random.uniform(-constants.pi/3, constants.pi/3, L)
sensing_powers = np.arange(1, total_power - 1, (total_power - 2) / NUM_POINTS)
comms_powers = total_power - sensing_powers
# Comms power optimisation
# Y_i = np.zeros(L, complex)
# for i in range(L):
#     Y_i[i] = P_u(Ncr, np.sin(link_angles[i])).H @ P_u(Ncr, np.sin(theta[i])) @ P_u(Nct, np.sin(theta[i])).H @ P_u(Nct, np.sin(-link_angles[i]))
# comms_power_1 = (total_power - (sensing_power * L)*link_dist**2) / (link_dist**2 + Y_i[0]*np.sum(link_dist**2 / Y_i[1:L]))
# if l == 0:
#     comms_power = comms_power_1 
# else:
#     comms_power = (comms_power_1 * Y_i[1] * link_dist**2) / (link_dist**2 * Y_i[l])
# print(comms_powers)

theory = np.zeros((NUM_PLOTS, NUM_POINTS))
sim_outage = np.zeros((NUM_PLOTS, NUM_POINTS))
sim_snr = np.zeros((NUM_PLOTS, NUM_POINTS))
sim_pd = np.zeros((NUM_PLOTS, NUM_POINTS))

c_noise_power = 10**(-120/10)

for a in range(NUM_PLOTS):
    print("Plot ", a+1, "of ", NUM_PLOTS)

    for d in range(NUM_POINTS):
        print("Point ", d+1, "of ", NUM_POINTS)
        outage_t = np.zeros(L, complex)
        radar_outage_count = 0
        outage_count = 0
        snr_total = 0

        sensing_power = sensing_powers[d]
        comms_power = comms_powers[d]

        for n in range(N):
            outage_iteration = False

            # Generate geometry
            link_dists = np.zeros(L)
            link_angles = np.zeros(L)
            # link_angles = np.random.uniform(-constants.pi/3, constants.pi/3, L)
            for i in range(L):
               link_dists[i] = link_dist
            
            for l in range(L): # for each link
                # Set up sim geometry
                theta[0] = link_angles[l]
                theta[1:P] = np.random.uniform(-constants.pi/2, constants.pi/2, P - 1)
                phi = -theta

                # Simulation
                H_c = np.matrix(np.zeros((Ncr, Nct), complex))
                alpha_angles = np.random.uniform(0, 2*constants.pi, P)
                alpha = np.cos(alpha_angles[0]) + 1j*np.sin(alpha_angles[0])
                omega = 2*constants.pi*c_carrier_f*relative_velocity*symbol_period*np.sin(theta[0])/constants.c #doppler frequency shift
                H_c += np.sqrt((K * Ncr * Nct)/(K+1)) * alpha * P_u(Ncr, np.sin(phi[0])) @ P_u(Nct, np.sin(theta[0])).H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))
                for i in range(1, P):
                    alpha = np.cos(alpha_angles[i]) + 1j*np.sin(alpha_angles[i])
                    omega = 2*constants.pi*c_carrier_f*relative_velocity*symbol_period*np.sin(theta[i])/constants.c #doppler frequency shift
                    H_c += np.sqrt((Ncr * Nct)/((K+1)*P)) * alpha * P_u(Ncr, np.sin(phi[i])) @ P_u(Nct, np.sin(theta[i])).H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))

                symbol = 1
                clutterInterference = 1.2153e-11
                car_dists = np.zeros(NUM_CARS) + reuse_dist # All dists at reuse_dist (worst case)
                z = 0.01
                jamming = np.sum((z**2*sensing_power*Nsr*tAntennaGain*rAntennaGain*target_rcs*s_carrier_w**2)/((4*constants.pi)**3*car_dists))
                radarSnr = (sensing_power * Nsr * tAntennaGain * rAntennaGain * target_rcs * s_carrier_w**2) / ((4*constants.pi)**3 * (s_noise_power + clutterInterference + jamming) * link_dists[l]**4)

                if (radarSnr < radar_snr_th):
                    radar_outage_count += 1

                rxSensitivity = 0.05
                angle_error = rxSensitivity / radarSnr
                f = P_u(Nct, (np.sin(theta[0]) + np.random.normal(0, angle_error)))
                receiverNoise = (np.random.normal(0, c_noise_power / np.sqrt(2), Nct) + 1j * np.random.normal(0, c_noise_power / np.sqrt(2), Nct)) @ np.identity(Nct)

                receivedSignal = np.sqrt((comms_power*c_carrier_w**2)/(Nct*(4*constants.pi*link_dists[l])**2))*H_c @ f * symbol + receiverNoise

                # include angle error
                r_error = np.random.normal(0, angle_error)
                omega = P_u(Ncr, (np.sin(theta[0]) + r_error))
                receivedSignal = omega.H @ receivedSignal

                # include compensation for velocity doppler shift
                complex_angle = -2*constants.pi*c_carrier_f*relative_velocity*symbol_period*(np.sin(theta[0])+r_error)*time_step/constants.c
                receivedSignal *= np.cos(complex_angle) + 1j*np.sin(complex_angle)

                comms_snr = (comms_power * c_carrier_w**2)/(Nct*(4*constants.pi*link_dists[l])**2) * np.abs(omega.H @ H_c @ f)**2 / c_noise_power
                snr_total += comms_snr
                if(comms_snr < comms_snr_th):
                    if (not outage_iteration):
                        outage_count += 1
                        outage_iteration = True

                # exp_omega = P_u(Ncr, (np.sin(theta[0]) -  + np.sqrt(2)*angle_error))
                # exp_f = P_u(Nct, (np.sin(theta[0]) + np.sqrt(2)*angle_error))
                # exp_comms_snr = (comms_power * c_carrier_w**2)/(Nct*(4*constants.pi*link_dists[l])**2) * np.abs(exp_omega.H @ H_c @ exp_f)**2 / c_noise_power
                # outage_t[l] += exp_comms_snr
                # outage_t[l] += comms_snr

        # outage_t = N / outage_t
        # temp = -(comms_snr_th * outage_t.sum())
        # if (not(temp.imag == 0)):
        #     print("Error: theoretical snr not real")
        # theory[a][d] = (1 - pow(np.e, temp.real))
        sim_outage[a][d] = outage_count / N
        sim_snr[a][d] = snr_total / N
        sim_pd[a][d] = 1 - (radar_outage_count / N)

# Convert to dB
sim_snr = 20*np.log10(sim_snr)
sim_rate = c_Bandwidth * np.log2(1 + sim_snr)
print(sim_outage)
print(sim_snr)
# print("Theory = ", theory, " Sim = ", sim)

# avg_diff = np.mean(np.abs(theory - sim) / sim)
# print("Avg diff = ", avg_diff)

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

colours = ["blue", "red", "yellow", "green"]

plt.figure()
for i in range(NUM_PLOTS):
    plt.plot(sensing_powers, sim_outage[i], 'ko-', linewidth=0.5, markerfacecolor="none", markersize=6)
    # plt.plot(theory[i], comms_powers, 'ko--', label="Theory = "+str(-170 + i*10) , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.ylabel("Radar Power (W)")
plt.ylabel("Outage")
plt.yscale('log')
# plt.ylim([0, 10])
plt.xlim([0, 16])
# plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.figure()
for i in range(NUM_PLOTS):
    plt.plot(sensing_powers, sim_rate[i], 'ko-', linewidth=0.5, markerfacecolor="none", markersize=6)
plt.ylabel("Radar Power (W)")
plt.xlabel("Rate (bits/sec)")
plt.yscale('log')
# plt.ylim([0, 10])
plt.xlim([0, 16])
# plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.figure()
for i in range(NUM_PLOTS):
    plt.plot(sensing_powers, sim_pd[i], 'ko-', linewidth=0.5, markerfacecolor="none", markersize=6)
plt.ylabel("Radar Power (W)")
plt.xlabel("Rate (bits/sec)")
plt.yscale('log')
# plt.ylim([0, 10])
plt.xlim([0, 16])
# plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.show()