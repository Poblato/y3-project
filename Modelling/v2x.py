import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import scipy

# SIM PARAMETERS
N = 50_000_000 # Num iterations
L = 1 # Number of links to target
NUM_CARS = 2 # Total number of cars for radar bounces
NUM_INTERFERERS = 2 # Total number of interfering cars
NUM_POINTS = 15
NUM_PLOTS = 1

target_rcs = 100
vru_rcs = 10
reuse_dist = 200
link_dist = 150

# ANTENNA PARAMETERS
Nct = 8 # Comms Transmit antennas
Ncr = 8 # Comms Receive antennas
Nst = 8 # Sensing Transmit antennas
Nsr = 8 # Sensing Receive antennas

tAntennaGain = 10**(8/10) # 8 dB
rAntennaGain = 10**(8/10) # 8 dB

# SIGNAL PARAMETERS
total_power = 16
comms_power = 12
sensing_power = 4

comms_snr_th = 10**(5/10) # 5 dB
radar_snr_th = 10**(5/10) # 5 dB

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
noiseTemp = 300
r_Bandwidth = 500_000_000
s_noise_power = constants.k * noiseFigure * noiseTemp * r_Bandwidth
c_Bandwidth = 100_000_000
# c_noise_power = 10**((-174 + 10*np.log10(c_Bandwidth) + 10)/10)


# DIST VARIATION
# R_r_max = np.pow((sensing_power * Nsr * rAntennaGain * tAntennaGain * target_rcs * s_carrier_w**2) / (np.pow(4*constants.pi, 3)*s_noise_power*radar_snr_th), 0.25)
# R_c_max = np.sqrt((comms_power * Ncr * c_carrier_w**2 * K)/(np.pow(4*constants.pi, 2) * c_noise_power*comms_snr_th*(K+1)))
# R_max = min(R_r_max, R_c_max, 150) # 150 m unless otherwise required
R_min = 50
R_max = 200
dists = np.arange(R_min, R_max, (R_max - R_min) / NUM_POINTS)

# Power variation
# sensing_powers = np.arange(1, total_power - 1, (total_power - 2) / NUM_POINTS)
# comms_powers = total_power - sensing_powers

# Reuse dist variation
# reuse_dists = np.arange(20, 200, (180 / NUM_POINTS))

# Bandwidth Variation
# c_Bandwidths = np.arange(100, 600, 100)
# r_Bandwidths = 600 - c_Bandwidths

theory = np.zeros((NUM_PLOTS, NUM_POINTS))
sim_outage = np.zeros((NUM_PLOTS, NUM_POINTS))
sim_snr = np.zeros((NUM_PLOTS, NUM_POINTS))
sim_pd = np.zeros((NUM_PLOTS, NUM_POINTS))
vru_pd = np.zeros((NUM_PLOTS, NUM_POINTS))
sim_pd_t = np.zeros((NUM_PLOTS, NUM_POINTS))
vru_pd_t = np.zeros((NUM_PLOTS, NUM_POINTS))

c_noise_power = 10**(-70/10)

for a in range(NUM_PLOTS):
    print("Plot ", a+1, "of ", NUM_PLOTS)

    if (a == 2):
        NUM_INTERFERERS = 6
    else:
        NUM_INTERFERERS = 2

    for d in range(NUM_POINTS):
        print("Point ", d+1, "of ", NUM_POINTS)
        outage_t = np.zeros(L, complex)
        radar_snr_total = 0
        vru_snr_total = 0
        outage_count = 0
        radar_outage_count = 0
        vru_outage_count = 0
        snr_total = 0

        link_dist = dists[d]

        # reuse_dist = reuse_dists[d]

        # c_Bandwidth = c_Bandwidths[d]
        # r_Bandwidth = r_Bandwidths[d]
        # s_noise_power = constants.k * noiseFigure * noiseTemp * r_Bandwidth

        # sensing_power = sensing_powers[d]
        # comms_power = comms_powers[d]

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
                phi[0] = -theta[0]
                phi[1:P] = np.random.uniform(-constants.pi/2, constants.pi/2, P - 1)

                # match a:
                #     case 0:
                #         # Worst case - interferers at reuse dist of SV, so +- link dist
                #         interferer_dists = np.array([reuse_dist - link_dist, reuse_dist + link_dist])
                #     case 1: 
                #         interferer_dists = np.array([-link_dist, link_dist]) + ((reuse_dist-12) * np.random.beta(5, 1, NUM_INTERFERERS))
                #     case 2:
                #         interferer_dists = np.zeros(NUM_INTERFERERS) + 12 # 6 Interferers at 12 m
                # for i in range(NUM_INTERFERERS): # Enforce minimum distance
                #     if interferer_dists[i] < 12:
                #         interferer_dists[i] = 12

                # # Comms channel matrix
                # H_c = np.matrix(np.zeros((Ncr, Nct), complex))
                # alpha_angles = np.random.uniform(0, 2*constants.pi, P)
                # alpha = np.cos(alpha_angles[0]) + 1j*np.sin(alpha_angles[0])
                # omega = 2*constants.pi*c_carrier_f*relative_velocity*symbol_period*np.sin(theta[0])/constants.c #doppler frequency shift
                # H_c += np.sqrt((K * Ncr * Nct)/(K+1)) * alpha * P_u(Ncr, np.sin(phi[0])) @ P_u(Nct, np.sin(theta[0])).H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))
                # for i in range(1, P):
                #     alpha = np.cos(alpha_angles[i]) + 1j*np.sin(alpha_angles[i])
                #     omega = 2*constants.pi*c_carrier_f*relative_velocity*symbol_period*np.sin(theta[i])/constants.c #doppler frequency shift
                #     H_c += np.sqrt((Ncr * Nct)/((K+1)*P)) * alpha * P_u(Ncr, np.sin(phi[i])) @ P_u(Nct, np.sin(theta[i])).H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))

                # # Interferer channel matrices
                # iH_c = np.zeros(NUM_INTERFERERS, np.matrix)
                # i_theta = np.zeros(P)
                # i_phi = np.zeros(P)
                # for k in range(NUM_INTERFERERS):
                #     # Interferer transmission angle
                #     i_theta[0] = link_angles[l] + np.random.uniform(constants.pi / 6, constants.pi/3)
                #     # NLOS path AoDs
                #     i_theta[1:P] = np.random.uniform(-constants.pi/2, constants.pi/2, P - 1)
                #     # AoAs
                #     i_phi[0] = -i_theta[0]
                #     i_phi[1:P] = np.random.uniform(-constants.pi/2, constants.pi/2, P - 1)

                #     iH_c[k] = np.matrix(np.zeros((Ncr, Nct), complex))
                #     alpha_angles = np.random.uniform(0, 2*constants.pi, P)
                #     alpha = np.cos(alpha_angles[0]) + 1j*np.sin(alpha_angles[0])
                #     omega = 2*constants.pi*c_carrier_f*relative_velocity*symbol_period*np.sin(i_theta[0])/constants.c #doppler frequency shift
                #     iH_c[k] += np.sqrt((K * Ncr * Nct)/(K+1)) * alpha * P_u(Ncr, np.sin(i_phi[0])) @ P_u(Nct, np.sin(i_theta[0])).H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))
                #     for i in range(1, P):
                #         alpha = np.cos(alpha_angles[i]) + 1j*np.sin(alpha_angles[i])
                #         omega = 2*constants.pi*c_carrier_f*relative_velocity*symbol_period*np.sin(i_theta[i])/constants.c #doppler frequency shift
                #         iH_c[k] += np.sqrt((Ncr * Nct)/((K+1)*P)) * alpha * P_u(Ncr, np.sin(i_phi[i])) @ P_u(Nct, np.sin(i_theta[i])).H * (np.cos(omega * time_step) + 1j*np.sin(omega * time_step))

                # # Interferer transmitter beamformers
                # i_f = np.zeros((NUM_INTERFERERS, Nct), complex)
                # for j in range(NUM_INTERFERERS):
                #     i_f[j] = P_u(Nct, (np.sin(i_theta[j])))[0] # Assume no error, since it doesn't affect much

                # Radar
                clutterInterference = 1.2153e-11
                car_dists = 12 + np.random.pareto(0.7, NUM_CARS)
                # car_dists = np.zeros(NUM_CARS) + 24
                z = 0.01
                s_noise = np.random.normal(0, s_noise_power)
                # s_noise = s_noise_power
                jamming = np.sum((z**2*sensing_power*Nsr*tAntennaGain*rAntennaGain*target_rcs*s_carrier_w**2)/((4*constants.pi)**3*car_dists**4))
                radarSnr = (sensing_power * Nsr * tAntennaGain * rAntennaGain * target_rcs * s_carrier_w**2) / ((4*constants.pi)**3 * (abs(s_noise) + clutterInterference + jamming) * link_dists[l]**4)
                vru_radarSnr = (sensing_power * Nsr * tAntennaGain * rAntennaGain * vru_rcs * s_carrier_w**2) / ((4*constants.pi)**3 * (abs(s_noise) + clutterInterference + jamming) * link_dist**4)

                radar_snr_total += radarSnr
                vru_snr_total += vru_radarSnr
                if(radarSnr < radar_snr_th):
                    radar_outage_count += 1
                if(vru_radarSnr < radar_snr_th):
                    vru_outage_count += 1

                # rxSensitivity = 0.05
                # angle_error = rxSensitivity / radarSnr
                # # Comms transmitter beamformer
                # f = P_u(Nct, (np.sin(theta[0]) + np.random.normal(0, angle_error)))
                # receiverNoise = (np.random.normal(0, c_noise_power / np.sqrt(2), Nct) + 1j * np.random.normal(0, c_noise_power / np.sqrt(2), Nct)) @ np.identity(Nct)

                # r_error = np.random.normal(0, angle_error)
                # # Comms receiver beamformer
                # omega = P_u(Ncr, (np.sin(theta[0]) + r_error))

                # interference = 0
                # for j in range(NUM_INTERFERERS):
                #     interference += np.abs(omega.H @ iH_c[j] @ i_f[j])**2 / (4*constants.pi*interferer_dists[j])**2

                # comms_snr = (comms_power * c_carrier_w**2)/Nct * np.abs(omega.H @ H_c @ f)**2 / ((4*constants.pi*link_dists[l])**2 * (np.abs(omega.H @ receiverNoise) + interference))
                # snr_total += comms_snr
                # if(comms_snr < comms_snr_th):
                #     if (not outage_iteration):
                #         outage_count += 1
                #         outage_iteration = True

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
        sim_outage[a][d] = float(outage_count) / N
        sim_snr[a][d] = float(snr_total) / N
        sim_pd_t[a][d] = pow(np.e, -(radar_snr_th * (N*L)) / radar_snr_total)
        vru_pd_t[a][d] = pow(np.e, -(radar_snr_th * (N*L)) / vru_snr_total)
        sim_pd[a][d] = 1 - float(radar_outage_count) / (N*L)
        vru_pd[a][d] = 1 - float(vru_outage_count) / (N*L)


# Convert to dB
sim_snr = 20*np.log10(sim_snr)
sim_rate = c_Bandwidth * np.log2(1 + sim_snr) / 1_000_000
sim_se = sim_rate*1_000_000 / (c_Bandwidth + r_Bandwidth)
print("Outage:\n", sim_outage)
print("SNR:\n", sim_snr)
print("PD:\n", sim_pd)
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

colours = ["red", "yellow", "blue"]
names = ["Ideal", "Realistic", "None"]

plt.figure()
for i in range(NUM_PLOTS):
    plt.plot(dists, sim_outage[i], 'ko-', label=names[i], linewidth=0.5, markerfacecolor=colours[i], markersize=6)
    # plt.plot(theory[i], comms_powers, 'ko--', label="Theory = "+str(-170 + i*10) , linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Hop Distance (m)")
plt.ylabel("Outage")
plt.yscale('log')
# plt.ylim([0, 10])
plt.xlim([R_min, R_max])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.figure()
for i in range(NUM_PLOTS):
    plt.plot(dists, sim_rate[i], 'ko-', label=names[i], linewidth=0.5, markerfacecolor=colours[i], markersize=6)
plt.xlabel("Hop Distance (m)")
plt.ylabel("Rate (Mbits/sec)")
plt.yscale('log')
# plt.ylim([0, 10])
plt.xlim([R_min, R_max])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plot_pd = np.mean(sim_pd, 0)

plt.figure()
plt.plot(dists, plot_pd, 'ko-', linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Hop Distance (m)")
plt.ylabel("Probability of Detection")
plt.yscale('log')
# plt.ylim([0.1, 1])
plt.xlim([R_min, R_max])
# plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plot_pd_t = np.mean(sim_pd_t, 0)

plt.figure()
plt.plot(dists, plot_pd_t, 'ko-', linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Hop Distance (m)")
plt.ylabel("Probability of Detection")
plt.yscale('log')
# plt.ylim([0.1, 1])
plt.xlim([R_min, R_max])
# plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plot_vru_pd_t = np.mean(vru_pd_t, 0)

plt.figure()
plt.plot(dists, plot_vru_pd_t, 'ko-', linewidth=0.5, markerfacecolor="none", markersize=6)
plt.xlabel("Hop Distance (m)")
plt.ylabel("Probability of Detection")
plt.yscale('log')
# plt.ylim([0.1, 1])
plt.xlim([R_min, R_max])
# plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.figure()
for i in range(NUM_PLOTS):
    plt.plot(dists, sim_se[i], 'ko-', label=names[i], linewidth=0.5, markerfacecolor=colours[i], markersize=6)
plt.xlabel("Hop Distance (m)")
plt.ylabel("Spectral Efficiency (Bits/s/Hz)")
# plt.yscale('log')
# plt.ylim([0, 10])
plt.xlim([R_min, R_max])
plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.show()