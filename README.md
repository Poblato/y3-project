# y3-project
A software modelling project on mobile wireless networks
# Requirements
This project uses python 3.12.6 (64 bit) with the following packages:
- numpy (version 2.1.1)
- scipy (version 1.14.1)
- matplotlib  (version 3.9.2)

Python can be installed from [here](https://www.python.org/downloads/release/python-3126/).
Packages can be installed from a terminal with the following commands:
```
pip install --force-reinstall -v "numpy==2.1.1"
pip install --force-reinstall -v "scipy==1.14.1"
pip install --force-reinstall -v "matplotlib==3.9.2"
```
# Files
## test.py
Basic Monte Carlo test software. This was used to gain familiarity with Monte Carlo Techniques
## basic_sim.py
This simulation attempted to recreate the results of an existing study. It simulates a user downlink from a BS, with 6 interfering BS's. The channel model includes path-loss, multipath fading, and shadowing. The received signal power (before multipath fading and shadowing) is calculated from the transmitted signal power using the two-slope model
$S = \frac{K}{r^a (1 + R/g)^b} S_t$, where $K$ is a constant, $R$ (m) is the distance between the mobile and the BS, $a$ is the path-loss exponent (two for this simulation), $b$ is the additional path-loss exponent, and $S_t$ (W) is the transmitted signal power.
This simulation uses Nakagami multipath fading, a model for signals dominated by NLOS paths. It is modelled using a gamma distribution with the following probability density function (PDF). $p_s(s) = (\frac{m}{\Omega})^m \frac{s^{m-1}}{\Gamma (m)} \exp(-m \frac{s}{\Omega}), s >= 0$, where $\Omega$ is the mean received power, $m$ is the Nakagami fading parameter, and $\Gamma()$ is the gamma function. Shadowing is modelled using a log-normal distribution with the following PDF:
$p_s(s) = \frac{\xi}{\sqrt{2 \pi} \sigma s} \exp(-\frac{(\xi \ln(s) - \mu) ^ 2}{2 \sigma ^ 2}),   s >= 0$
where $\xi = 10/\ln{10}$ is a constant, $\mu = \xi \ln{S}$ is the area mean received power, and $\sigma$ is the standard deviation of the distribution.
The SNR of the received signal can be calculated as:
$\gamma_d = \frac{S_d(r)}{\sum^{n_I}_{i=1}S_i(r_i) + n_c}$
where $S_d(r)$ is the received signal power at a distance $r$, $S_i(r_i)$ is power received from the ith interferer at a distance $r_i$, and $n_c$ is Gaussian noise at the receiver, modelled by $n_c \sim \mathcal{N}(0, \sigma^2)$, where $\sigma^2$ is the variance of the noise.
The measured performance indicators for this simulation are the area spectral efficiency (ASE), rate, and outage.
The ASE is calculated by:
$A_e = \frac{4}{\pi R_u^2R^2} \log_2(1 + \gamma_d)$, where $R_u$ is the normalised reuse distance (shared area between nearby BS's) and $R$ is the distance to the user. The rate is calculated using the Shannon-Hartley Theorem:
$R = B\log_2(1 + \gamma_d)$, where $B$ is the bandwidth of the link.
The outage is:
$P_{out} = \Pr\{\gamma_d < \gamma_{th}\}$, where $\gamma_{th}$ is a threshold SNR.
User position and BS positions are determined randomly for each iteration.
The simulation requires a varying number of iterations to ensure convergence to 3 significant figures depending on its configuration. Without multipath fading or shadowing, 10000 iterations are required. With either multipath fading or shadowing, 100000 iterations are required. With both, 300000 iterations are required.
## isac.py
This simulation attempted to recreate the results of an existing study. It simulates a basic radar channel with complex Gaussian noise, and evaluates the trade-off between theoretical rate of communication and probability of detection in sensing. It uses a simple channel model that combines both the Radar Cross Section (RCS) of the target, and the signal attenuation due to path loss and fading into a single scalar.

The total power available is allocated between the sensing and communication waveforms, such that $P_t = P_r + P_c$. Where $P_t$ is the total power available, $P_r$ is the power allocated to the sensing waveform, and $P_c$ is the power allocated to the communication waveform.

The basis of the sensing system is the following set of equations, where $H_0$ represents the null hypothesis (where there is no target present), and $H_1$ the alternative hypothesis (where there is a target present).
$H_0: \begin{cases}
        x_d = \gamma _d  s_r + n_d \\
        x_s = n_s
    \end{cases}
H_1: \begin{cases}
        x_d = \gamma _d  s_r + n_d \\
        x_s = \gamma _t  s_r + n_s
    \end{cases}$, where $x_d, x_s$ represent the reference and surveillance signals respectively, $\gamma _d, \gamma _t$ are the channel coefficients, $n_d, n_s$ are $L * L$ matrices of the complex Gaussian noise at the reference and surveillance receivers, and $s_r$ is an arbitrary $L * 1$ vector representing the sampled radar waveform, such that $||s_r||^2 = P_r$. $L$ is a constant representing the size of the delay-Doppler matrix.

The measured characteristics of the sensing system are its probability of detection ($P_D$) and probability of false alarm ($P_{FA}$), quantified by the following equations:
$P_D = \Pr\{x_s^H B x_s \geq \sigma ^2 \gamma | H_1\}$
$P_{FA} = \Pr\{x_s^H B x_s \geq \sigma ^2 \gamma | H_0\}$, where $B = \frac{x_d x_d^H + \sigma ^2 \gamma I}{||x_d||^2 + \sigma ^2 \gamma}$, $\sigma$ is an arbitrary constant and $\gamma$ is a constant typically chosen to achieve a specified $P_{FA}$.

The theoretical information rate of the communication waveform in bits per channel per second is as follows:
$R = B\log_2(1 + P_c \frac{|\gamma_c| ^ 2}{\sigma_c^2})$, where $P_c \frac{|\gamma_c| ^ 2}{\sigma_c^2}$ represents the instantaneous SNR at the receiver, $\gamma_c$ is the communication channel coefficient, and $\sigma_c^2$ is the variance of the noise at the communication receiver.
## v2x.py
![image](Diagrams/sim_geometry.pdf "Simulation Geometry")

The V2V simulation emulates an arbitrary stretch of straight road, as shown in \ref{fig:sim_geometry}. Each car on the road is capable of radar detection, and wireless transmission and reception and it is assumed they all use the same configuration. It is based on the simulation developed in \cite{Multi-hop-v2v-network}, using the same models for radar and communications channels.
It simulates a source vehicle (SV) that detects its surroundings using radar, and sends a communication signal to a designated target vehicle (TV). This is done through multiple hops, via relay vehicles (RVs) using DF relaying. Each vehicle has $N_{c,r}$ and $N_{c,t}$ receive and transmit antennas for communication respectively, as well as $N_{r,t}$ and $N_{r,r}$ receive and transmit antennas for radar respectively.
A simplified model for radar is used, that assumes valid detection of the relevant target given that the radar SNR:
$\gamma_{r,n} = \frac{p_{r,n}N_{r,r}G_tG_r\sigma\lambda_r^2}{(4\pi)^3(\eta_r+C+J)R_{n}^4}$ exceeds a specified threshold $\gamma_{r,th}$. $p_{r,n}$ represents the power allocated to the radar (link $n$), $G_t$ and $G_r$ are the Tx and Rx antenna gains, $\sigma$ is the target RCS, $\lambda_r$ is the radar carrier wavelength, and $R_{n}$ is the distance to the target. The noise portion is composed of the receiver noise $\eta_r$, clutter interference $C$, and jamming $J$. Receiver noise is given by $\eta_r=k_bT_0FB_r$, where $k_b$ is the Boltzmann constant, $T_0$ is the noise temperature, $F$ is the noise figure of the radar, and $B_r$ is the bandwidth of the radar receiver. Clutter is a constant. Jamming is assumed to be caused by reflections from other nearby vehicles. It is therefore calculated by 
$J=\sum_i\frac{z^2p_{r,n}N_{r,r}G_tG_r\sigma\lambda_r^2}{(4\pi)^3R_i^4}$, where $z$ represents the gain of the side-lobes and $R_i$ is the distance to the $i$ th closest vehicle.

Based on this, probability of detection $P_D$ can be defined as
$P_D = \Pr\{\gamma_{r,n,l} > \gamma_{r,th}\}$
Probability of false alarm cannot be easily calculated from this method, but can be indirectly controlled through selection of $\gamma_{r,th}$ (where higher $\gamma_{r,th}$ corresponds to a lower PFA).

The channel matrix for communications for a link $n$ at time instant $k$ is defined by:
$H_{c,n,k} = \sqrt{\frac{K N_{c,r} N_{c,t}}{K+1}}\alpha_{n,0} a_r(\theta_{n,0}) a_t^*(\theta_{n,0})e^{j\omega_0 k} + \sqrt{\frac{N_{c,r} N_{c,t}}{(K+1) P}}\sum_{i=1}^P\alpha_{n,i} a_r(\phi_{n,i}) a_t^*(\theta_{n,i})e^{j\omega_i k}$,
where the first term represents the LOS component, and the second represents the P NLOS components. $K$ is the Rician factor of the channel representing the relative power of the LOS component, $\alpha_{n,i}$ is a unity gain, constant phase factor used to simulate multipath fading, $\omega_i$ is the Doppler frequency shift of the signal, $\phi_{n,i}$ and $\theta_{n,i}$ are the angles of arrival (AoAs) and angles of departure (AoDs) of the signal paths. Taking $c$ as the speed of light, $f_c$ as the frequency of the communications carrier, $v_m$ as the relative velocity of the target, and $T_s$ as symbol period, the Doppler frequency shift is defined as $\omega_i = 2\pi f_cv_mT_s\sin{(\phi_{n,i})}/c$. Using the function $P_U(\psi) = \frac{1}{\sqrt{U}}[1, e^{j\pi\psi}, ... , e^{j\pi\psi(U-1)}]^T$ to generate beamforming vectors, $a_r(\theta_{n,i})$ and $a_r(\phi_{n,i})$ can be defined as $a_t(\theta_{n,i}) = P_{N_{c,t}}(\sin{\theta_{n,i}})$ and $a_r(\phi_{n,i}) = P_{N_{c,r}}(\sin{\phi_{n,i}})$.

In communication, the transmit beamforming vector is calculated by:
$f_{n} = P_{N_{c,t}}(\sin{\theta_{n,0} + e_{t,n}})$
where $e_{t,n}$ represents the error in the angle to the TV measured by the radar system, modelled by $e_{t,n} \sim \mathcal{N}(0, \beta/\gamma_{r,n})$, where $\beta$ is the radar's Rx sensitivity. Error in the distance is not considered, as it is small, and has very little effect on the received signal, whereas due to beamforming of the signal, a small error in angle can result in a large drop in received signal magnitude.

The signal received, combined by the receiver's beamforming vector of 
$\omega_{n} = P_{n_{c,r}}(\sin{\theta_{n,0} + e_{r,n}})$
where $e_{r,n}$ uses the same model as $e_{t,n}$ to produce the signal:
$y_{c,n,k} = \omega_{n}^H( \sqrt{\frac{p_{c,n}\lambda_c^2}{N_{c,t}(4\pi R_{n})^2}}H_{c,n,k} f_{n}s + n_{n,k} + i_{n,k})$
where $s$ is the transmitted signal, and $n_{n,k}$ represents the instantaneous complex Gaussian noise at the receiver, modelled by $n_{n,k} \sim \mathcal{CN}(0, \eta_r I_{N_{c,t}})$. $i$ represents the interference at the receiver, calculated as the sum of contribution from each interferer, modelled by 
$i_{n,k} = \Sigma_i\sqrt{\frac{p_{c,n}\lambda_c^2}{N_{c,t}(4\pi R_{i})^2}}H_{i,n,k} f_{i,n}$
where $R_i$ is the distance to the ith interferer, $H_{i,n,k}$ is the channel matrix of the ith interferer, and $f_{i,n}$ is the transmit beamforming vector of the ith interferer.
In addition, the sensed velocity information can be used to compensate for the Doppler shifts caused by the relative velocity of the receiver. This produces the compensated signal:
$y_{c,n,l,k} = e^{-j2\pi f_c v_m T_s(\sin{\theta_{n,l,0}} + e_{r,n,l})k/c}y_{c,n,l,k}.$
Which has an SNR of:
$\gamma_{c,n} = \frac{p_{c,n}\lambda_c^2}{N_{c,t}} \frac{|\omega_{n}^H H_{c,n,k}f_{n}|^2 / (4\pi R_{n})^2}{\eta_c + \Sigma_{i}|\omega_{n}^H H_{i,n,k}f_{i,n}|^2 / (4\pi R_{i})^2}$.
Outage for a given link is given by:
$P_{out,l} = \Pr\{\gamma_{c,n,l} < \gamma_{c,th}\}$
or can be calculated analytically by:
$P_{out,l} = 1 - \prod^{N_l}_{n=1}\exp(-\frac{\gamma_{c,th}}{\gamma_{c,n,l}})$
where $N_l$ is the number of hops in the link.

The theoretical rate possible in a link can be calculated using Shannon-Hartley theorem:
$C = B \log_2(1 + \gamma)$
where $B$ is the bandwidth allocated to a link, and $\gamma$ is its SNR.

![image](Diagrams/simulation_model.pdf "Simulation Flowchart")

Figure \ref{fig:sim_flowchart} shows a flowchart of the simulation. This is a general case, measuring the rate and outage of the system. Full simulations have small changes to vary different parameters or take different measurements, but the core algorithm of the simulation remains the same.

In this paper, three models for bandwidth allocation are considered. First: no bandwidth allocation. In this model all cars present use the same frequencies, and are all interferers. This is modelled by six interferers at a distance of 15 m. Second: ideal bandwidth allocation. In this model all cars are assigned different frequency bands up to a specified reuse distance. Assuming worst case, this is modelled by two interferers at the specified reuse distance from the SV. Third: a realistic bandwidth allocation model. This model is the same as the second, but overheads and inaccuracies is the bandwidth allocation cause the 'real' reuse distance to vary below the specified value (down to a minimum value). This is modelled by a Beta distribution, such that the distance from the SV to each interferer is:
$R_i = R_{min} + \rho_n(R_u - R_{min}), \rho_n \sim Beta(5, 1)$
where $R_u$ is the reuse distance and $R_{min}$ is the minimum distance.
Note: These distances (for the second and third model) are measured to the SV, so the distances to the TV/RV will be these distances plus or minus the link distance.

Different variants of the file v2x.py are in the Power, Dist, Reuse_Dist, Bandwidth, and Hops folders, designed to vary and plot those different variables.

# Figures
Contains the plotted results of the simulations
# Diagrams
Contains other diagrams created for the project