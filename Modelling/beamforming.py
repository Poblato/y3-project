import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

def P_u(u, psi):
    arr = np.zeros(u, complex)
    for i in range(u):
        arr[i] = (np.cos(constants.pi * psi * i) + 1j*np.sin(constants.pi * psi * i)) / (2*np.sqrt(u))
    return np.matrix(arr).transpose()

angles = np.arange(-np.pi/2, np.pi/2, 0.01)
gains = np.zeros(angles.shape)

signal = np.zeros(8, complex) + 1

for i in range(angles.size):
    gains[i] = np.abs(signal @ P_u(8, np.sin(angles[i])))


plt.figure()
plt.plot(angles, gains, 'ko-', linewidth=0.5, markerfacecolor="none", markersize=0)
plt.xlabel("Signal Angle (radians)")
plt.ylabel("Gain")
# plt.yscale('log')
# plt.ylim([0, 10])
# plt.xlim([0, 15])
# plt.legend()
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--')

plt.show()