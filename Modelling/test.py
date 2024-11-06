import random as r
import matplotlib.pyplot as plt
import numpy as np
import math

xmin = 0.0
xmax = 1.0
numIterations = 1000

def ProbDensityFunc(x):
    #Ensure Input is in range
    if ((x < xmin) | (x > xmax)):
        return 0
    #Uniform
    return x

def Model(x):
    #Ensure Input is in range
    if ((x < xmin) | (x > xmax)):
        return 0
    
    # For now an exponential
    return 2 ** x

# Set seed
r.seed(2, 100)

results = np.zeros((numIterations, 2))

for i in range(numIterations):
    x = r.random()
    x = ProbDensityFunc(x)
    y = Model(x)
    results[i] = np.array((x, y))

#calculate average analytically (for exponential):
a_avg = 1/math.log(2, math.e)

#calculate average from MC
mc_avg = sum(results[:,1]) / numIterations
error = abs(a_avg - mc_avg) / a_avg

print("Analytical average = ", a_avg)
print("MC average = ", mc_avg)
print("Error = ", error)

fig, ax = plt.subplots()
ax.scatter(results[:,0], results[:,1])
plt.show()