import random as r
import matplotlib.pyplot as plt
import numpy as np
import math
from err import Err

xmin = 0.0
xmax = 1.0
numIterations = 1000

def ProbDensityFunc(x):
    #Ensure Input is in range
    if ((x < xmin) | (x > xmax)):
        return {0, Err.RangeError}
    #Uniform
    return {1, Err.OK}

def Model(x):
    #Ensure Input is in range
    if ((x < xmin) | (x > xmax)):
        return {0, Err.RangeError}
    
    # For now an exponential
    return {2 ** x, Err.OK}

# Set seed for debugging
r.seed(2, 100)

results = np.zeros((numIterations, 2))

for i in range(numIterations):
    x = r.random()
    y, err = ProbDensityFunc(x)
    if (err != Err.OK):
        print(Err.ShowError(err), " in pdf (iteration: ", i, ")")
        if (err > 0): # Error is fatal, skip iteration
            continue
    y, err = Model(x)
    if (err != Err.OK):
        print(Err.ShowError(err), " in model (iteration: ", i, ")")
        if (err > 0): # Error is fatal, skip iteration
            continue
    results[i] = np.array((x, y))

#calculate average analytically:
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