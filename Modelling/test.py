from enum import Enum
import random as r
import matplotlib.pyplot as plt
import numpy as np

class Err(Enum):
    OK = 0
    GenericNonFatalError = -1
    GenericFatalError = 1
    RangeError = 2

def ShowError(err):
    errorCodes = {
        0: "OK",
        -1: "Generic Non-Fatal Error",
        1: "Generic Fatal Error",
        2: "Range Error"
    }
    return errorCodes[err]

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
        print(ShowError(err) + " in pdf (iteration: " + i + ")")
        if (err > 0): # Error is fatal, skip iteration
            continue
    y, err = Model(x)
    if (err != Err.OK):
        print(ShowError(err) + " in model (iteration: " + i + ")")
        if (err > 0): # Error is fatal, skip iteration
            continue
    results[i] = np.array((x, y))

fig, ax = plt.subplots()
ax.scatter(results[:,0], results[:,1])
plt.show()