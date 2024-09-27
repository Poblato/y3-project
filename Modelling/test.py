from enum import Enum
import random as r
import matplotlib as plt
import numpy as np

class Err(Enum):
    OK = 0
    GenericFatalError = 1
    GenericNonFatalError = 2
    RangeError = 3

def ShowError(err):
    errorCodes = {
        "OK",
        "Generic Non-Fatal Error",
        "Generic Fatal Error",
        "Range Error"
    }
    return errorCodes[err]

xmin = 0
xmax = 1
numIterations = 1000

def ProbDensityFunc(x):
    #Ensure Input is in range
    if (x < xmin | x > xmax):
        return {0, Err.RangeError}
    #Uniform
    return {1, Err.OK}

def Model(x):
    #Ensure Input is in range
    if (x < xmin | x > xmax):
        return {0, Err.RangeError}
    
    # For now an exponential
    return {2 ** x, Err.OK}

# Set seed for debugging
r.seed(2, 100)

results = np.zeros(numIterations, 2)

for i in range(numIterations):
    x = r.random()
    y, err = ProbDensityFunc(x)
    if (err != Err.OK): # If error, skip rest of loop iteration
        print(ShowError() + " in pdf, skipping iteration " + i)
        continue
    y, err = Model(x)
    if (err != Err.OK): # If error, skip rest of loop iteration
        print(ShowError() + " in model, skipping interation " + i)
        ShowError()
        continue
    results[i] = {x, y}

fig, ax = plt.subplots()
ax.scatter(results)
plt.show()