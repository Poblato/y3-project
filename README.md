# y3-project
A software modelling project on mobile wireless networks
# Requirements
This project uses python 3.12.6 (64 bit) with the following packages:
- numpy (version 2.1.1)
- scipy (version 1.14.1)
- matplotlib  (version 3.9.2)

Python can be intalled from [here](https://www.python.org/downloads/release/python-3126/).
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
Simulation of a generic mobile network, calculating area spectral efficiency. It recreates the simulation from
## isac.py
Simulation of a generic ISAC system, measuring the tradeoff between probability of detection and rate for a given probability of false alarm. It recreates the simulation created in 
## v2x.py
Simulation of an ISAC system for a V2V network, measuring outage and power consumption