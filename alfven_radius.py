import numpy as nup
import pprint
from sympy import symbols, Eq, solve


pp = pprint.PrettyPrinter(indent=4)

# Available data
Rsun = 696e6  # [m]
Rs = Rstar = 4*Rsun  # [m]
vinf = 600e3  # [m/s]
Bp = 1  # 1 Tesla = 1e4 Gauss
bet_degrees = 60
bet = nup.deg2rad(bet_degrees)  # Angle beta (magnetic vs rotation axis))
zet_degrees = 0
zet = nup.deg2rad(zet_degrees)  # Zeta angle (magnetic longitude)
Msun = 2e30  # SolarMass in Kg
Mlos_ = 1e-9  # Mass Loss [Solar Masses / year]
Mlos = Mlos_ * Msun / (365*24*3600)  # [Kg / s]
T = 1 * 24 * 3600  # Period [s]
w = 2*nup.pi/T
mu = magnetic_permeability = 1.256637e-6

# Variable
r = symbols('r')

# Formulas
vw = vinf * (1 - Rs / r)
B = 1/2 * Bp * (Rs / r)**3
d = r * nup.sqrt(1 - nup.sin(bet)**2 * nup.cos(zet)**2)
ro = Mlos / (4 * nup.pi * r**2 * vw)

eq1 = Eq(-B**2/(8*nup.pi) + 1/2 * ro * vw**2 + 1/2 * ro * w**2 * d**2, 0)
pp.pprint(solve(eq1))

aa = 4*Mlos*w**2*(1-(nup.sin(bet))**2*(nup.cos(zet))**2)
bb = 4*Mlos*(vinf**2)
cc = -8*Mlos*Rs*(vinf**2)
dd = 4*Mlos*(Rs**2)*(vinf**2)
ee = -(Bp**2)*(Rs**6)*vinf/mu
ff = (Bp**2)*(Rs**7)*vinf/mu

eq2 = Eq(aa * r**7 + bb * r**5 + cc * r**4 + dd * r**3 + ee * r + ff, 0)
pp.pprint(solve(eq2))

"""
Other formulas and constants
kB = 1.380649e-23  # Boltzmann constant [J/K]
np = 1e7  # [cm^(-3)]
Tp = 1e6  # [K]

L = (Bp**2 / (16 * nup.pi * np * kB * Tp)) ** (1/6)
"""
