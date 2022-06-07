import numpy as nup
from sympy import symbols, Eq, solve

Rsun = 696e6  # [m]
Rs = Rstar = 4*Rsun  # 4 * Rsun
vinf = 600e3  # [m/s]
Bp = 1  # 1 Tesla = 1e4 Gauss
bet_degrees = 60
bet = nup.deg2rad(bet_degrees)  # Angle beta (magnetic vs rotation axis))
zet_degrees = 30
zet = nup.deg2rad(zet_degrees)  # Zeta angle (magnetic longitude)
Msun = 2e30  # SolarMass in Kg
Mlos_ = 1e-9  # Solar Masses / year
Mlos = Mlos_ * Msun / (365*24*3600)  # [Kg / s]
T = 1 * 24 * 3600  # Period [s]
w = 2*nup.pi/T

kB = 1.380649e-23  # Boltzmann constant [J/K]
np = 1e7  # [cm^(-3)]
Tp = 1e6  # [K]


r = symbols('r')
vw = vinf * (1 - Rs / r)
B = 1/2 * Bp * (Rs / r)**3
d = r * nup.sqrt(1 - nup.sin(bet)**2 * nup.cos(zet)**2)
ro = Mlos / (4 * nup.pi * r**2 * vw)

eq1 = Eq(-B**2/(8*nup.pi) + 1/2 * ro * vw**2 + 1/2 * ro * w**2 * d**2)
solve(eq1)

L = (Bp**2 / (16 * nup.pi * np * kB * Tp)) ** (1/6)

