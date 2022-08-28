"""
alfven_radius.py

2022 - Marc Rosanes Siscart (marcrosanes@gmail.com)
C/ Carles Collet, 7; Barcelona (08031); Catalonia

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

The objective(s) of this file are:
  - Compute the Alfvén Radius of a UCD (or other (sub)stellar/stellar) object
"""

import numpy as np
import pprint
from sympy import symbols, Eq, solve

pp = pprint.PrettyPrinter(indent=4)

# Available data
Rsun = 6.96e8  # [m]
R_obj = 4*Rsun  # [m]
vinf = 600e3  # [m/s]
# Bp = 1  # 1 Tesla = 1e4 Gauss
Bp = 1e4  # [Gauss]
bet_degrees = 60
bet = np.deg2rad(bet_degrees)  # Angle beta (magnetic vs rotation axis))
zet_degrees = 0
zet = np.deg2rad(zet_degrees)  # Zeta angle (magnetic longitude)
Msun = 2e30  # SolarMass in Kg
Mlos_ = 1e-9  # Mass Loss [Solar Masses / year]
Mlos = Mlos_ * Msun / (365*24*3600)  # [Kg / s]
T = 1 * 24 * 3600  # Period [s]
w = 2*np.pi/T
mu = magnetic_permeability = 1.256637e-6

# Variable
r = symbols('r')

# Formulas
vw = vinf * (1 - R_obj / r)
B = 1/2 * Bp * (R_obj / r)**3
ro = Mlos / (4 * np.pi * r**2 * vw)

rotation_phase = zet_array = np.deg2rad(np.array(range(0, 361, 5)))
alfven_radius_array = []
for zet in zet_array:
    d = r * np.sqrt(1 - np.sin(bet)**2 * np.cos(zet)**2)
    eq1 = Eq(-B**2/(8*np.pi) + 1/2 * ro * vw**2 + 1/2 * ro * w**2 * d**2, 0)
    solutions = solve(eq1)
    # pp.pprint(solve(eq1))
    for solution in solutions:
        if solution.is_real:
            if solution >= 0 and solution > 2*R_obj:
                alfven_radius_array.append(solution)

print(alfven_radius_array)

Ra = np.average(alfven_radius_array)
print(Ra)

