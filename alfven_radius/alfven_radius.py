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

import time
import pprint
import numpy as np
import pyqtgraph as pg
from sympy import symbols, Eq, solve

pp = pprint.PrettyPrinter(indent=4)

# Constants and known data
Rsun = 6.96e8  # [m]
Msun = 2e30  # SolarMass in Kg


def approximate_alfven_radius(beta=60, zeta=0, Robj2Rsun=4, P_rot=1, Bp=1e4,
                              vinf=600e3, Mlos_ = 1e-9):
    """
    Compute approximate Alfvén Radius -> Ra computed at a single
    Magnetic Longitude zeta

    :param beta: angle beta (angle between magnetic and rotation axes)
    :param Robj2Rsun: scale factor of Robj compared to Rsun
    :param P_rot: [days]
    :param Bp: Magnetic Field Strength at the Poles [Gauss]
    :param vinf: wind velocity at "infinity" [m/s]
    :param Mlos_: Mass Loss [Solar Masses / year]
    :return:
    """
    start_time = time.time()

    # Variable
    r = symbols('r')

    # Initial Data and Conversions
    R_obj = Robj2Rsun * Rsun  # [m]
    beta = np.deg2rad(beta)
    zeta = np.deg2rad(zeta)
    P_rot_sec = P_rot * 24 * 3600  # Period [s]
    w = 2 * np.pi / P_rot_sec
    Mlos = Mlos_ * Msun / (365 * 24 * 3600)  # [Kg / s]

    # Formulas
    vw = vinf * (1 - R_obj / r)
    B = 1/2 * Bp * (R_obj / r)**3
    ro = Mlos / (4 * np.pi * r**2 * vw)

    d = r * np.sqrt(1 - np.sin(beta)**2 * np.cos(zeta)**2)
    eq1 = Eq(-B**2/(8*np.pi) + 1/2 * ro * vw**2
             + 1/2 * ro * w**2 * d**2, 0)
    solutions = solve(eq1)
    # pp.pprint(solve(eq1))
    for solution in solutions:
        if solution.is_real:
            if solution >= 0 and solution > 2*R_obj:
                alfven_radius_approx = solution

    alfven_radius_approx_norm = alfven_radius_approx / R_obj
    print("\nApproximate Alfvén Radius:")
    print("{:.4g}".format(alfven_radius_approx_norm))

    duration = (time.time() - start_time) / 60.0
    print("\nApproximate Alfvén Radius computation took"
          + " {:.4g}".format(duration) + " minutes\n")
    return alfven_radius_approx


def averaged_alfven_radius(beta=60, Robj2Rsun=4, P_rot=1, Bp=1e4,
                           vinf=600e3, Mlos_ = 1e-9):
    """
    Plot 1D of Alfvén Radius as a function of the magnetic longitude zeta and
    compute the average Alfvén Radius

    :param beta: angle beta (angle between magnetic and rotation axes)
    :param Robj2Rsun: scale factor of Robj compared to Rsun
    :param P_rot: [days]
    :param Bp: Magnetic Field Strength at the Poles [Gauss]
    :param vinf: wind velocity at "infinity" [m/s]
    :param Mlos_: Mass Loss [Solar Masses / year]
    :return:
    """
    start_time = time.time()

    # Variable
    r = symbols('r')

    # Initial Data and Conversions
    R_obj = Robj2Rsun * Rsun  # [m]
    beta = np.deg2rad(beta)
    P_rot_sec = P_rot * 24 * 3600  # Period [s]
    w = 2 * np.pi / P_rot_sec
    Mlos = Mlos_ * Msun / (365 * 24 * 3600)  # [Kg / s]

    # Formulas
    vw = vinf * (1 - R_obj / r)
    B = 1 / 2 * Bp * (R_obj / r) ** 3
    ro = Mlos / (4 * np.pi * r ** 2 * vw)

    # Magnetic Longitude [rad]
    magnetic_longitude_angles = np.array(range(0, 361, 5))
    magnetic_longitude_angles = np.deg2rad(magnetic_longitude_angles)  # [rad]
    alfven_radius_array = []
    for zeta in magnetic_longitude_angles:
        d = r * np.sqrt(1 - np.sin(beta) ** 2 * np.cos(zeta) ** 2)
        eq1 = Eq(-B ** 2 / (8 * np.pi) + 1 / 2 * ro * vw ** 2
                 + 1 / 2 * ro * w ** 2 * d ** 2, 0)
        solutions = solve(eq1)
        # pp.pprint(solve(eq1))
        for solution in solutions:
            if solution.is_real:
                if solution >= 0 and solution > 2 * R_obj:
                    alfven_radius_array.append(solution)

    alfven_radius_array = np.array(alfven_radius_array)
    alfven_radius_array = alfven_radius_array.astype(float)

    # Normalized Alfvén Radius, in R* units
    alfven_radius_array_norm = np.round(alfven_radius_array / R_obj, 3)
    Ra = np.average(alfven_radius_array_norm)

    print(magnetic_longitude_angles)
    print()
    print(alfven_radius_array_norm)
    print()
    print("Average Alfvén Radius:")
    print(Ra)

    duration = (time.time() - start_time) / 60.0
    print("\nAlfvén Radius computation took"
          + " {:.4g}".format(duration) + " minutes\n")

    app = pg.mkQApp()
    pg.plot(magnetic_longitude_angles, alfven_radius_array_norm,
            pen="b", symbol='o')
    app.exec_()
    return Ra


if __name__ == "__main__":
    approximate_alfven_radius()
    # averaged_alfven_radius()

