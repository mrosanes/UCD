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

The objectives of this file are:
  - Compute the Alfvén Radius of a (Sub)Stellar object
"""

import time
import pprint
import numpy as np
import pyqtgraph as pg
from sympy import symbols, Eq, solve

from constants import Rsun, Msun, Mp

pp = pprint.PrettyPrinter(indent=4)


def alfven_radius_at_given_zeta(beta=60, zeta=0, Robj2Rsun=4, P_rot=1, Bp=1e4,
                                v_inf=600, M_los = 1e-9):
    """
    Compute Alfvén Radius (Ra) at a given magnetic longitude zeta (ζ)

    :param beta: angle β between magnetic and rotation axes of the object
    :param zeta: angle ζ indicating the magnetic longitude
    :param Robj2Rsun: scale factor of Robj compared to Rsun
    :param P_rot: [days]
    :param Bp: Magnetic Field Strength at the Poles [Gauss]
    :param v_inf: wind velocity at "infinity" [km/s]
    :param M_los: Mass Loss [Solar_Masses/year]
    :return: Ra_at_given_zeta in [R*]
    """
    start_time = time.time()

    # Variable
    r = symbols('r')

    # Initial Data and Conversions
    R_obj = Robj2Rsun * Rsun  # [cm]
    beta = np.deg2rad(beta)
    zeta = np.deg2rad(zeta)
    P_rot_sec = P_rot * 24 * 3600  # Period [s]
    w = 2 * np.pi / P_rot_sec
    Mlos = M_los * Msun / (365 * 24 * 3600)  # [g/s]
    v_inf = v_inf * 1e5  # [cm/s]

    # Formulas
    vw = v_inf * (1 - R_obj / r)
    B = 1/2 * Bp * (R_obj / r)**3
    ro = Mlos / (4 * np.pi * r**2 * vw)

    angular_term = np.sqrt(1 - np.sin(beta)**2 * np.cos(zeta)**2)
    d = r * angular_term
    eq1 = Eq(-B**2/(8*np.pi) + 1/2 * ro * vw**2
             + 1/2 * ro * w**2 * d**2, 0)
    solutions = solve(eq1)
    # pp.pprint(solve(eq1))
    for solution in solutions:
        if solution.is_real:
            if solution >= 0 and solution > 2*R_obj:
                Ra_at_given_zeta = solution
    Ra_at_given_zeta_norm = Ra_at_given_zeta / R_obj

    ###########################################################################
    # Compute 'neA' in Trigilio04 (a.k.a. 'nw' in Leto06) density of
    # electrons at the Alfvén Radius:
    # B_Ra = 1/2*Bp*(Robj/Ra_at_given_zeta)³ [Trigilio04 - Formula(5)]
    # -> B_Ra = 0.5*Bp / Ra_at_given_zeta_norm³
    # neA = B_Ra² / (4 PI Mp (v² + w²*d²))

    d = Ra_at_given_zeta * angular_term
    B_Ra = 0.5 * Bp / Ra_at_given_zeta_norm**3
    neA = B_Ra**2 / (4 * np.pi * Mp * (v_inf**2 + w**2 * d**2))

    print("\nAlfvén Radius at magnetic longitude ζ " + str(zeta) + "º, is:")
    print("{:.4g}".format(Ra_at_given_zeta_norm) + " R*")

    print("\nB_Ra: B field at the Alfvén Radius at ζ " + str(zeta) + "º, is:")
    print("{:.4g}".format(B_Ra) + " G")

    print("\nne,A: Density of electrons at the Alfvén Radius at ζ "
          + str(zeta) + "º, is:")
    print("{:.4g}".format(neA) + " cm^(-3)")

    duration = (time.time() - start_time) / 60.0
    print("\nAlfvén Radius and neA, at a given longitude, computation took"
          + " {:.4g}".format(duration) + " minutes\n")
    return Ra_at_given_zeta_norm


def averaged_alfven_radius(beta=60, Robj2Rsun=4, P_rot=1, Bp=1e4,
                           v_inf=600e3, M_los = 1e-9):
    """
    Plot 1D of Alfvén Radius as a function of the magnetic longitude zeta and
    compute the average Alfvén Radius

    :param beta: angle β between magnetic and rotation axes of the object
    :param Robj2Rsun: scale factor of Robj compared to Rsun
    :param P_rot: [days]
    :param Bp: Magnetic Field Strength at the Poles [Gauss]
    :param v_inf: wind velocity at "infinity" [km/s]
    :param M_los: Mass Loss [Solar_Masses/year]
    :return:
    """
    start_time = time.time()

    # Variable
    r = symbols('r')

    # Initial Data and Conversions
    R_obj = Robj2Rsun * Rsun  # [cm]
    beta = np.deg2rad(beta)
    P_rot_sec = P_rot * 24 * 3600  # Period [s]
    w = 2 * np.pi / P_rot_sec
    Mlos = M_los * Msun / (365 * 24 * 3600)  # [g/s]
    v_inf = v_inf * 1e5  # [cm/s]

    # Formulas
    vw = v_inf * (1 - R_obj / r)
    B = 1 / 2 * Bp * (R_obj / r) ** 3
    ro = Mlos / (4 * np.pi * r ** 2 * vw)

    # Magnetic Longitude [rad]
    magnetic_longitude_angles = np.array(range(0, 361, 10))
    magnetic_longitude_angles = np.deg2rad(magnetic_longitude_angles)  # [rad]
    alfven_radius_array = []
    B_Ra_array = []
    neA_array = []
    for zeta in magnetic_longitude_angles:
        angular_term = np.sqrt(1 - np.sin(beta) ** 2 * np.cos(zeta) ** 2)
        d = r * angular_term
        eq1 = Eq(-B ** 2 / (8 * np.pi) + 1 / 2 * ro * vw ** 2
                 + 1 / 2 * ro * w ** 2 * d ** 2, 0)
        solutions = solve(eq1)
        # pp.pprint(solve(eq1))
        for solution in solutions:
            if solution.is_real:
                if solution >= 0 and solution > 2 * R_obj:
                    alfven_radius = float(solution)
                    alfven_radius_array.append(alfven_radius)
        B_Ra = 0.5 * Bp / (alfven_radius / R_obj)**3
        B_Ra_array.append(B_Ra)
        d = alfven_radius * angular_term
        neA = B_Ra ** 2 / (4 * np.pi * Mp * (v_inf ** 2 + w ** 2 * d ** 2))
        neA_array.append(neA)

    alfven_radius_array = np.array(alfven_radius_array)
    B_Ra_array = np.array(B_Ra_array)
    neA_array = np.array(neA_array)

    # Normalized Alfvén Radius, in R* units
    alfven_radius_array_norm = np.round(alfven_radius_array / R_obj, 3)
    Ra_avg = np.round(np.average(alfven_radius_array_norm), 3)

    # B_Ra array
    B_Ra_array = np.round(B_Ra_array, 3)
    B_Ra_avg = np.round(np.average(B_Ra_array), 3)

    # neA array
    neA_array = np.round(neA_array, 3)
    neA_avg = np.round(np.average(neA_array), 3)

    magnetic_longitude_degrees = (180 / np.pi) * np.array(
        magnetic_longitude_angles)
    magnetic_longitude_degrees = np.round(magnetic_longitude_degrees, 2)

    print("\nMagnetic longitudes array")
    print(magnetic_longitude_degrees)
    print("\nAlfvén Radius array")
    print(alfven_radius_array_norm)
    print("\nAverage Alfvén Radius:")
    print(Ra_avg)
    print("\nAverage B(Ra) [Gauss]:")
    print(B_Ra_avg)
    print("\nAverage neA [cm^(-3)] (density of electrons at Alfvén Radius):")
    print(neA_avg)

    duration = (time.time() - start_time) / 60.0
    print("\nAlfvén Radius computations took"
          + " {:.4g}".format(duration) + " minutes\n")

    # Commented if the QCoreApplication::exec event loop is already running
    # app = pg.mkQApp()
    window = pg.plot(magnetic_longitude_degrees, alfven_radius_array_norm,
                     pen="b", symbol='o')
    window.setWindowTitle("Alfvén Radius")

    window = pg.plot(magnetic_longitude_degrees, B_Ra_array,
                     pen="b", symbol='o')
    window.setWindowTitle("B_Ra [Gauss]")

    window = pg.plot(magnetic_longitude_degrees, neA_array,
                     pen="b", symbol='o')
    window.setWindowTitle("neA: e- density at Ra [cm(-3)]")
    # app.exec_()
    return Ra_avg, B_Ra_avg, neA_avg


if __name__ == "__main__":
    alfven_radius_at_given_zeta()
    # averaged_alfven_radius()

