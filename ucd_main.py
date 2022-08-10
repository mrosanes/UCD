"""
ucd_main.py

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

The objective(s) of this file is to execute the 3D model, implementing the
developments done in the other Python files of the project. This file
contains the function "main".
"""

import time
import numpy as np
import pyqtgraph as pg

from ucd import UCD


def plot_3D(n=7, beta=0, rotation_angle=0, inclination=90,
            plot3d=True):
    ucd = UCD(
        n=n, beta=beta, rotation_angle=rotation_angle, inclination=inclination,
        plot3d=plot3d)
    # LoS grid points in different systems of coordinates
    points_LoS, points_LoS_in_B = ucd.LoS_cube()
    # Compute and Plot the UCD (or other (sub)stellar object) dipole
    # magnetic vector field and plot it together with the rotation and the
    # magnetic axes, and the different coordinate systems
    ucd.ucd_compute_and_plot(points_LoS_in_B, points_LoS)


def specific_intensities_2D(n=25, beta=0, rotation_angle=0, inclination=90,
                            plot3d=True):
    """
    Notes:
      - Create a UCD with a low grid sampling "n" per edge (eg: <13),
        to be able to plot the vectors without losing a relatively good
        visibility of the vectors
      - Create a UCD with a higher grid sampling "n" per edge (eg: >31 <101)
        in order to have a middle-magnetosphere with a better sampling
        resolution, but without going to too long computation times (for
        101 points per cube edge, ~2min for finding the points of the
        middle-magnetosphere)
    """
    start_time = time.time()
    ucd = UCD(
        n=n, beta=beta, rotation_angle=rotation_angle, inclination=inclination,
        plot3d=plot3d)
    # LoS grid points in different systems of coordinates
    points_LoS, points_LoS_in_B = ucd.LoS_cube()
    # Compute and Plot the UCD (or other (sub)stellar object) dipole
    # magnetic vector field and plot it together with the rotation and the
    # magnetic axes, and the different coordinate systems
    ucd.ucd_compute_and_plot(points_LoS_in_B, points_LoS)
    voxels_middlemag = ucd.find_middle_magnetosphere()
    # ucd.plot_middlemag_in_slices(voxels_middlemag, marker_size=2)
    ucd.LoS_voxel_rays()
    ucd.compute_flux_density_LoS()
    print("Total Flux Density in the plane perpendicular to the LoS:")
    print("{:.4g}".format(ucd.total_flux_density_LoS))
    ucd.plot_2D_specific_intensity_LoS()
    end_time = time.time()
    print("\nUCD computations for %d elements per cube edge, took:\n"
          "%d seconds\n" % (ucd.n, end_time - start_time))


def flux_densities_1D(
        n=15, beta=0, inclination=90,
        plot3d=False):
    """
    Flux densities 1D in function of the rotation phase angles of the UCD (
    or other (sub)stellar object)
    """
    start_time_flux_densities = time.time()
    # Rotation phase angles from 0º to 360º (each 10º)
    rotation_phases = []
    for i in range(36):
        rotation_phases.append(i * 10)

    # Flux densities in function of the rotation phase angles
    flux_densities = []
    for rot_phase in rotation_phases:
        ucd = UCD(n=n,
                  beta=beta, inclination=inclination, rotation_angle=rot_phase,
                  plot3d=plot3d)
        points_LoS, points_LoS_in_B = ucd.LoS_cube()
        ucd.ucd_compute_and_plot(points_LoS_in_B, points_LoS)
        ucd.find_middle_magnetosphere()
        ucd.LoS_voxel_rays()
        ucd.compute_flux_density_LoS()
        # ucd.plot_2D_specific_intensity_LoS()
        flux_densities.append(np.round(ucd.total_flux_density_LoS, 3))

    end_time_flux_densities = time.time()
    print("Time to compute the 1D specific intensities graph along the\n"
          " rotation of the UCD (or other (sub)stellar object), "
          "took:\n%d seconds\n" % (
                  end_time_flux_densities - start_time_flux_densities))

    # 1D Plot of the flux densities in function of the rotation phase angles
    app = pg.mkQApp()
    pg.plot(rotation_phases, flux_densities, pen="b", symbol='o')
    app.exec_()


def main():
    # Angles in degrees
    beta = 0
    rotation_angle = 0
    inclination = 90

    plot_3D(n=7, beta=beta, rotation_angle=rotation_angle,
            inclination=inclination, plot3d=True)

    specific_intensities_2D(
        n=25,
        beta=beta, rotation_angle=rotation_angle, inclination=inclination,
        plot3d=False)

    flux_densities_1D(n=7, beta=beta, inclination=inclination, plot3d=False)


if __name__ == "__main__":
    main()

