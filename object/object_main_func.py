"""
obj.py

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
    - This file contains the main functions of the project to create a 3D
    model of a (sub)stellar object radio emission; especially, the
    gyrosynchrotron radiation produced in the middle-magnetosphere. The file
    instantiates the objects from other files of the project (obj.py, etc.),
    to work with them and find the radio emission which result from
    a series of input parameters.
"""

import time
import numpy as np
import pyqtgraph as pg

from object.obj import OBJ


def plot_3D(L=30, n=7, beta=0, rotation_angle=0, inclination=90,
            Robj_Rsun_scale=4, Bp=3000, Pr=1, D_pc=1, f=1e9, Ra=16,
            l_middlemag=4, δ=2, r_ne=0.002, v_inf=600, plot3d=True):
    obj = OBJ(L=L, n=n, beta=beta, rotation_angle=rotation_angle,
              inclination=inclination, Robj_Rsun_scale=Robj_Rsun_scale, Pr=Pr,
              Bp=Bp, D_pc=D_pc, f=f, Ra=Ra, l_middlemag=l_middlemag, δ=δ,
              r_ne=r_ne, v_inf=v_inf, plot3d=plot3d)
    # LoS grid points in different systems of coordinates
    points_LoS, points_LoS_in_B = obj.LoS_cube()
    # Compute and Plot the (sub)stellar object dipole magnetic vector field
    # and plot it together with the rotation and the magnetic axes, and the
    # different coordinate systems
    obj.obj_compute_and_plot(points_LoS_in_B, points_LoS)


def specific_intensities_2D(
        L=30, n=25, beta=0, rotation_angle=0, inclination=90,
        Robj_Rsun_scale=4, Bp=3000, Pr=1, D_pc=1, f=1e9, Ra=16,
        l_middlemag=4, δ=2, r_ne=0.002, v_inf=600, plot3d=True):
    """
    Notes:
      - Create a OBJ with a low grid sampling "n" per edge for
        "3D magnetic field" plotting (eg: ~7), to be able to plot the
        Magnetic Field vectors without losing a good visibility of the vectors
      - Create a OBJ with higher grid sampling "n" per edge (eg: >30 <101)
        in order to have a better resolution, but without going to too
        long computation times for "2D specific intensity images" and for
        "1D flux density plots"
    """
    start_time = time.time()
    obj = OBJ(L=L, n=n, Robj_Rsun_scale=Robj_Rsun_scale,
              beta=beta, rotation_angle=rotation_angle,
              inclination=inclination, Bp=Bp, Pr=Pr, D_pc=D_pc, f=f, Ra=Ra,
              l_middlemag=l_middlemag, δ=δ, r_ne=r_ne, v_inf=v_inf,
              plot3d=plot3d)
    # LoS grid points in different systems of coordinates
    points_LoS, points_LoS_in_B = obj.LoS_cube()
    # Compute and Plot the (sub)stellar object dipole magnetic vector field
    # and plot it together with the rotation and the magnetic axes, and the
    # different coordinate systems
    obj.obj_compute_and_plot(points_LoS_in_B, points_LoS)
    voxels_inner, voxels_middle = obj.find_magnetosphere_regions()
    # obj.plot_middlemag_in_slices(voxels_middle, marker_size=2)
    obj.LoS_voxel_rays()
    obj.compute_flux_density_LoS()
    print("\n- Total Flux Density in the plane perpendicular to the LoS:")
    print("{:.4g}".format(obj.total_flux_density_LoS) + " mJy")
    end_time = time.time()
    print("- Time to compute the 2D specific intensities image,\n"
          " using %d elements per cube edge:"
          " %d seconds" % (obj.n, end_time - start_time))
    obj.plot_2D_specific_intensity_LoS()


def flux_densities_1D(
        L=30, n=7, beta=0, inclination=90, Robj_Rsun_scale=4, Bp=3000,
        Pr=1, D_pc=1, f=1e9, Ra=16, l_middlemag=4, δ=2, r_ne=0.002, v_inf=600,
        rotation_angle_step=10, plot3d=False, mk_new_qapp=False):
    """
    Flux densities 1D in function of the rotation phase angles of the
    (sub)stellar object
    """
    start_time_flux_densities = time.time()
    # Rotation phase angles from 0º to 360º (every "rotation_angle_step"
    #  degrees)
    rotation_phases = []
    for i in range(0, 361, rotation_angle_step):
        rotation_phases.append(i)

    # Flux densities in function of the rotation phase angles
    flux_densities = []
    for rot_phase in rotation_phases:
        obj = OBJ(L=L, n=n, Robj_Rsun_scale=Robj_Rsun_scale,
                  beta=beta, rotation_angle=rot_phase, inclination=inclination,
                  Bp=Bp, Pr=Pr, D_pc=D_pc, f=f, Ra=Ra, l_middlemag=l_middlemag,
                  δ=δ, r_ne=r_ne, v_inf=v_inf, plot3d=plot3d)
        points_LoS, points_LoS_in_B = obj.LoS_cube()
        obj.obj_compute_and_plot(points_LoS_in_B, points_LoS)
        obj.find_magnetosphere_regions()
        obj.LoS_voxel_rays()
        obj.compute_flux_density_LoS()
        print("Flux Density at rotation angle " + str(rot_phase) + " is: "
              + str(np.round(obj.total_flux_density_LoS, 2)))
        # obj.plot_2D_specific_intensity_LoS()
        flux_densities.append(np.round(obj.total_flux_density_LoS, 2))
        print("Angle " + str(rot_phase) + " computed\n")

    duration = (time.time() - start_time_flux_densities) / 60.0
    print("- Time to compute the 1D specific intensities graph along the\n"
          " rotation of the (sub)stellar object:"
          + " {:.4g}".format(duration) + " minutes\n")

    # 1D Plot of the flux densities in function of the rotation phase angles
    # Make new QApp (running EventLoop), only if a QApp has not been
    # previously instantiated
    if mk_new_qapp:
        app = pg.mkQApp()
        pg.plot(rotation_phases, flux_densities, pen="b", symbol='o')
        app.exec_()
    else:
        pg.plot(rotation_phases, flux_densities, pen="b", symbol='o')

