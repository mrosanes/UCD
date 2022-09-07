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
            Robj_Rsun_scale=4, Bp=7700, Pr=1, D_pc=10, f=1e9, Ra=15,
            l_middlemag=7, δ=2, r_ne=0.002, v_inf=600, plot3d=True):
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
        L=30, n=25, inclination=90, beta=0, rotation_angle=0,
        rotation_offset=0, Robj_Rsun_scale=4, Bp=7700, Pr=1, D_pc=10, f=1e9,
        Ra=15, l_middlemag=7, δ=2, neA=3e6, r_ne=0.002, v_inf=600,
        inner_contrib=True, n_p0=3e9, T_p0=1e5, scale_colors=1,
        colormap="linear", plot3d=True):
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
    obj = OBJ(
        L=L, n=n, Robj_Rsun_scale=Robj_Rsun_scale, inclination=inclination,
        beta=beta, rotation_angle=rotation_angle,
        rotation_offset=rotation_offset, Bp=Bp, Pr=Pr, D_pc=D_pc, f=f, Ra=Ra,
        l_middlemag=l_middlemag, δ=δ, neA=neA, r_ne=r_ne, v_inf=v_inf,
        inner_contrib=inner_contrib, n_p0=n_p0, T_p0=T_p0, plot3d=plot3d)
    # LoS grid points in different systems of coordinates
    points_LoS, points_LoS_in_B = obj.LoS_cube()
    obj.obj_compute_and_plot(points_LoS_in_B, points_LoS)
    voxels_middle = obj.find_magnetosphere_regions()
    # obj.plot_middlemag_in_slices(voxels_middle, marker_size=2)
    obj.LoS_voxel_rays()
    obj.compute_flux_density_LoS()
    print("\n- Total Flux Density in the plane perpendicular to the LoS:")
    print("{:.4g}".format(obj.total_flux_density_LoS) + " mJy")
    end_time = time.time()
    print("- Time to compute the 2D specific intensities image,\n"
          " using %d elements per cube edge:"
          " %d seconds" % (obj.n, end_time - start_time))
    obj.plot_2D_specific_intensity_LoS(scale_colors=scale_colors,
                                       colormap=colormap)


def flux_densities_1D(
        L=30, n=7, inclination=90, beta=0, rotation_offset=0,
        Robj_Rsun_scale=4, Bp=7700, Pr=1, D_pc=10, f=1e9, Ra=15, l_middlemag=7,
        δ=2, neA=3e6, r_ne=0.002, v_inf=600, rotation_angle_step=10,
        inner_contrib=True, n_p0=3e9, T_p0=1e5, use_symmetry=False,
        plot3d=False, mk_new_qapp=False):
    """
    Flux densities 1D in function of the rotation angles of the (sub)stellar
    object
    """
    start_time_flux_densities = time.time()
    # Rotation angles from 0º to 360º (every "rotation_angle_step" degrees)
    rotation_angles = []
    if use_symmetry:
        end_angle = 181
    else:
        end_angle = 361
    for i in range(0, end_angle, rotation_angle_step):
        rotation_angles.append(i)

    # Flux densities in function of the rotation of the object
    flux_densities = []
    for rot_angle in rotation_angles:
        obj = OBJ(
            L=L, n=n, Robj_Rsun_scale=Robj_Rsun_scale,
            inclination=inclination, beta=beta, rotation_angle=rot_angle,
            rotation_offset=rotation_offset, Bp=Bp, Pr=Pr, D_pc=D_pc, f=f,
            Ra=Ra, l_middlemag=l_middlemag, δ=δ, neA=neA, r_ne=r_ne,
            v_inf=v_inf, inner_contrib=inner_contrib, n_p0=n_p0, T_p0=T_p0,
            plot3d=plot3d)
        points_LoS, points_LoS_in_B = obj.LoS_cube()
        obj.obj_compute_and_plot(points_LoS_in_B, points_LoS)
        obj.find_magnetosphere_regions()
        obj.LoS_voxel_rays()
        obj.compute_flux_density_LoS()
        print("Flux Density at rotation angle " + str(rot_angle) + " is: "
              + str(np.round(obj.total_flux_density_LoS, 3)) + " mJy")
        # obj.plot_2D_specific_intensity_LoS()
        flux_densities.append(np.round(obj.total_flux_density_LoS, 3))
        print("Angle " + str(rot_angle) + " computed\n")

    if use_symmetry:
        pop_elem = False
        if 180 in rotation_angles:
            pop_elem = True
        rotation_angles = []
        for i in range(0, 361, rotation_angle_step):
            rotation_angles.append(i)
        second_half_flux_densities = flux_densities[::-1]
        if pop_elem:
            second_half_flux_densities.pop(0)
        flux_densities += second_half_flux_densities

    print("Rotation angles:")
    print(rotation_angles)
    print("Flux densities:")
    print(flux_densities)
    duration = (time.time() - start_time_flux_densities) / 60.0
    print("\n- Time to compute the 1D specific intensities graph along the\n"
          " rotation of the (sub)stellar object:"
          + " {:.2g}".format(duration) + " minutes\n")

    # 1D Plot of the flux densities in function of the object rotation.
    # Make new QApp (running EventLoop), only if a QApp has not been
    # previously instantiated
    if mk_new_qapp:
        app = pg.mkQApp()
        pg.plot(rotation_angles, flux_densities, pen="b", symbol='o')
        app.exec_()
    else:
        pg.plot(rotation_angles, flux_densities, pen="b", symbol='o')

