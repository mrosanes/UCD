"""
ucd.py

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
    - Create the 3D model of the UCD (or other (sub)stellar) object
    - Compute the magnetic vector field 'B' of a dipole of a
      (sub)stellar object at different points of a meshgrid, for an object
      with not aligned magnetic, rotation, and line of sight [LoS] axes.
    - Find all points of the grid inside the middle magnetosphere
    - Apply the Pipeline of C.Trigilio el al. (ESO 2004) [Appendix A]
      A&A 418, 593–605 (2004)
      DOI: 10.1051/0004-6361:20040060
"""

import pprint
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

from voxel import Voxel
from LoS_voxels_ray import LoS_Voxels_Ray

###############################################################################
# PIPELINE

"""
– Magnetosphere 3D sampling and magnetic field vectors B calculation (DONE)
– Definition of the Alfvén radius and of inner, middle and outer 
  magnetosphere (DONE)

– Calculation of the number density ne of the non-thermal electrons in each 
point of the grid (middle-magnetosphere)
– Calculation of emission and absorption coefficients

– Integration of the transfer equation along paths parallel to the 
line of sight
– Brightness distribution in the plane of the sky, total flux emitted 
toward the Earth
"""
###############################################################################
# Acronyms and Glossary
# - LoS: Line of Sight
# - in_B (suffix): Magnetic Field coordinates system
# - Middlemag: Middle-magnetosphere
# - Rotax: Rotation axis coordinates system
# - Roted: Rotated (sub)stellar object coordinates system
###############################################################################


class UCD(object):
    """
    Class UCD dedicated to the object under study. The object studied with
    the present 3D model, can be a UCD or another stellar object with similar
    magnetic characteristics (MPC star, etc.).
    """
    def __init__(self, n=5, L=30, Robj_Rsun_scale=2, Pr=1, Bp=1,
                 beta=1, rotation_angle=1, inclination=89, plot3d=False):
        """
        Constructor method
        :param int n: Number of points per side of the mesh grid
        :param int L: Length of the mesh grid in stellar radius units
        :param float Robj_Rsun_scale: Ratio of the UCD or other
          (sub)stellar object radius, regarding the Sun
        :param float Pr: Rotation Period [days]
        :param float Bp: Magnetic Field at the pole of the (sub)stellar object
        :param float beta: Angle between Magnetic and Rotation axis [degrees]
        :param float rotation_angle: Rotation angle [degrees]
        :param float inclination: Angle between LoS and Rotation axis [degrees]
        :param bool plot3d: Plot or not the magnetic field in a 3D plot
        """

        #######################################################################
        # Utils
        self.pp = pprint.PrettyPrinter(indent=4)

        #######################################################################
        self.voxels = []

        #######################################################################
        # Constants:
        # Length magnetic and rotation axes
        self.len_axes = 40

        # Radius of sun
        self.Rsun = 6.96e8

        # Jupyter Radius in meters [m] (approx) ~ Brown Dwarf Radius
        self.Rj = 0.1 * self.Rsun

        # Robj_Rsun_scale: Ratio of the UCD or other (sub)stellar object
        # radius, regarding the Sun radius.
        # R_obj: radius of the object; being the object a (sub)stellar object
        # like a UCD, or a bigger stellar object, like an MCP star or
        # other with similar magnetic properties
        self.R_obj = Robj_Rsun_scale * self.Rsun

        # TODO: Alfvén radius to be updated according to output of script:
        #  alfven_radius.py
        self.Ra = 15 * self.R_obj

        # Masa del protón:
        Mp = 1.6726e-27

        # TODO: We take the velocity of the wind as a free parameter?
        # Velocity of the wind:
        v_inf = 400e3  # [m/s]

        #######################################################################
        # Parameters of the model
        # n: Num points per edge in meshgrid cube (use odd number in order to
        # get one point in the origin of coordinates [center of the grid and
        # of the UCD]):
        self.n = n

        # Distance from the Earth (point of observation) to the source (UCD)
        # or other (sub)stellar object
        # TODO: in Parsecs??
        self.D = 1

        # Total length of the meshgrid cube in number of (sub)stellar radius:
        self.L = L

        # Voxel edge length
        self.voxel_len = (L / (n - 1)) * self.R_obj

        # Star Period of Rotation in days
        self.Pr = Pr

        # Strength of the B at the pole of the star
        self.Bp = Bp * 10000  # in Gauss (1T = 10000 Gauss)

        # Magnetic Momentum:  m = 1/2 (Bp Rs)
        self.m = 1 / 2 * self.Bp * self.R_obj

        # |B_Ra| -> z = 0
        self.B_Ra = self.m / (self.Ra ** 3)

        # TODO: Verify
        # From Leto2006
        # B**2 / (8*PI) = 1/2 * pw * v_inf**2
        # With: pw = Mp * nw
        nw = neA = self.B_Ra**2 / (4 * np.pi * Mp * v_inf**2)
        r_ne = 0.5  # Ratio r_ne = Ne / neA: from 10^(-4) - 1 (Trigilio_2004)
        self.Ne = Ne = r_ne * neA

        # Lorentz factor
        γ = 1.2

        # δ~2 in some MCP stars according C.Trigilio el al. (ESO 2004))
        self.δ = δ = 2

        # In each point of the middle magnetosphere (Formula 6 - Trigilio_2004)
        # Electrons isotropically distributed
        self.N_γ = N_γ = Ne * (γ - 1) ** (-δ)
        print("Ne and N_γ")
        print(Ne)
        print(N_γ)

        # Array 2D (image) of specific intensities in the plane perpendicular
        # to the LoS
        self.specific_intensities_array = np.zeros((self.n, self.n))

        # Total Flux Density (Sv) in the plane perpendicular to the LoS
        self.total_flux_density_LoS = 0

        """
        r: Radius vector:
           Distance from the (sub)stellar object center till a concrete point
           outside of the star (at the surface of the star: r = R_obj)
        """

        #######################################################################
        # Free Parameters of the Model
        # (Parameter ranges indicated with: [x, y])

        # l_mid (or 'l'): equatorial thickness of the magnetic shell for the
        # middle magnetosphere (which is added to Ra)
        self.l_mid = 4 * self.R_obj

        # l/rA: equatorial thickness of the magnetic shell
        # in Alfvén Radius units
        # l/rA: [0.025, 1];
        self.eq_thick = self.l_mid / self.Ra

        """
        – Ne: total number density of the non-thermal electrons: 
            Ne: [10^2 - 10^6 cm^(−3)], with Ne < n{e,A}
            [Ne/n{e,A} in the range 10^(−4) to 1]
            with:
              n{e,A}: number density of thermal plasma at the Alfvén point

        – δ: spectral index of the non-thermal electron energy distribution
            δ: [2, 4]
        – Tp:  temperature of the inner magnetosphere
            Tp: [10^5 K, 10^7 K]
        - np: number plasma density at the base of the inner magnetosphere 
        (r = R∗)
            np: [10^7 –10^10 cm(−3)]
            with: 
              Tp.np.k{B} = p{ram} -> Tp.np = p{ram}/k{B}: 
              [10^14 cgs – 10^15 cgs]
          (Tp and np, related with the plasma in the Star post-shock region: 
          inside the region protected by the Star closed magnetic field lines)       
        – Rotation: if the star does not rotate, the thermal plasma density is 
            considered constant within the inner magnetosphere; if the star 
            rotates, the thermal plasma density decreases linearly outward, 
            while the temperature increases. Tp and np are considered as the 
            values at r = R∗ (base of the inner magnetosphere)
        """

        #######################################################################
        # Angles

        # Expressed in degrees:
        # Angle from rotation to magnetic axis: [~0º - ~180º]
        self.beta = beta
        # UCD star rotation [~0º - ~360º]
        self.phi = self.rotation = rotation_angle
        # Rotation Axis inclination measured from the Line of Sight:
        # [~0º - ~180º]
        # Information about rotation axis orientations:
        #   . Orbits with the rotation axis in the plane of the sky (~90º)
        #   . Orbits with the rotation axis towards the LoS (~0º)
        self.inc = inclination

        # Transformed to radians:
        self.beta_r = np.deg2rad(self.beta)
        self.phi_r = np.deg2rad(self.phi)
        self.inc_r = np.deg2rad(self.inc)

        #######################################################################
        # Trigonometric functions of the used angles
        sin_b = np.round(np.sin(self.beta_r), 4)
        cos_b = np.round(np.cos(self.beta_r), 4)

        sin_p = np.round(np.sin(self.phi_r), 4)
        cos_p = np.round(np.cos(self.phi_r), 4)

        sin_i = np.round(np.sin(self.inc_r), 4)
        cos_i = np.round(np.cos(self.inc_r), 4)

        #######################################################################
        # Rotation Matrices

        # Beta (b): angle from magnetic axis to rotation axis (b = beta)s
        # (Note: from my calculation, I think R1 should be
        # [[cos_b, 0, sin_b], [0, 1, 0], [-sin_b, 1, cos_b]] instead of the
        # indicated in the paper:
        # [[cos_b, 0, -sin_b], [ 0, 1, 0],[sin_b, 1, cos_b]])
        self.R1 = np.array([[ cos_b,  0,  - sin_b],
                            [ 0,      1,  0      ],
                            [ sin_b,  0,  cos_b  ]])

        # p (phi): Rotation of the star angle (p = phi = rot)
        self.R2 = np.array([[cos_p, -sin_p, 0],
                            [sin_p, cos_p,  0],
                            [0,     0,      1]])

        # i (inc): Inclination of the orbit, from line of sight to rotation
        # axis
        self.R3 = np.array([[ sin_i,  0,  cos_i],
                            [ 0,      1,  0    ],
                            [-cos_i,  0,  sin_i]])

        # Rotation Matrices for each degree of freedom
        self.R1_inv = self.R1.transpose()
        self.R2_inv = self.R2.transpose()
        self.R3_inv = self.R3.transpose()

        """
        Information about "complete" Rotation Matrices R and R_inv:

        R = R3 . R2 . R1 to go from points expressed in (x, y, z), to points
           expressed in (x', y', z')

        R^(-1) = R1^(-1) . R2^(-1) . R3^(-1) to go from points expressed
           in (x', y', z'), to points expressed in (x, y, z)
        """
        # Rotation Matrices (Complete Rotation)
        self.R = self.R3.dot(self.R2).dot(self.R1)
        self.R_inv = self.R1_inv.dot(self.R2_inv).dot(self.R3_inv)

        #######################################################################
        # Formulas
        """
        Ram pressure:
        p{ram} = ρ.v^2
        
        p{ram} = B^2 / (8*PI) ~ np Tp K{B}
        
        v(r) = v(inf).(1 - R_obj/r)
          with r: [Rstar, +inf]
        
        Delta = Smin / Smax 
          -> if Smin == Smax -> Delta = 1 
        
        L = (Bp^2/(16.PI.np.Tp.K{B}))^(1/6)
          L ~ [18.Rstar, 23.Rstar] for 
          Bp ~ [5000 Gauss, 10000 Gauss]
        
        Gas density in outer region:
        ro = dM / (4.PI.r².v(r)) 
          with dM: loss of mass of the star in solar masses per year
        
        Equation of field line in magnetic dipole:
        r = L.cos²(lambda)
        with: 
        . lambda: angle between magnetic equatorial plane and radius vector 'r'
        
        m = 1/2 (Bp.RStar)
           with: Bp: Strength of B at the star pole   
        
        Magnetic energy density in the equatorial plane of the “inner” 
        magnetosphere; strength of a dipolar magnetic field:
        B = 1/2 (Bp/RStar)³
        
        Dipole Magnetic Field components in the magnetic field frame:
         Bx = 3m xz/r^5
         By = 3m yz/r^5
         Bz = m(3z^2/r^5 - 1/r^3)
        """

        # x, y, z different positions in the LoS coordinates
        self.x_ = np.linspace(
            -self.L / 2 * self.R_obj, self.L / 2 * self.R_obj, self.n)
        self.y_ = np.linspace(
            -self.L / 2 * self.R_obj, self.L / 2 * self.R_obj, self.n)
        self.z_ = np.linspace(
            -self.L / 2 * self.R_obj, self.L / 2 * self.R_obj, self.n)
        self.x_pplot = np.linspace(-self.L / 2, self.L / 2, self.n)
        self.y_pplot = np.linspace(-self.L / 2, self.L / 2, self.n)
        self.z_pplot = np.linspace(-self.L / 2, self.L / 2, self.n)

        # Coordinates Y'Z' in plane perpendicular to the LoS (x')
        self.coordinates_yz = []
        for y in self.y_pplot:
            for z in self.z_pplot:
                self.coordinates_yz.append([y, z])

        # LoS_rays of voxels along the LoS. Each ray passing through a
        # specific Y'Z' coordinate in the plane perpendicular to the LoS
        self.LoS_rays = []

        # Plotting canvas
        self.plot3d = plot3d
        if self.plot3d:
            self.fig = plt.figure(figsize=(10, 7))
            min_lim_axis = -18
            max_lim_axis = 18
            self.ax = self.fig.add_subplot(121, projection='3d')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_zlabel('z')
            self.ax.set_xlim3d(min_lim_axis, max_lim_axis)
            self.ax.set_ylim3d(min_lim_axis, max_lim_axis)
            self.ax.set_zlim3d(min_lim_axis, max_lim_axis)

            self.ax2 = self.fig.add_subplot(122, projection='3d')
            self.ax2.set_xlim3d(min_lim_axis, max_lim_axis)
            self.ax2.set_ylim3d(min_lim_axis, max_lim_axis)
            self.ax2.set_zlim3d(min_lim_axis, max_lim_axis)

            # Plot (sub)stellar object rotation axis
            self.plot_axis(rotation_matrix=self.R3, color="darkorange",
                           len_axis=self.len_axes)

            # Plot the (sub)stellar dipole magnetic axis
            self.plot_axis(rotation_matrix=self.R, color="darkblue",
                           len_axis=self.len_axes)

    def coordinate_system_computation(self, coordinate_system_id="LoS"):
        """
        Compute Coordinate Systems: computes the values of the vectors
        forming a given coordinate system from the four different
        coordinate systems used in the model, expressed in the coordinates
        of the Line of Sight [LoS] coordinate system
        :param coordinate_system_id:
        :return coordinate_system:
        """
        coordinate_system_init_x = [1, 0, 0]
        coordinate_system_init_y = [0, 1, 0]
        coordinate_system_init_z = [0, 0, 1]

        if coordinate_system_id == "magnetic_field":
            rotation_matrix = self.R
        elif coordinate_system_id == "rotation_axis":
            rotation_matrix = self.R3.dot(self.R2)
        elif coordinate_system_id == "ucd_rotated":
            rotation_matrix = self.R3
        elif coordinate_system_id == "LoS":
            rotation_matrix = np.identity(3)
        else:
            raise "Choose a valid Coordinate System"

        coordinate_system_x = rotation_matrix * coordinate_system_init_x
        coordinate_system_y = rotation_matrix * coordinate_system_init_y
        coordinate_system_z = rotation_matrix * coordinate_system_init_z
        coordinate_system = [coordinate_system_x,
                             coordinate_system_y,
                             coordinate_system_z]
        return coordinate_system

    def display_coordinate_system(self, plot, origin_point, coordinate_system,
                                  scale=1, label="LoS"):
        """
        Coordinates System Display
        In general: [x: red; y: green; z: blue]

        :param plot:
        :param origin_point:
        :param coordinate_system:
        :param scale:
        :param label:
        :return:
        """
        plot.quiver(
            origin_point[0], origin_point[1], origin_point[2],
            scale * coordinate_system[0][0],
            scale * coordinate_system[0][1],
            scale * coordinate_system[0][2],
            color="red")
        plot.quiver(
            origin_point[0], origin_point[1], origin_point[2],
            scale * coordinate_system[1][0],
            scale * coordinate_system[1][1],
            scale * coordinate_system[1][2],
            color="green")
        plot.quiver(
            origin_point[0], origin_point[1], origin_point[2],
            scale * coordinate_system[2][0],
            scale * coordinate_system[2][1],
            scale * coordinate_system[2][2],
            color="blue")
        plot.text(origin_point[0], origin_point[1], origin_point[2] + 5, label)

    ###########################################################################
    # Points of the LoS cube, expressed in LoS coordinates
    def LoS_cube(self):
        """
        Points of the LoS (Line of Sight) cube, expressed in the LoS
        coordinates; converted grid to array of 3D vectors. Each vector
        contains the geometric coordinates of a 3D point in the grid.
        Grid in the orientation and coordinates of the LoS (x', y', z').
        The coordinates of each vector represents the center of each of the
        voxels of the grid.

        :returns:
            - points_LoS
            - points_LoS_plot
            - points_LoS_in_B
        """

        x, y, z = np.meshgrid(self.x_, self.y_, self.z_)
        # Grid for plotting in LoS coordinates (x_plot being the line of sight)
        x_plot, y_plot, z_plot = np.meshgrid(
            self.x_pplot, self.y_pplot, self.z_pplot)
        # Points of the grid in the Line of Sight (LoS) coordinates.
        points_LoS = []
        # Array for plotting in units of (sub)stellar radius R_obj
        points_LoS_plot = []
        for i in range(0, self.n):
            for j in range(0, self.n):
                for k in range(0, self.n):
                    points_LoS.append(
                        np.array([x[i, j, k], y[i, j, k], z[i, j, k]]))
                    points_LoS_plot.append(np.array([x_plot[i, j, k],
                                                     y_plot[i, j, k],
                                                     z_plot[i, j, k]]))

        # points_LoS_in_B: Points of the Line of Sight cube (LoS coordinates),
        #   expressed in the magnetic coordinates
        points_LoS_in_B = []
        for point_LoS in points_LoS:
            point_LoS_in_B = self.R_inv.dot(point_LoS)
            points_LoS_in_B.append(point_LoS_in_B)

        return points_LoS, points_LoS_plot, points_LoS_in_B

    def magnetic_field_vectors_LoS(self, points_LoS_plot, points_LoS_in_B):
        """
        Magnetic field vectors B in each point of the grid in Line of Sight
        [LoS] coordinates (each point representing the center of each voxel)

        :param points_LoS_plot:
        :param points_LoS_in_B:
        :returns:
            - Bs_LoS
        """

        # B = [Bx, By, Bz], in the LoS coordinates system frame:
        #  in the points given by the grid of the LoS cube
        Bs_LoS = []
        for i in range(len(points_LoS_plot)):
            # point_LoS in units of sub(stellar) radius
            point_LoS = points_LoS_plot[i]
            point_LoS_in_B = points_LoS_in_B[i]
            x = point_LoS_in_B[0]
            y = point_LoS_in_B[1]
            z = point_LoS_in_B[2]
            if x == 0 and y == 0 and z == 0:
                continue
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            # Compute Bx, By, Bz in the points of the grid of the LoS cube
            #   expressed in magnetic coordinates x, y, z
            Bx = 3 * self.m * (x * z / r ** 5)
            By = 3 * self.m * (y * z / r ** 5)
            Bz = self.m * (3 * z ** 2 / r ** 5 - 1 / r ** 3)
            # Magnetic Field Vector B in Magnetic Coordinates System (x, y, z)
            B = np.array([Bx, By, Bz])
            # Magnetic Field Vector B in LoS Coordinates System (x', y', z')
            B_LoS = self.R.dot(B)
            Bs_LoS.append(B_LoS)
            voxel = Voxel(B_LoS, self.voxel_len,
                          position_LoS_plot=point_LoS,
                          position_in_B=point_LoS_in_B,
                          δ=self.δ, Ne=self.Ne)
            self.voxels.append(voxel)
        return Bs_LoS

    def plot_B_LoS_unit_vectors(self, Bs_LoS, points_LoS_plot):
        """
        Plot magnetic unit vectors in LoS coordinates:
        - Compute unit vectors from magnetic field vectors B in the LoS
          coordinates system
        - Take all points except the origin of coordinates, which is the
          center of the (sub)stellar object
        - Plot magnetic (Bs) unit vectors

        :param Bs_LoS:
        :param points_LoS_plot:
        :return Bs_LoS_unit:
        """
        Bs_LoS_unit = []
        for B_LoS in Bs_LoS:
            B_LoS_unit = B_LoS / abs(np.linalg.norm(B_LoS))
            for i in [0, 1, 2]:
                if abs(B_LoS_unit[i]) < 0.01:
                    B_LoS_unit[i] = 0
            Bs_LoS_unit.append([round(B_LoS_unit[0], 3),
                                round(B_LoS_unit[1], 3),
                                round(B_LoS_unit[2], 3)])

        # Remove origin point
        points_LoS_plot_no_origin = []
        for point_LoS_plot in points_LoS_plot:
            if point_LoS_plot.any():
                points_LoS_plot_no_origin.append(point_LoS_plot)

        # Plot scaled unit vectors
        for i in range(len(points_LoS_plot_no_origin)):
            point_LoS = points_LoS_plot_no_origin[i]
            B_LoS_unit = Bs_LoS_unit[i]

            # Grid points
            x_plot = point_LoS[0]
            y_plot = point_LoS[1]
            z_plot = point_LoS[2]
            # Do not show grid points, only vectors
            # ax.scatter(x_plot, y_plot, z_plot, color='b')

            # B vectors in each grid point
            scale_factor = 3
            self.ax.quiver(
                x_plot, y_plot, z_plot,
                scale_factor * B_LoS_unit[0],
                scale_factor * B_LoS_unit[1],
                scale_factor * B_LoS_unit[2])

        return Bs_LoS_unit

    def on_move(self, event):
        """Plotting: Link both subplots"""
        if event.inaxes == self.ax:
            if self.ax.button_pressed in self.ax._rotate_btn:
                self.ax2.view_init(elev=self.ax.elev, azim=self.ax.azim)
            elif self.ax.button_pressed in self.ax._zoom_btn:
                self.ax2.set_xlim3d(self.ax.get_xlim3d())
                self.ax2.set_ylim3d(self.ax.get_ylim3d())
                self.ax2.set_zlim3d(self.ax.get_zlim3d())
        elif event.inaxes == self.ax2:
            if self.ax2.button_pressed in self.ax2._rotate_btn:
                self.ax.view_init(elev=self.ax2.elev, azim=self.ax2.azim)
            elif self.ax2.button_pressed in self.ax2._zoom_btn:
                self.ax.set_xlim3d(self.ax2.get_xlim3d())
                self.ax.set_ylim3d(self.ax2.get_ylim3d())
                self.ax.set_zlim3d(self.ax2.get_zlim3d())
        else:
            return
        self.fig.canvas.draw_idle()

    def plot_axis(self, rotation_matrix, color="blue", len_axis=20):
        """
        Plot (sub)stellar object axes
        :param rotation_matrix:
        :param color:
        :param len_axis:
        :return:
        """
        axis_point1 = [0, 0, -len_axis / 2]
        axis_point2 = [0, 0, len_axis / 2]
        axis_point1_LoS = rotation_matrix.dot(axis_point1)
        axis_point2_LoS = rotation_matrix.dot(axis_point2)
        line_x, line_y, line_z = (
            [axis_point2_LoS[0], axis_point1_LoS[0]],
            [axis_point2_LoS[1], axis_point1_LoS[1]],
            [axis_point2_LoS[2], axis_point1_LoS[2]])
        self.ax.plot(line_x, line_y, line_z, color=color)

    def ucd_compute_and_plot(self, points_LoS_in_B, points_LoS_plot):
        """
        UCD compute and plot
        :param points_LoS_in_B:
        :param points_LoS_plot:
        :param plot:
        """
        if self.plot3d:
            # Compute and plot magnetic field vectors
            ###################################################################
            # Plot (sub)stellar object coordinates systems
            scale_axis = 10
            # LoS coordinate system
            coord_system = self.coordinate_system_computation(
                coordinate_system_id="LoS")
            self.display_coordinate_system(
                plot=self.ax2, origin_point=[-10, -10, 0],
                coordinate_system=coord_system, scale=scale_axis, label="LoS")

            # (Sub)Stellar object rotated coordinate system
            coord_system = self.coordinate_system_computation(
                coordinate_system_id="ucd_rotated")
            self.display_coordinate_system(
                plot=self.ax2, origin_point=[-10, 10, 0],
                coordinate_system=coord_system, scale=scale_axis,
                label="Roted")

            # Rotation axis coordinate system
            coord_system = self.coordinate_system_computation(
                coordinate_system_id="rotation_axis")
            self.display_coordinate_system(
                plot=self.ax2, origin_point=[10, 10, 0],
                coordinate_system=coord_system, scale=scale_axis,
                label="Rotax")

            # Magnetic Field coordinate system
            coord_system = self.coordinate_system_computation(
                coordinate_system_id="magnetic_field")
            self.display_coordinate_system(
                plot=self.ax2, origin_point=[10, -10, 0],
                coordinate_system=coord_system, scale=scale_axis, label="Mag")

            ###################################################################
            # Plot LoS Grid, compute and plot magnetic field vectors
            Bs_LoS = self.magnetic_field_vectors_LoS(
                points_LoS_plot, points_LoS_in_B)
            self.plot_B_LoS_unit_vectors(Bs_LoS, points_LoS_plot)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
            plt.show()
        else:
            # Compute magnetic field vectors
            self.magnetic_field_vectors_LoS(points_LoS_plot, points_LoS_in_B)

    ###########################################################################
    # Find Middle-Magnetosphere
    def find_middle_magnetosphere(self):
        """
        Finding the points belonging to the inner, middle and outer
        magnetosphere. GyroSynchrotron Emission created in the
        middle-magnetosphere points
        - Points in the middle magnetosphere (between the inner and the outer
          magnetosphere): the ones with a clear contribution to the UCD radio
          emission arriving to the earth. The radio emission occurs in this
          zone, which contains the open magnetic field lines generating the
          current sheets: (Ra < r < Ra + l_mid)
          with l_mid: width of the middle magnetosphere
        - The radio emission in the inner magnetosphere is supposed to be
          self-absorbed by the UCD
        - In the outer magnetosphere the density of electrons decreases with
          the distance, which also lowers its contribution to the radio
          emission

        :return voxels_middlemag:
        """
        voxels_inner = []
        voxels_middlemag = []
        voxels_outer = []
        for i in range(len(self.voxels)):
            # We first find the angle λ (lam) associated with the specific
            # point of the LoS grid expressed in B coordinates. It is the angle
            # between the magnetic dipole "equatorial" plane and the radius
            # vector r of the point
            voxel = self.voxels[i]
            point_LoS_in_B = voxel.position_in_B
            if point_LoS_in_B[0] or point_LoS_in_B[1]:
                L_xy = np.sqrt(point_LoS_in_B[0]**2 + point_LoS_in_B[1]**2)
                L_z = point_LoS_in_B[2]
                lam = np.arctan(L_z/L_xy)
                L_xyz = np.sqrt(point_LoS_in_B[0]**2 +
                                point_LoS_in_B[1]**2 +
                                point_LoS_in_B[2]**2)
                # Using the equation of the dipole field lines
                # r = L_xyz = L cos²λ; with L in [Ra, Ra+l_mid]
                # We have the longitude 'r' (r = L_xyz) and λ of the specific
                # point of the grid that have been calculated in the B
                # coordinate system.
                # Equation of the field line touching the Alfvén Surface:
                # Ra * (np.cos(lam))**2
                r_min = self.Ra * (np.cos(lam))**2
                r_max = (self.Ra + self.l_mid) * (np.cos(lam))**2

                if L_xyz < r_min:
                    voxel.set_inner_mag(True)
                    voxel.set_Ne(0)
                if r_min <= L_xyz <= r_max:
                    voxel.set_middle_mag(True)
                    voxel.set_Ne(self.N_γ)
                    voxels_middlemag.append(voxel)
                if L_xyz > r_max:
                    voxel.set_outer_mag(True)
                    voxel.set_Ne(0)
        return voxels_middlemag

    def plot_middlemag_in_slices(self, voxels_middlemag, marker_size=2):
        """
        Show middle-magnetosphere in 2D slices perpendicular to the LoS
        ('x' axis)

        :param voxels_middlemag:
        :param marker_size:
        :return:
        """
        x_points_middlemag = []
        y_points_middlemag = []
        z_points_middlemag = []

        for voxel_middlemag in voxels_middlemag:
            x_points_middlemag.append(voxel_middlemag.position_LoS_plot[0])
            y_points_middlemag.append(voxel_middlemag.position_LoS_plot[1])
            z_points_middlemag.append(voxel_middlemag.position_LoS_plot[2])

        x_different_values_in_middlemag = []
        for x_point_middlemag in x_points_middlemag:
            if x_point_middlemag not in x_different_values_in_middlemag:
                x_different_values_in_middlemag.append(x_point_middlemag)
        x_different_values_in_middlemag = sorted(
            x_different_values_in_middlemag)

        # Middle-magnetosphere slices, to show the emitting points (points
        # which belong to the middle magnetosphere)
        plt.close()
        fig2 = plt.figure(figsize=(3, 3))
        for ii in range(len(x_different_values_in_middlemag)):
            y_slice = []
            z_slice = []
            x_middlemag = x_different_values_in_middlemag[ii]
            for jj in range(len(x_points_middlemag)):
                x_coord = x_points_middlemag[jj]
                if x_coord == x_middlemag:
                    y_slice.append(y_points_middlemag[jj])
                    z_slice.append(z_points_middlemag[jj])

            plt.plot(y_slice, z_slice, 'ro', markersize=marker_size)
            plt.show()

    def LoS_voxel_rays(self):
        """
        Define the LoS_Voxels_Ray objects, thanks to the voxels of the UCD.
        Each LoS_Voxels_Ray object contains the voxels in a given "ray"
        parallel to the LoS. All voxels of the same ray share the same
        coordinates Y'Z' of the plane perpendicular to the LoS (x')
        :return: LoS_rays
        """

        # NOTE: This organization of the voxels into rays along the LoS
        # is slow

        # As many rays as points present in the plane Y'Z' perpendicular to
        # the LoS (x')
        for coordinate_yz in self.coordinates_yz:
            y = coordinate_yz[0]
            z = coordinate_yz[1]
            ray = LoS_Voxels_Ray(y=y, z=z)
            for voxel in self.voxels:
                if (voxel.position_LoS_plot[1] == y
                        and voxel.position_LoS_plot[2] == z):
                    ray.LoS_voxels_in_ray.append(voxel)
            ray.set_number_voxels_in_ray(len(ray.LoS_voxels_in_ray))
            self.LoS_rays.append(ray)
        for LoS_ray in self.LoS_rays:
            LoS_ray.voxels_optical_depth()
            LoS_ray.compute_specific_intensity_ray()

        for i in range(self.n):
            column_list_intensities = []
            for j in range(self.n):
                column_list_intensities.append(
                    self.LoS_rays[i * self.n + j].ray_specific_intensity)
            self.specific_intensities_array[:, i] = column_list_intensities

    def plot_2D_specific_intensity_LoS(self):
        """
        Plot (2D) specific intensity in the plane perpendicular to the LoS
        at the specific rotation phase of the UCD (or other (sub)stellar)
        object
        """
        plt.figure(figsize=(3, 3))
        plt.imshow(self.specific_intensities_array, cmap='gray_r',
                   vmin=np.amin(self.specific_intensities_array),
                   vmax=np.amax(self.specific_intensities_array))
                   # vmin=0, vmax=255)
        plt.show()

    def compute_flux_density_LoS(self):
        total_flux_density_LoS = 0
        for LoS_ray in self.LoS_rays:
            total_flux_density_LoS += (
                LoS_ray.ray_specific_intensity * self.voxel_len**2)

        self.total_flux_density_LoS = 1/(self.D**2) * total_flux_density_LoS

    def plot_1D_flux_densities_rotation(self, rotation_phase, flux_densities):
        """Plot 1D of the Flux Density in the different rotation phases"""
        app = pg.mkQApp()
        pg.plot(rotation_phase, flux_densities, pen="b", symbol='o')
        app.exec_()

    def get_B_Ra(self):
        return self.B_Ra

