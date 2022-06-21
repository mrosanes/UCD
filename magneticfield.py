"""magneticfield.py
   The objectives of this file are:
    - Compute the magnetic vector field 'B' of a dipole of a
      (sub)stellar object at different points of a meshgrid, for an object
      with not aligned magnetic, rotation, and line of sight [LoS] axes.
    - Find all points of the grid inside the middle magnetosphere
    - Apply the Pipeline of C.Trigilio el al. (ESO 2004) [Appendix A]
      A&A 418, 593–605 (2004)
      DOI: 10.1051/0004-6361:20040060

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
"""

import numpy as np
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

# TODO: Questions
"""
Question Related with 'B' and with 'm'
Units of magnetic field in Tesla:
Shall distances be expressed in meters or in Star Radius, etc.?
"""

##############################################################################
# PIPELINE

"""
– Magnetosphere 3D sampling and magnetic field vectors B calculation (DONE)
– Definition of the Alfvén radius and of inner, middle and outer 
  magnetosphere (~DONE)

– Calculation of the number density ne of the non-thermal electrons in each 
point of the grid
– Calculation of emission and absorption coefficients
– Integration of the transfer equation along paths parallel to the 
line of sight
– Brightness distribution in the plane of the sky, total flux emitted 
toward the Earth
"""

##############################################################################
# Acronyms and Glossary
# LoS: Line of Sight

##############################################################################
# Parameters, constants and so on:
# Num points per edge in meshgrid cube (use odd number in order to get one
# point in the origin of coordinates [center of the grid and of the UCD]):
n = 5

# Total length of the meshgrid cube in number of (sub)stellar radius:
L = 18

# Jupyter Radius in meters [m] ~ Brown Dwarf Radius
Rj = 7e7

# UCD Radius
R_ucd = Rs = 1 * Rj

# Star Period of Rotation in days
Pr = 1

# Strength of the B at the pole of the star
Bp = 1  # in Tesla [T];  (1T = 1e4G)

# Magnetic Momentum:  m = 1/2 (Bp Rs)
m = 1/2 * Bp * R_ucd

# TODO: Alfvén radius [TO BE COMPUTED]
Ra = 5 * R_ucd

"""
Other Important Parameters:
r: Radius vector:
   Distance from the (sub)stellar object center till a concrete point 
   outside of the star (at the surface of the star: r = R_ucd) 
"""

##############################################################################
# Free Parameters of the Model (Parameter ranges indicated with: [x, y])

# l_mid (or 'l'): equatorial thickness of the magnetic shell for the
# middle magnetosphere (which is added to Ra)
l_mid = 4 * R_ucd
# l/rA: equatorial thickness of the magnetic shell in Alfvén Radius units
# l/rA: [0.025, 1];
eq_thick = l_mid/Ra
"""
– Ne: total number density of the non-thermal electrons: 
    Ne: [10^2 - 10^6 cm^(−3)], with Ne < n{e,A}
    with:
      n{e,A}: number density of thermal plasma at the Alfvén point
– δ: spectral index of the non-thermal electron energy distribution
    δ: [2, 4]
– Tp:  temperature of the inner magnetosphere
    Tp: [10^5 K, 10^7 K]
- np: number plasma density at the base of the inner magnetosphere (r = R∗)
    np: [10^7 –10^10 cm(−3)]
    with: 
      Tp.np.k{B} = p{ram} -> Tp.np = p{ram}/k{B}: [10^14 cgs – 10^15 cgs]
  (Tp and np, related with the plasma in the Star post-shock region: inside 
  the region protected by the Star closed magnetic field lines)       
– Rotation: if the star does not rotate, the thermal plasma density is 
    considered constant within the inner magnetosphere; if the star rotates, 
    the thermal plasma density decreases linearly outward, while the 
    temperature increases. Tp and np are considered as the values 
    at r = R∗ (base of the inner magnetosphere)
"""

##############################################################################
# Angles

# Expressed in degrees:
# Angle from rotation to magnetic axis: [~0º - ~180º]
beta = 3  # 5
# UCD star rotation [~0º - ~360º]
phi = rotation = 1  # 5
# Rotation Axis inclination measured from the Line of Sight: [~0º - ~180º]
# Information about rotation axis orientations:
#   . Orbits with the rotation axis in the plane of the sky (~90º)
#   . Orbits with the rotation axis towards the LoS (~0º)
inc = inclination = 89

# Transformed to radians:
b_r = b_rad = np.deg2rad(beta)
p_r = p_rad = np.deg2rad(phi)
i_r = i_rad = np.deg2rad(inc)

##############################################################################
# Formulas

"""
Ram pressure:
p{ram} = ρ.v^2

p{ram} = B^2 / (8*PI) ~ np Tp K{B}

v(r) = v(inf).(1 - R_ucd/r)
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

Magnetic energy density in the equatorial plane of the “inner” magnetosphere;
strength of a dipolar magnetic field:
B = 1/2 (Bp/RStar)³

Dipole Magnetic Field components in the magnetic field frame:
 Bx = 3m xz/r^5
 By = 3m yz/r^5
 Bz = m(3z^2/r^5 - 1/r^3)
"""

##############################################################################
# Points of the LoS (Line of Sight) cube, expressed in the LoS coordinates
# Converted grid to array of 3D vectors. Each vector contains the geometric
# coordinates of a 3D point in the grid. Grid in the orientation and
# coordinates of the LoS (x', y', z'). The coordinates of each vector
# represents the center of each of the voxels of the grid.

x_ = np.linspace(-L/2 * R_ucd, L/2 * R_ucd, n)
y_ = np.linspace(-L/2 * R_ucd, L/2 * R_ucd, n)
z_ = np.linspace(-L/2 * R_ucd, L/2 * R_ucd, n)
x_pplot = np.linspace(-L/2, L/2, n)
y_pplot = np.linspace(-L/2, L/2, n)
z_pplot = np.linspace(-L/2, L/2, n)

x, y, z = np.meshgrid(x_, y_, z_)
# Grid for plotting in LoS coordinates (x_plot being the line of sight)
x_plot, y_plot, z_plot = np.meshgrid(x_pplot, y_pplot, z_pplot)
# Points of the grid in the Line of Sight (LoS) coordinates.
points_LoS = []
# Array for plotting in units of (sub)stellar radius R_ucd
points_LoS_plot = []
for i in range(0, n):
    for j in range(0, n):
        for k in range(0, n):
            points_LoS.append(np.array([x[i, j, k], y[i, j, k], z[i, j, k]]))
            points_LoS_plot.append(np.array([x_plot[i, j, k],
                                             y_plot[i, j, k],
                                             z_plot[i, j, k]]))

# pp.pprint(points_LoS)

###############################################################################
# Trigonometric functions of the used angles
sin_b = np.round(np.sin(b_r), 4)
cos_b = np.round(np.cos(b_r), 4)

sin_p = np.round(np.sin(p_r), 4)
cos_p = np.round(np.cos(p_r), 4)

sin_i = np.round(np.sin(i_r), 4)
cos_i = np.round(np.cos(i_r), 4)

###############################################################################
# Rotation Matrices

# Beta (b): angle from magnetic axis to rotation axis (b = beta)
R1 = np.array([[ cos_b,  0,  sin_b  ],  # - sin_b],
               [ 0,      1,  0      ],
               [-sin_b,  0,  cos_b  ]])

# Beta (b): angle from magnetic axis to rotation axis (b = beta)
R1 = np.array([[ cos_b,  0,  sin_b  ],  # - sin_b],
               [ 0,      1,  0      ],
               [-sin_b,  0,  cos_b  ]])

# p (phi): Rotation of the star angle (p = phi = rot)
R2 = np.array([[cos_p, -sin_p, 0],
               [sin_p, cos_p,  0],
               [0,     0,      1]])

# i (inc): Inclination of the orbit, from line of sight to rotation axis
R3 = np.array([[ sin_i,  0,  cos_i],
               [ 0,      1,  0    ],
               [-cos_i,  0,  sin_i]])

# Rotation Matrices for each degree of freedom
R1_inv = R1.transpose()
R2_inv = R2.transpose()
R3_inv = R3.transpose()

"""
Information about "complete" Rotation Matrices R and R_inv:

R = R3 . R2 . R1 to go from points expressed in (x, y, z), to points
   expressed in (x', y', z')

R^(-1) = R1^(-1) . R2^(-1) . R3^(-1) to go from points expressed
   in (x', y', z'), to points expressed in (x, y, z)
"""
# Rotation Matrices (Complete Rotation)
R = R3.dot(R2).dot(R1)
R_inv = R1_inv.dot(R2_inv).dot(R3_inv)


###############################################################################
def coordinate_system_computation(coordinate_system_id="LoS"):
    """ Compute Coordinate Systems: computes the values of the vectors
    forming a given coordinate system from the four different
    coordinate systems used in the model, expressed in the coordinates
    of the Line of Sight [LoS] coordinate system"""
    coordinate_system_init_x = [1, 0, 0]
    coordinate_system_init_y = [0, 1, 0]
    coordinate_system_init_z = [0, 0, 1]

    if coordinate_system_id == "magnetic_field":
        rotation_matrix = R
    elif coordinate_system_id == "rotation_axis":
        rotation_matrix = R3.dot(R2)
    elif coordinate_system_id == "ucd_rotated":
        rotation_matrix = R3
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


def display_coordinate_system(plot, origin_point, coordinate_system, scale=1):
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
        scale * coordinate_system[2][2],
        color="green")
    plot.quiver(
        origin_point[0], origin_point[1], origin_point[2],
        scale * coordinate_system[2][0],
        scale * coordinate_system[2][1],
        scale * coordinate_system[2][2],
        color="blue")


###############################################################################
# Magnetic field vectors B in each point of the grid in Line of Sight
# [LoS] coordinates (each point representing the center of each voxel)

# 1:
# points_LoS_in_B: Points of the Line of Sight cube (LoS coordinates),
#   expressed in the magnetic coordinates
points_LoS_in_B = []
for point_LoS in points_LoS:
    point_LoS_in_B = R_inv.dot(point_LoS)
    points_LoS_in_B.append(point_LoS_in_B)

# 2:
# B = [Bx, By, Bz], in the LoS coordinates system frame:
#  in the points given by the grid of the LoS cube
Bs_LoS = []
for point_LoS_in_B in points_LoS_in_B:
    x = point_LoS_in_B[0]
    y = point_LoS_in_B[1]
    z = point_LoS_in_B[2]
    if x == 0 and y == 0 and z == 0:
        continue
    r = np.sqrt(x**2 + y**2 + z**2)
    # Compute Bx, By, Bz in the points of the grid given by the LoS cube
    #   expressed in magnetic coordinates x, y, z
    Bx = 3*m * (x*z/r**5)
    By = 3*m * (y*z/r**5)
    Bz = m * (3*z**2/r**5 - 1/r**3)
    # Magnetic Field Vector B in Magnetic Coordinates System (x, y, z)
    B = np.array([Bx, By, Bz])
    # Magnetic Field Vector B in the LoS Coordinates System (x', y', z')
    B_LoS = R.dot(B)
    Bs_LoS.append(B_LoS)

# pp.pprint(Bs_LoS)

###############################################################################
# Plotting

# For plotting, make unit vectors from magnetic field vectors B in
#  the LoS coordinates system

Bs_LoS_unit = []
for B_LoS in Bs_LoS:
    B_LoS_unit = B_LoS / abs(np.linalg.norm(B_LoS))
    for i in [0, 1, 2]:
        if abs(B_LoS_unit[i]) < 0.01:
            B_LoS_unit[i] = 0
    Bs_LoS_unit.append([round(B_LoS_unit[0], 3),
                        round(B_LoS_unit[1], 3),
                        round(B_LoS_unit[2], 3)])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-18, 18)
ax.set_ylim3d(-18, 18)
ax.set_zlim3d(-18, 25)

# For plotting, take all points except the origin of coordinates, which is the
# center of the (sub)stellar object
points_LoS_plot_no_origin = []
for point_LoS_plot in points_LoS_plot:
    if point_LoS_plot.any():
        points_LoS_plot_no_origin.append(point_LoS_plot)

for i in range(len(points_LoS_plot_no_origin)):
    point_LoS = points_LoS_plot_no_origin[i]
    B_LoS_unit = Bs_LoS_unit[i]

    # Grid points
    x_plot = point_LoS[0]
    y_plot = point_LoS[1]
    z_plot = point_LoS[2]
    # ax.scatter(x_plot, y_plot, z_plot, color='b')

    # B vectors in each grid point
    u = round(B_LoS_unit[0], 3)
    v = round(B_LoS_unit[1], 3)
    w = round(B_LoS_unit[2], 3)
    scale_factor = 3
    ax.quiver(x_plot, y_plot, z_plot,
              scale_factor*u, scale_factor*v, scale_factor*w)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


###############################################################################
# Plot (sub)stellar object axes
def plot_axis(rotation_matrix=R, color="blue", len_axis=20):
    # global line_x, line_y, line_z
    axis_point1 = [0, 0, -len_axis / 2]
    axis_point2 = [0, 0, len_axis / 2]
    axis_point1_LoS = rotation_matrix.dot(axis_point1)
    axis_point2_LoS = rotation_matrix.dot(axis_point2)
    line_x, line_y, line_z = (
        [axis_point2_LoS[0], axis_point1_LoS[0]],
        [axis_point2_LoS[1], axis_point1_LoS[1]],
        [axis_point2_LoS[2], axis_point1_LoS[2]])
    ax.plot(line_x, line_y, line_z, color=color)


# Plot (sub)stellar object rotation axis
plot_axis(rotation_matrix=R3, color="purple")

# Plot the (sub)stellar dipole magnetic axis
plot_axis(rotation_matrix=R, color="orange")

# LoS coordinate system
coord_system = coordinate_system_computation(
    coordinate_system_id="magnetic_field")
display_coordinate_system(plot=ax, origin_point=[-7.5, 0, 20],
                          coordinate_system=coord_system, scale=3)

# Rotation axis coordinate system
coord_system = coordinate_system_computation(
    coordinate_system_id="rotation_axis")
display_coordinate_system(plot=ax, origin_point=[-2.5, 0, 20],
                          coordinate_system=coord_system, scale=3)

# (Sub)Stellar object rotated coordinate system
coord_system = coordinate_system_computation(
    coordinate_system_id="ucd_rotated")
display_coordinate_system(plot=ax, origin_point=[2.5, 0, 20],
                          coordinate_system=coord_system, scale=3)

# Magnetic Field coordinate system
coord_system = coordinate_system_computation(
    coordinate_system_id="LoS")
display_coordinate_system(plot=ax, origin_point=[7.5, 0, 20],
                          coordinate_system=coord_system, scale=3)
###############################################################################
"""
Finding the points belonging to the middle magnetosphere
- Points in the middle magnetosphere (between the inner and the outer
  magnetosphere): the ones with a clear contribution to the UCD radio emission
  arriving to the earth. The radio emission occurs in this zone, which contains
  the open magnetic field lines that generating the current sheets
  (Ra < r < Ra + l_mid)
  with l_mid: width of the middle magnetosphere
- The radio emission in the inner magnetosphere is supposed to be
  self-absorbed by the UCD
- In the outer magnetosphere the density of electrons decreases with the
  distance, which also lowers its contribution to the radio emission
"""

points_grid_middle_magnetosphere = []
for i in range(len(points_LoS_in_B)):
    # We first find the angle λ (lam) associated with the specific point of
    # the LoS grid expressed in B coordinates. It is the angle between the
    # magnetic dipole "equatorial" plane and the radius vector r of the point
    point_LoS_in_B = points_LoS_in_B[i]
    if point_LoS_in_B[0] or point_LoS_in_B[1]:
        L_xy = np.sqrt(point_LoS_in_B[0]**2 + point_LoS_in_B[1]**2)
        L_z = point_LoS_in_B[2]
        lam = np.arctan(L_z/L_xy)

        # Now by using the equation of the dipole field lines
        # r = L cos²λ  (with L_xyz being L)
        L_xyz = np.sqrt(point_LoS_in_B[0]**2 +
                        point_LoS_in_B[1]**2 +
                        point_LoS_in_B[2]**2)
        # We have the longitude 'r' and λ of the specific point of the grid
        # that have been calculated in the B coordinate system:
        r = L_xyz * (np.cos(lam))**2

        # Equation of the field line touching the Alvén Surface:
        # Ra * (np.cos(lam))**2
        r_min = Ra  # Emitting points are located outside the Alfvén Surface
        r_max = (Ra + l_mid) * (np.cos(lam))**2
        if r_min < r < r_max:
            point_grid_middle_magnetosphere = (
                points_LoS_plot[i], point_LoS_in_B, lam)
            points_grid_middle_magnetosphere.append(
                point_grid_middle_magnetosphere)

            x_middlemag_plot = points_LoS_plot[i][0]
            y_middlemag_plot = points_LoS_plot[i][1]
            z_middlemag_plot = points_LoS_plot[i][2]
            ax.scatter(x_middlemag_plot, y_middlemag_plot, z_middlemag_plot,
                       color='r')

            """
            # Verification that the length 'r' of each specific point in the 
            # middle magnetosphere is between (Ra < r < Ra + l_mid), and that
            # the distance to the found points is the same regardless
            # of the system of coordinates in which they are expressed
            print(np.sqrt(point_LoS_in_B[0]**2
                          + point_LoS_in_B[1]**2
                          + point_LoS_in_B[2]**2
                          ))
            print(np.sqrt(point[0] ** 2
                          + point[1] ** 2
                          + point[2] ** 2
                          ))
            """
plt.show()


