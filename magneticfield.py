import numpy as np
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

"""
The GOAL of this file is to find all points of the grid inside the middle 
magnetosphere
"""

"""
PIPELINE
– Magnetosphere 3D sampling and magnetic field vectors B calculation
– Definition of the Alfvén radius and of inner, middle and outer magnetosphere
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
# Constants:

# Jupyter Radius in meters [m] ~ Brown Dwarf Radius
Rj = 7e7

# UCD Radius
R_ucd = Rs = 1*Rj

# Star Period of Rotation in days
Pr = 1

# Total length of the cube:
# L = 20 * R_ucd
L = 20  # * R_ucd

# Strength of the B at the pole of the star
Bp = 1  # in Tesla [T];  (1T = 1e4G)

# Magnetic Momentum:  m = 1/2 (Bp Rs)
m = 1/2 * Bp * R_ucd

"""
Notes about magnetic field
Units of magnetic field in Gauss:
Question: Shall I use the distances in meters or in Star Radius?
"""
##############################################################################
# Important Parameters

""" 
Pr: Period of Rotation:
  Pr ~ 1 day

r: Radius vector:
   Distance from the star center till a concrete point outside of the star:
   (At the surface of the star: r = R_ucd) 
"""

##############################################################################
# Formulas

"""
# Ram pressure:
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

# Magnetic energy density in the equatorial plane of the “inner” magnetosphere;
  strength of a dipolar magnetic field:
B = 1/2 (Bp/RStar)³
"""

# Dipole Magnetic Field Definitions in the magnetic field frame:
# Bx = 3m xz/r^5
# By = 3m yz/r^5
# Bz = m(3z^2/r^5 - 1/r^3)

##############################################################################
# Free Parameters of the Model (Parameter ranges indicated with: [x, y])

"""
– l: equatorial thickness of the magnetic shell 
– l/rA: equatorial thickness of the magnetic shell in Alfvén Radius units 
    l/rA: [0.025, 1];    
– Ne: total number density of the non-thermal electrons: 
    Ne: [10^2 - 10^6 cm^(−3)], with Ne < n{e,A}
    with:
      n{e,A}: number density of thermal plasma at the Alfvén point
– δ: spectral index of the non-thermal electron energy distribution
    δ: [2, 4]
– Tp:  temperature of the inner magnetosphere
    Tp: [10^5 K, 10^7 K]
- np: number plasma density at the base of the inner magnetosphere 
    np: [10^7 –10^10 cm(−3)]
    with: 
      Tp.np.k{B} = p{ram} -> Tp.np = p{ram}/k{B}: [10^14 cgs – 10^15 cgs]
  (Tp and np, related with the plasma in the Star post-shock region: inside 
  the region protected by the Star closed magnetic field lines)       
– Rotation: if the star does not rotate, the thermal plasma density is 
    considered constant within the inner magnetosphere; if the star rotates, 
    the thermal plasma density decreases linearly outward, while the 
    temperature increases. Tp and np are considered as the values at r = R∗
"""

##############################################################################
# Angles

# Expressed in degrees:
# Angle from rotation to magnetic axis: [~0º - ~180º]
beta = 15  # 5
# UCD star rotation [~0º - ~360º]
phi = rotation = 90  # 5
# Rotation Axis inclination measured from the Line of Sight: [~0º - ~180º]
# Notes:
#   . Orbits with the rotation axis in the plane of the sky (~90º)
#   . Orbits with the rotation axis towards the LoS (~0º)
inc = inclination = 89

# Transformed to radians:
b_r = b_rad = np.deg2rad(beta)
p_r = p_rad = np.deg2rad(phi)
i_r = i_rad = np.deg2rad(inc)

##############################################################################
# Points of the LoS (Line of Sight) cube, expressed in the LoS coordinates
# Converted grid to array of 3D vectors. Each vector contains the geometric
# coordinates of a 3D point in the grid. Grid in the orientation and
# coordinates of the LoS (x', y', z'). The coordinates of each vector
# represents the center of each of the voxels of the grid.

# Num points in edge (use odd number in order to get one dot in the center
# of the grid: center of the UCD):
n = 5
x_ = np.linspace(-L/2, L/2, n)
y_ = np.linspace(-L/2, L/2, n)
z_ = np.linspace(-L/2, L/2, n)

x, y, z = np.meshgrid(x_, y_, z_)
# Points of the grid in the Line of Sight (LoS) coordinates.
points_LoS = []
for i in range(0, n):
    for j in range(0, n):
        for k in range(0, n):
            points_LoS.append(np.array([x[i, j, k], y[i, j, k], z[i, j, k]]))

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
R1 = np.array([[ cos_b,  0,  - sin_b],  # sin_b],
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
Notes about "complete" Rotation Matrices R and R_inv:

R = R3 . R2 . R1 to go from points expressed in (x, y, z), to points
   expressed in (x', y', z')

R^(-1) = R1^(-1) . R2^(-1) . R3^(-1) to go from points expressed
   in (x', y', z'), to points expressed in (x, y, z)
"""
# Rotation Matrices (Complete Rotation)
R = R3.dot(R2).dot(R1)
R_inv = R1_inv.dot(R2_inv).dot(R3_inv)

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
# Make unit vectors from magnetic field vectors B in the LoS coordinates system
Bs_LoS_unit = []
for B_LoS in Bs_LoS:
    B_LoS_unit = B_LoS / abs(np.linalg.norm(B_LoS))
    for i in [0, 1, 2]:
        if abs(B_LoS_unit[i]) < 0.01:
            B_LoS_unit[i] = 0
    Bs_LoS_unit.append([round(B_LoS_unit[0], 3),
                        round(B_LoS_unit[1], 3),
                        round(B_LoS_unit[2], 3)])

###############################################################################
# Plotting
# UNCOMMENT AFTER FINDING MIDDLE MAGNETOSPHERE
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

points_LoS_no_null = []
for point_LoS in points_LoS:
    if point_LoS.any():
        points_LoS_no_null.append(point_LoS)

for i in range(len(points_LoS_no_null)):
    point_LoS = points_LoS_no_null[i]
    B_LoS_unit = Bs_LoS_unit[i]

    # Grid points
    x = point_LoS[0]
    y = point_LoS[1]
    z = point_LoS[2]
    ax.scatter(x, y, z, color='b')

    # B vectors in each grid point
    u = round(B_LoS_unit[0], 3)
    v = round(B_LoS_unit[1], 3)
    w = round(B_LoS_unit[2], 3)
    scale_factor = 3
    ax.quiver(x, y, z,
              scale_factor*u, scale_factor*v, scale_factor*w)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

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
Ra = 5
# l for the middle magnetosphere added to Ra
l_mid = 4

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
                point_LoS, point_LoS_in_B, lam)
            points_grid_middle_magnetosphere.append(
                point_grid_middle_magnetosphere)

            x = point_LoS_in_B[0]
            y = point_LoS_in_B[1]
            z = point_LoS_in_B[2]
            ax.scatter(x, y, z, color='r')

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


