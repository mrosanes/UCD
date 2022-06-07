import numpy as np
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

"""
PIPELINE
– 3D sampling of the magnetosphere and calculation of the magnetic 
field vector B;
– Definition of the Alfvén radius and of inner, middle and outer magnetosphere;
– Calculation of the number density ne of the non-thermal electrons in each 
point of the grid;
– Calculation of emission and absorption coeﬃcients;
– Integration of the transfer equation along paths parallel to the 
line of sight;
– Brightness distribution in the plane of the sky, total flux emitted 
toward the Earth.
"""

##############################################################################
# Acronyms and Glossary
# LoS: Line of Sight

##############################################################################
# Important Parameters
""" 
r: Radius Vector:
   Distance from the star center till a concrete point outside of the star:
   (At the surface of the star: r = Rstar) 
"""

##############################################################################
# Formulas

"""
# Ram pressure:
p{ram} = ρ.v^2

p{ram} = B^2 / (8*PI) ~ np Tp K{B}

v(r) = v(inf).(1 - Rstar/r)
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
"""
##############################################################################
# Free Parameters of the Model (Parameter ranges indicated with: [x, y])

"""
– l: equatorial thickness of the magnetic shell 
– l/rA: equatorial thickness of the magnetic shell in Alfvén Radius units 
    l/rA: [0.025, 1];    
– Ne: total number density of the non-thermal electrons: 
    Ne: [10^2, 10^6 cm^(−3)], with Ne < n{e,A}
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
# Constants used along the script:

# Jupyter Radius in meters [m] ~ Brown Dwarf Radius
Rj = 7e7

# UCD Radius
R = R_ucd = 1*Rj

"""
Notes about magnetic field
Units of magnetic field in Gauss: 
Question: Shall I use the distances in meters or in Star Radiuses?
"""
# Bp in Gauss [G] ; Strength of the B at the pole of the star
Bp = 1e4

# Dipole Magnetic Field Definitions in the magnetic field frame:
# Bx = 3m xz/r^5
# By = 3m yz/r^5
# Bz = m(3z^2/r^5 - 1/r^3)

# With Magnetic Momentum:  m = 1/2 (Bp Rs)
m = 1/2 * Bp * R_ucd
# print(m)

# Total length of the cube:
# L = 20 * R_ucd
L = 20

# Angles in degrees:
beta = 5  # Angle from rotation to magnetic axis
phi = rotation = 5  # UCD star rotation
inc = inclination = 84  # Orbit inclination measured from the Line of Sight
# Note: orbits with the rotation axis in the plane of the sky, does not modify
# the coordinates system

# Angles to radians:
b_r = b_rad = np.deg2rad(beta)
p_r = p_rad = np.deg2rad(phi)
i_r = i_rad = np.deg2rad(inc)

##############################################################################
# Vectors of the LoS (Line of Sight) cube, expressed in the LoS coordinates
# Converted grid to array of 3D vectors. Each vector contains the geometric
# coordinates of a 3D point in the grid. Grid in the orientation and
# coordinates of the LoS (x', y', z'). The coordinates of each vector
# represents the center of each of the voxels of the grid.

# Num points in edge (use odd number in order to get one dot in the center
# of the grid: center of the UCD):
n = 7
x_ = np.linspace(-L/2, L/2, n)
y_ = np.linspace(-L/2, L/2, n)
z_ = np.linspace(-L/2, L/2, n)

x, y, z = np.meshgrid(x_, y_, z_)
vectors = []

for i in range(0, n):
    for j in range(0, n):
        for k in range(0, n):
            vectors.append([x[i, j, k], y[i, j, k], z[i, j, k]])

# pp.pprint(vectors)

"""
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for vector in vectors:
    ax.scatter(vector[0], vector[1], vector[2], color='b')   
plt.show()
"""

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
R1 = np.array([[ cos_b,  0,  sin_b],
               [ 0,      1,  0    ],
               [-sin_b,  0,  cos_b]])

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

R = R3 . R2 . R1 to go from vectors expressed in (x, y, z), to vectors
   expressed in (x', y', z')

R^(-1) = R1^(-1) . R2^(-1) . R3^(-1) to go from vectors expressed 
   in (x', y', z'), to vectors expressed in (x, y, z)
"""
# Rotation Matrices (Complete Rotation)
R = R3.dot(R2).dot(R1)
R_inv = R1_inv.dot(R2_inv).dot(R3_inv)

###############################################################################
# Magnetic field vectors B in each point of the grid in Line of Sight
# [LoS] coordinates (each point representing the center of each voxel)

# 1:
# vectors_LoS_in_B: Vectors of the Line of Sight cube (LoS coordinates),
#   expressed in the magnetic coordinates
vectors_LoS_in_B = []
for vector in vectors:
    vec_LoS_in_B = R_inv.dot(vector)
    vectors_LoS_in_B.append(vec_LoS_in_B)
# pp.pprint(vectors_LoS_in_B)

# 2:
# B = [Bx, By, Bz], in the points given by the grid of the LoS cube
Bs_LoS = []
for vec_LoS_in_B in vectors_LoS_in_B:
    x = vec_LoS_in_B[0]
    y = vec_LoS_in_B[1]
    z = vec_LoS_in_B[2]
    if x == 0 and y == 0 and z == 0:
        continue
    r = np.sqrt(x**2 + y**2 + z**2)
    # Compute Bx, By, Bz in the points of the grid given by the LoS cube,
    #   but expressed in magnetic coordinates x, y, z
    Bx = 3*m * (x*z/r**5)
    By = 3*m * (y*z/r**5)
    Bz = m * (3*z**2/r**5 - 1/r**3)
    # Magnetic Field Vector B in Magnetic Coordinates System (x, y, z)
    B = np.array([Bx, By, Bz])
    # Magnetic Field Vector B in the LoS Coordinates System (x', y', z')
    B_LoS = R.dot(B)
    Bs_LoS.append(B_LoS)

print()
# pp.pprint(Bs_LoS)

###############################################################################
# Plotting
"""
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for vector in vectors_LoS_in_B:
    ax.scatter(vector[0], vector[1], vector[2], color='b')
plt.show()
"""
###############################################################################
# Make unit vectors from magnetic field vectors B in the LoS coordinates system
Bs_LoS_unit = []
for B_LoS in Bs_LoS:
    B_LoS_unit = B_LoS / np.linalg.norm(B_LoS)
    Bs_LoS_unit.append(B_LoS_unit)
print()
pp.pprint(Bs_LoS_unit)


###############################################################################
# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

print(len(vectors_LoS_in_B))
print(len(Bs_LoS_unit))

# Remove point 0,0,0 from mesh (center of the star), to avoid mathematical
# singularities
vectors_LoS_in_B_no_null = []
for vector_LoS_in_B in vectors_LoS_in_B:
    if vector_LoS_in_B.any():
        vectors_LoS_in_B_no_null.append(vector_LoS_in_B)

print(len(vectors_LoS_in_B_no_null))
print(len(Bs_LoS_unit))
for i in range(len(vectors_LoS_in_B_no_null)):
    vector_LoS_in_B = vectors_LoS_in_B_no_null[i]
    B_LoS_unit = Bs_LoS_unit[i]

    # Grid points
    x = vector_LoS_in_B[0]
    y = vector_LoS_in_B[1]
    z = vector_LoS_in_B[2]
    ax.scatter(x, y, z, color='b')

    # B vectors in each grid point
    u = B_LoS_unit[0]
    v = B_LoS_unit[1]
    w = B_LoS_unit[2]
    ax.quiver(x, y, z, 3*u, 3*v, 3*w)

plt.show()

