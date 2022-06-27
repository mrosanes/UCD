"""
meshgrid_plt.py

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
  - Compute, through rotation matrices, a "magnetic" grid, and plot it,
    in order to apply it later, in other files.
"""

import numpy as np
import matplotlib.pyplot as plt

x_ = np.linspace(-1., 1., 3)
y_ = np.linspace(-1., 1., 3)
z_ = np.linspace(-1., 1., 3)

x, z, y = np.meshgrid(x_, z_, y_)
vectors = []

# Convert meshgrid to array of 3D vectors. Each vector contains the 
# geometric coordinates of a 3D point in the mesh. The coordinates
# of each vector represents the center of each of the voxels of the grid.
for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 3):
            vectors.append([y[i, j, k], x[i, j, k], z[i, j, k]])
            
for vector in vectors:
    print(vector)

"""
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for vector in vectors:
    ax.scatter(vector[0], vector[1], vector[2], color='b')   
plt.show()
"""

###############################################
# ANGLES:

beta = 30
phi = rotation = 0
inc = inclination = 90

b_r = b_rad = np.deg2rad(beta)
p_r = p_rad = np.deg2rad(phi)
i_r = i_rad = np.deg2rad(inc)

# np.round(np.cos(np.deg2rad(60)),4)

# Trigonometric functions of the used angles
sin_b = np.round(np.sin(b_r),4)
cos_b = np.round(np.cos(b_r),4)

sin_p = np.round(np.sin(p_r),4)
cos_p = np.round(np.cos(p_r),4)

sin_i = np.round(np.sin(i_r),4)
cos_i = np.round(np.cos(i_r),4)

# Beta angle from magnetic axis to rotation axis (b = beta)
R1 = np.array([[ cos_b,  0,  sin_b ], 
               [ 0,      1,  0     ], 
               [-sin_b, 0,  cos_b  ]])

# Rotation of the star angle (p = phi = rot)
R2 = np.array([[ cos_p, -sin_p, 0  ], 
               [ sin_p, cos_p,  0  ], 
               [ 0,     0,      1  ]])

# Inclination (i = inc) of the orbit, from line of sight to rotation axis
R3 = np.array([[ sin_i,  0,  cos_i ], 
               [ 0,      1,  0     ], 
               [-cos_i,  0,  sin_i ]])

R1_inv = R1.transpose()
R2_inv = R2.transpose()
R3_inv = R3.transpose()

print(R1_inv)
print(R2_inv)
print(R3_inv)

R = R3.dot(R2).dot(R1)
R_inv = R1_inv.dot(R2_inv).dot(R3_inv)

print()
print(R)
print()
print(R_inv)

vector = [1, 1, 1]
vec_out = R_inv.dot(vector)
print(vec_out)

# vec_LoS_in_B: Vector of the cube from Line of Sight, expressed
# in the magnetic coordinates
vectors_LoS_in_B = []
for vector in vectors:
    vec_LoS_in_B = R_inv.dot(vector)
    vectors_LoS_in_B.append(vec_LoS_in_B)
    
print(vectors_LoS_in_B)    
    
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for vector in vectors_LoS_in_B:
    ax.scatter(vector[0], vector[1], vector[2], color='b')   
plt.show()   

