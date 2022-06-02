import numpy as np
import matplotlib.pyplot as plt

x_ = np.linspace(-1., 1., 3)
y_ = np.linspace(-1., 1., 3)
z_ = np.linspace(-1., 1., 3)

x, z, y = np.meshgrid(x_, z_, y_)
vectors = []

# Convert meshgrid to array of 3D vectors. Each vector contains the 
#     geometric coordinates of a 3D point in the mesh. The coordinates
#     of each vector represents de center of each of the voxels of the grid.
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            vectors.append([y[i,j,k], x[i,j,k], z[i,j,k]])
            
for vector in vectors:
    print(vector)


