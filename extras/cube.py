import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

cube = voxelarray = np.ones((3, 3, 3), dtype='bool')
# print(cube[:,:,0])
# print(np.shape(cube[:,:,0]))

ax = plt.figure().add_subplot(projection='3d')

alpha = 0.4
ax.voxels(cube, edgecolors='k', facecolors=(0, 0, 1, alpha))
# for angle in (15, 30, 45):
#    ax.view_init(30, angle)
#    plt.pause(2)
plt.show()




