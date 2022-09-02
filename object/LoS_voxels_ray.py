"""
LoS_voxels_ray.py

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

The objective of this file is:
    - Define the class LoS_Voxels_Ray to group de Voxels in Rays. A single
    Ray (group of Voxels) in each of the points of the Y'Z' plane
    perpendicular to the LoS direction (direction x')
"""

import numpy as np


class LoS_Voxels_Ray(object):
    def __init__(self, y, z):
        """
        :param y: position y' from plane perpendicular to LoS (Y'Z')
        :param z: position z' from plane perpendicular to LoS (Y'Z')
        :param voxels: Array of voxels in the same direction along x' for a
        specific point of the plane Y'Z'
        """

        self.n = 0
        self.y = y
        self.z = z
        self.LoS_voxels_in_ray = []
        self.ray_specific_intensity = 0

    def set_number_voxels_in_ray(self, n):
        self.n = n

    def voxels_optical_depth(self):
        # Calculation of the column matter optical depth between each grid
        # element and the Earth
        for i in range(self.n):
            voxel = self.LoS_voxels_in_ray[i]
            for j in range(i+1, self.n):
                voxel_next_in_ray = self.LoS_voxels_in_ray[j]
                voxel.optical_depth += (
                        voxel_next_in_ray.ab * voxel_next_in_ray.voxel_len)

    def compute_specific_intensity_ray(self):
        for voxel in self.LoS_voxels_in_ray:
            self.ray_specific_intensity += (
                    voxel.spec_intensity * np.e**(-voxel.optical_depth))

