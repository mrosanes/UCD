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
        # Calculation of the column matter optical depth between each voxel
        # (grid element) and the Earth, in the direction of the LoS

        # We reverse the array, to begin adding optical depth, from the voxel
        # that is closest to the Earth (in the LoS direction) to the voxel
        # that is farthest from Earth; in this way we add each time the
        # precedent voxel optical_depth, increasing at each loop turn, a
        # single voxel optical depth, to the total ray optical depth
        self.LoS_voxels_in_ray.reverse()
        ray_position = np.sqrt(self.y ** 2 + self.z ** 2)
        optical_depth = 0
        for i in range(self.n):
            voxel = self.LoS_voxels_in_ray[i]
            if voxel.inside_object:
                # If the voxel is inside object its specific intensity is 0
                continue
            if i != 0:
                previous_voxel_in_ray = self.LoS_voxels_in_ray[i - 1]
                if (previous_voxel_in_ray.optical_depth >= 1000
                        or (ray_position <= 1 and voxel.position_LoS[0] < 0)):
                    # If previous voxel inside object, or the coordinates of
                    # the voxel are behind the object in the LoS coordinates,
                    # voxel is eclipsed and thus, its specific intensity is
                    # not seen from the Earth, so it is set to 0
                    voxel.set_voxel_eclipsed()
                else:
                    # In the contrary, the contribution to the optical depth
                    # is set by adding the contribution of the previous voxel
                    # in the ray (closer to the Earth in the LoS direction)
                    optical_depth += (previous_voxel_in_ray.ab *
                                      previous_voxel_in_ray.voxel_len)
                    voxel.optical_depth = optical_depth

    def compute_specific_intensity_ray(self):
        for voxel in self.LoS_voxels_in_ray:
            self.ray_specific_intensity += (
                    voxel.spec_intensity * np.e**(-voxel.optical_depth))

