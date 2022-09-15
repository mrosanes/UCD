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
        optical_depth = 0
        for i in range(self.n):
            voxel = self.LoS_voxels_in_ray[i]

            if not voxel.inside_object and not voxel.eclipsed and i != 0:
                """The optical depth shall be computed for all voxels not being
                the closest to Earth in the LoS direction (i!=0); and only
                if the voxels are not inside the object nor eclipsed by it"""
                previous_voxel_in_ray = self.LoS_voxels_in_ray[i - 1]
                # The contribution to the optical depth is set by adding, at
                # each loop turn, the contribution of the previous voxel in
                # the ray (the one closer to the Earth in the LoS direction)
                optical_depth += (previous_voxel_in_ray.ab *
                                  previous_voxel_in_ray.voxel_len)
                voxel.optical_depth = optical_depth

    def compute_specific_intensity_ray(self):
        for voxel in self.LoS_voxels_in_ray:
            if not voxel.eclipsed:
                self.ray_specific_intensity += (
                        voxel.spec_intensity * np.e**(-voxel.optical_depth))

