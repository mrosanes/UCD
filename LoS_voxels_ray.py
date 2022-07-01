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

The objectives of this file is:
    - Define the class LoS_Voxels to group de Voxels. A single group of Voxels
    in each of the points of the plane Y'Z' perpendicular to the LoS
    direction (x')
"""


class LoS_Voxels_Ray(object):
    def __init__(self, y, z, LoS_voxels_in_ray):
        """

        :param y: position y' from plane perpendicular to LoS (Y'Z')
        :param z: position z' from plane perpendicular to LoS (Y'Z')
        :param voxels: Array of voxels in the same direction along x' for a
        specific point of the plane Y'Z'
        """

        self.y = y
        self.z = z
        self.LoS_voxels_in_ray = LoS_voxels_in_ray

