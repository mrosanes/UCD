"""
voxel.py

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
    - Define the class Voxel which will be later used for each of the voxels
    of the grid with specific parameters each one of them.
"""

import numpy as np


class Voxel(object):
    def __init__(self, B_LoS, voxel_len,
                 position_LoS=np.array([0,0,0]),
                 position_in_B=np.array([0,0,0]),
                 inner_mag=False, middle_mag=False, outer_mag=False,
                 δ=1.2, Ne=0):
        """
        :param B_LoS: B field of the voxel
        :param voxel_len: Voxel length
        :param f: Frequency of Radio radiation
        :param Ne: Electron Density Number
        :param position_LoS: Position of the center of the voxel in Line of
        Sight (LoS) coordinates.
        :param position_in_B: Position of the center of the voxel in B
        (Dipole Magnetic Field) coordinates.
        """

        # Boolean indicating if the Voxel belongs or not to the
        # middle-magnetosphere.
        self.inner_mag = inner_mag
        self.middle_mag = middle_mag
        self.outer_mag = outer_mag

        # Gyrofrequency of electrons; Frequency of radiation
        v = 5e9  # 5 GHz

        # Free parameters:  l, Ne , δ, Tp , np
        self.Ne = Ne

        # For testing purposes: Uncomment to use an imposed (non-computed) Ne
        # self.Ne = Ne = 1e50

        # Module of the Magnetic Field
        self.B = B = np.linalg.norm(B_LoS)

        # Emission and absorption coefficients in the middle-magnetosphere:
        # Gyrosynchrotron Emission from a Power-Law Electron Distribution
        # Gudel, Manuel; 2002 (pag.6);
        # Annual Review of Astronomy & Astrophysics 40:217-261
        self.em = 10**(-31.32 + 5.24 * δ) * Ne * B**(
                -0.22 + 0.9 * δ) * v**(1.22 - 0.9 * δ)
        self.ab = 10**(-0.47 + 6.06 * δ) * Ne * B**(
                0.3 + 0.98 * δ) * v**(- 1.3 - 0.98 * δ)

        #######################################################################
        # In units of sub(stellar) radius
        self.position_LoS = position_LoS
        self.position_in_B = position_in_B
        # Tp: from ~10e5 to ~10e6 caused by the rotating magnetosphere

        """
        As frequencies for the radio emission we can use, for instance, 
        frequencies like: 5, 8.4 and 15 GHz. 
        The model is not computed at 1.4 and 22 GHz because:
        1) at low frequency the emitting region extends far from the
        star and close to the Alfvén surface, where the geometry of
        the magnetosphere is not yet well known
        2) at high frequency the magnetosphere must be so closely
        sampled that computational times are prohibitive
        """

        # Hard energetic population of non-thermal-emitting electrons, and an
        # inner magnetosphere filled by a thermal plasma consistent with a
        # wind-shock model that provides also X-ray emission

        # Specific intensity inside the voxel object
        self.voxel_len = voxel_len
        self.spec_intensity = (self.em / self.ab) * (
                1 - np.e**(-self.ab * self.voxel_len))

        # Column matter optical depth between each grid element and the Earth
        self.optical_depth = 0

    def set_inner_mag(self, bool_inner_mag):
        self.inner_mag = bool_inner_mag

    def set_middle_mag(self, bool_middle_mag):
        self.middle_mag = bool_middle_mag

    def set_outer_mag(self, bool_outer_mag):
        self.outer_mag = bool_outer_mag

    def set_Ne(self, Ne):
        self.Ne = Ne

