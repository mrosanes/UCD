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
    def __init__(self, B_LoS_x, voxel_len, f=5e9, Ne=1,
                 position_LoS_plot=np.array([0,0,0]),
                 position_in_B=np.array([0,0,0]),
                 inner_mag=False, middle_mag=False, outer_mag=False,
                 coef_emission=1, coef_absorption=1):
        """
        :param B_LoS_x: B field of the voxel in the LoS (x') direction
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

        # Emission and absorption coefficients
        self.em = coef_emission
        self.ab = coef_absorption

        # Free parameters:  l, Ne , δ, Tp , np

        """
        NOTE: What interests us is the Longitudinal Magnetic Field (along 
        the Line of Sight (LoS), which 
        """
        # Longitudinal Magnetic Field
        self.B_LoS_x = B_LoS_x

        # Gyrofrequency of electrons
        self.f = f

        # Lorentz factor
        γ = 1.2

        # δ~2 in some MCP stars according C.Trigilio el al. (ESO 2004))
        self.δ = 2

        #######################################################################
        # Attributes of Voxel objects
        self.Ne = Ne
        # In units of sub(stellar) radius
        self.position_LoS_plot = position_LoS_plot
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

        self.f = 5e9  # Frequency of radiation 5 GHz

        # Hard energetic population of non-thermal-emitting electrons, and an
        # inner magnetosphere filled by a thermal plasma consistent with a
        # wind-shock model that provides also X-ray emission

        # Specific intensity inside the voxel object
        # TODO: use correct formula; Ne and B shall influence the emission and
        #   absorption coefficients: they shall be embedded in them.
        self.voxel_len = voxel_len
        self.spec_intensity = self.Ne * np.abs(B_LoS_x) * self.em / self.ab * (
                1 - np.e**(-self.ab * self.voxel_len))

        # Column matter optical depth between each grid element and the Earth
        self.optical_depth = 0

    def set_inner_mag(self, bool_inner_mag):
        self.inner_mag = bool_inner_mag

    def set_middle_mag(self, bool_middle_mag):
        self.middle_mag = bool_middle_mag

    def set_outer_mag(self, bool_outer_mag):
        self.outer_mag = bool_outer_mag

