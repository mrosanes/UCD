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

The objective of this file is:
    - Define the class Voxel which will be later used for each of the voxels
    of the grid with specific parameters each one of them.
"""

import numpy as np


class Voxel(object):
    def __init__(self, B_LoS, voxel_len, position_LoS=[0,0,0],
                 position_in_B=[0,0,0], f=1e9, δ=2):
        """
        :param B_LoS: B field of the voxel
        :param voxel_len: Voxel length
        :param f: Frequency of Radio radiation [Hz]
        :param Ne: Electron Density Number
        :param position_LoS: Position of the center of the voxel in Line of
        Sight (LoS) coordinates.
        :param position_in_B: Position of the center of the voxel in B
        (Dipole Magnetic Field) coordinates.
        """

        self.voxel_len = voxel_len

        # Gyrofrequency of electrons; Frequency of radiation
        self.f = f  # [Hz]

        # Free parameters:  l, Ne , δ, Tp , np
        # Hard energetic population of non-thermal-emitting electrons, and an
        # inner magnetosphere filled by a thermal plasma consistent with a
        # wind-shock model that provides also X-ray emission
        self.δ = δ

        # Boolean indicating if the Voxel belongs or not to the inner,
        # middle or outer magnetosphere.
        self.inside_object = False
        self.eclipsed = False
        self.inner_mag = False
        self.middle_mag = False

        # Module of the Magnetic Field
        self.B = np.linalg.norm(B_LoS)

        # Absorption and emission coefficients initialisation
        self.em = 0
        self.ab = 0
        self.spec_intensity = 0

        #######################################################################
        # In units of sub(stellar) radius
        self.position_LoS = position_LoS
        self.position_in_B = position_in_B
        # Tp: from ~10e5 to ~10e6 caused by the rotating magnetosphere

        # Initialize column matter optical depth between each grid element
        # and the Earth
        self.optical_depth = 0

    def set_inside_object(self):
        """Voxel is located inside the (sub)stellar object. If the voxel is
        inside the object, its specific intensity is 0"""
        self.inside_object = True
        self.spec_intensity = 0
        self.optical_depth = 1000

    def set_eclipsed(self):
        """If the voxel is eclipsed, it is not visible from the Earth. The
        specific intensity is set to 0 to account for the eclipse"""
        self.eclipsed = True
        self.spec_intensity = 0
        self.optical_depth = 1000

    def set_inner_mag(self, n_p, T_p, gudel_threshold,
                      Tp0_higher_than_threshold):
        self.inner_mag = True
        """
        Emission and absorption coefficients in the Inner-Magnetosphere:
        Bremsstrahlung (Free-Free Emission) thermal radiation;
        Gudel, Manuel; 2002 (pag.5 / Formula [6]);
        Annual Review of Astronomy & Astrophysics 40:217-261
        """
        # If the temperature at the surface of the (sub)stellar object is
        # higher than the threshold indicated in Gudel02 of 3.2e5 K, then the
        # temperatures of the inner magnetosphere plasma will always be higher
        # than the threshold, as the temperature increases with the distance
        # to the star:
        if Tp0_higher_than_threshold:
            self.ab = (0.01 * n_p ** 2 * T_p ** (-3 / 2) * self.f ** (-2)
                       * np.log(4.7e10 * T_p / self.f))
        else:
            if T_p >= gudel_threshold:
                self.ab = (0.01 * n_p ** 2 * T_p ** (-3 / 2) * self.f ** (-2)
                           * np.log(4.7e10 * T_p / self.f))
            else:
                self.ab = (0.01 * n_p ** 2 * T_p ** (-3 / 2) * self.f ** (-2)
                           * np.log(5e7 * T_p ** (3 / 2) / self.f))

        # Also from Gudel02 (using units of Gaussian cgs):
        # em = ab * (2 * k * Tp * f**2) / c**2 ->
        self.em = self.ab * 3.1e-37 * T_p * self.f**2

        # Specific intensity of the voxel
        self.spec_intensity = (self.em / self.ab) * (
                1 - np.e ** (-self.ab * self.voxel_len))

    def set_middle_mag(self, Ne):
        self.middle_mag = True
        # Emission and absorption coefficients in the Middle-Magnetosphere:
        # Gyrosynchrotron Emission from a Power-Law Electron Distribution
        # Gudel, Manuel; 2002 (pag.6);
        # Annual Review of Astronomy & Astrophysics 40:217-261
        self.em = 10 ** (-31.32 + 5.24 * self.δ) * Ne * self.B ** (
                -0.22 + 0.9 * self.δ) * self.f ** (1.22 - 0.9 * self.δ)
        self.ab = 10 ** (-0.47 + 6.06 * self.δ) * Ne * self.B ** (
                0.3 + 0.98 * self.δ) * self.f ** (- 1.3 - 0.98 * self.δ)
        # Specific intensity of the voxel
        self.spec_intensity = (self.em / self.ab) * (
                1 - np.e ** (-self.ab * self.voxel_len))

