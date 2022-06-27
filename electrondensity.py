"""
electrondensity.py

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
"""

"""
Knowing "Ra" and defining "l", look for the points of the grid that are
inside the middle magnetosphere, that are the ones which have a contribution
in the radio (non-thermal) emission. If possible, paint these points in
another color, in order to know that the computations are correct
"""


class UCD:
    """UCD object"""

    def compute_electron_density():
        print("Electron density")
        return "hi"


ucd = UCD()
ucd_hiho = ucd.compute_electron_density()
print(ucd_hiho)
