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

import numpy as np
import pyqtgraph as pg

"""Plot 1D Alfvén radius along magnetic equator (magnetic longitude)"""

Rsun = 6.96e8
R_obj = 4 * Rsun
alfven_radius_array = [64979666585.0315, 51855086500.8173, 63577100496.4035,
                       52492395361.1043, 60517419853.7907]

alfven_radius_array_norm = []
alfven_radius_array = np.array(alfven_radius_array)
alfven_radius_array_norm = np.round(alfven_radius_array / R_obj, 3)

rotation_phase = zet = [0, 80, 160, 240, 320]

print(alfven_radius_array_norm)

app = pg.mkQApp()
pg.plot(rotation_phase, alfven_radius_array_norm, pen="b", symbol='o')
app.exec_()


"""
Other formulas and constants
kB = 1.380649e-23  # Boltzmann constant [J/K]
n_p = 1e7  # [cm^(-3)]
T_p = 1e6  # [K]

L = (Bp**2 / (16 * np.pi * n_p * kB * Tp)) ** (1/6)
"""

"""
Inner magnetosphere:
r < Ralfvén

Middle magnetosphere:
r > Ralfvén
r < Ralfvén + l

Outter magnetosphere
r > Ralfvén + l

with the equation of a field line:
r = L [cos(lambda)]^2
"""

