"""
plot_1D_flux_densities.py

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
  - Presenting an example of flux densities 1D plot (a.k.a. Ligth Curve)
"""

import numpy as np
import pyqtgraph as pg

# Rotation phase [0 - 2PI]
rotation_angles = [
 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310,
 320, 330, 340, 350, 360]

# Rotation phase: [0-1]  ([0-1] -> [0º-360º])
rotation_phases = np.array(rotation_angles) / 360.0

# Flux densities as a function of the rotation phase:
flux_densities = [
 0.095, 0.1, 0.111, 0.128, 0.14, 0.153, 0.165, 0.159, 0.159, 0.156, 0.146,
 0.135, 0.116, 0.103, 0.096, 0.095, 0.097, 0.107, 0.127, 0.153, 0.158, 0.17,
 0.175, 0.184, 0.179, 0.173, 0.181, 0.176, 0.174, 0.17, 0.16, 0.157, 0.134,
 0.113, 0.1, 0.095, 0.095]

# Average Flux Densities (in mJy)
FD_avg = np.average(flux_densities)
print("\nAverage Flux Density:")
print(" {:.4g}".format(FD_avg))
print()

app = pg.mkQApp()
window = pg.plot(rotation_phases, flux_densities,
                 pen="b", symbol='o')
window.setWindowTitle("Flux Densities")
app.exec_()

