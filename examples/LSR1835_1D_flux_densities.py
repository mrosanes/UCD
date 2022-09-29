"""
LSR1835_1D_flux_densities.py

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

###############################################################################
# Data from a real astronomic observation of LSRJ1835+3259 (RCP):

rotation_phases_0 = [
 0, 0.0937, 0.1933, 0.2929, 0.4686, 0.6444, 0.8201, 0.9959, 1.1716,
 1.3474, 1.5231, 1.6989
]
# Flux densities in mJy:
flux_densities_0 = [
 0.4351, 0.4331, 0.3644, 0.3375, 0.2142, 0.2679, 0.2248, 0.3011, 0.3305,
 0.2917, 0.2315, 0.1886]

###############################################################################
# Data obtained using the software:
# L = 15; Bp = 5000; neA = 5416; Ra = 31

# Rotation phase [0 - 2PI]
rotation_angles = [
 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310,
 320, 330, 340, 350, 360]

# Rotation phase: [0-1]  ([0-1] -> [0º-360º])
rotation_phases = 0.13 + np.array(rotation_angles) / 360.0

# Flux densities as a function of the rotation phase:
flux_densities = 20 * np.array([
 0.016, 0.019, 0.019, 0.019, 0.024, 0.02, 0.019, 0.021, 0.018, 0.021, 0.017,
 0.017, 0.016, 0.015, 0.016, 0.011, 0.011, 0.01, 0.011, 0.01, 0.011, 0.011,
 0.016, 0.015, 0.016, 0.017, 0.017, 0.021, 0.018, 0.021, 0.019, 0.02, 0.024,
 0.019, 0.019, 0.019, 0.016])

###############################################################################

# Average Flux Densities (in mJy)
FD_avg = np.average(flux_densities)
print("\nAverage Flux Density:")
print(" {:.4g}".format(FD_avg))
print()

###############################################################################
# Plotting both light curves (real observation and computed), in
# the same graph:

app = pg.mkQApp()
plt = pg.plot()

orange = (217, 83, 25)
plt.plot(rotation_phases_0, flux_densities_0,
         pen=pg.mkPen(orange, width=2),
         symbol='o', symbolPen='w', symbolBrush=orange)
plt.plot(rotation_phases, flux_densities,
         pen=pg.mkPen('b', width=2),
         symbol='o', symbolPen='w')

plt.setWindowTitle("Flux Densities")
app.exec_()

