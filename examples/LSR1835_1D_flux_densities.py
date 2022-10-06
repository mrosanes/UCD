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
rotation_phases_observed = [
    0.0469, 0.1435, 0.2431, 0.3808, 0.5565, 0.7323, 0.9080, 1.0838, 1.2595,
    1.4353, 1.6110, 1.8571]

# Flux densities in mJy:
flux_densities_observed = [
    0.4351, 0.4331, 0.3644, 0.3375, 0.2142, 0.2679, 0.2248, 0.3011, 0.3305,
    0.2917, 0.2315, 0.1886]

###############################################################################
# Data obtained using the software:
# L = 15; Bp = 5000; neA = 5416; Ra = 31

rotation_angles = []
for angle in range(-50, 610, 10):
    rotation_angles.append(angle)
# Rotation phase: [0-1]  ([0-1] -> [0º-360º])
rotation_phases = 0.23 + np.array(rotation_angles) / 360.0

# Flux densities as a function of the rotation phase:
flux_densities_initial_i140_beta50 = 20 * np.array([
    0.024, 0.019, 0.019, 0.019, 0.016,
    0.016, 0.019, 0.019, 0.019, 0.024, 0.02, 0.019, 0.021, 0.018, 0.021, 0.017,
    0.017, 0.016, 0.015, 0.016, 0.011, 0.011, 0.01, 0.011, 0.01, 0.011, 0.011,
    0.016, 0.015, 0.016, 0.017, 0.017, 0.021, 0.018, 0.021, 0.019, 0.02, 0.024,
    0.019, 0.019, 0.019, 0.016,
    0.016, 0.019, 0.019, 0.019, 0.024, 0.02, 0.019, 0.021, 0.018, 0.021,
    0.017, 0.017, 0.016, 0.015, 0.016, 0.011, 0.011, 0.01, 0.011, 0.01,
    0.011, 0.011, 0.016, 0.015])

flux_densities_fitted_i150_beta30 = 20 * np.array(
    [0.021, 0.019, 0.019, 0.021, 0.021,
     0.021, 0.021, 0.019, 0.019, 0.021, 0.018, 0.019, 0.018, 0.019, 0.017,
     0.016, 0.015, 0.017, 0.011, 0.013, 0.012, 0.011, 0.012, 0.012, 0.012,
     0.011, 0.012, 0.013, 0.011, 0.017, 0.015, 0.016, 0.017, 0.019, 0.018,
     0.019, 0.018, 0.021, 0.019, 0.019, 0.021, 0.021,
     0.021, 0.021, 0.019, 0.019, 0.021, 0.018, 0.019, 0.018, 0.019, 0.017,
     0.016, 0.015, 0.017, 0.011, 0.013, 0.012, 0.011, 0.012, 0.012, 0.012,
     0.011, 0.012, 0.013, 0.011])

###############################################################################

# Average Flux Densities (in mJy)
FD_avg_initial = np.average(flux_densities_initial_i140_beta50)
print("\nAverage Flux Density Initial (inclination=140º; beta=50º):")
print(" {:.4g}".format(FD_avg_initial))
print()

# Average Flux Densities Fitted (in mJy)
FD_avg_fitted = np.average(flux_densities_fitted_i150_beta30)
print("\nAverage Flux Density Fitted (inclination=150º; beta=30º):")
print(" {:.4g}".format(FD_avg_fitted))
print()
###############################################################################
# Plotting both light curves (real observation and computed), in
# the same graph:

app = pg.mkQApp()
plt = pg.plot()

orange = (217, 83, 25)
plt.plot(rotation_phases_observed, flux_densities_observed,
         pen=pg.mkPen(orange, width=2),
         symbol='o', symbolPen='w', symbolBrush=orange)
plt.plot(rotation_phases, flux_densities_initial_i140_beta50,
         pen=pg.mkPen('b', width=2),
         symbol='o')
plt.setWindowTitle("Flux Densities")

plt2 = pg.plot()
plt2.plot(rotation_phases_observed, flux_densities_observed,
          pen=pg.mkPen(orange, width=2),
          symbol='o', symbolPen='w', symbolBrush=orange)
plt2.plot(rotation_phases, flux_densities_fitted_i150_beta30,
          pen=pg.mkPen('b', width=2),
          symbol='o')
plt2.setWindowTitle("Flux Densities Fitted")

app.exec_()

