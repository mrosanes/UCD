"""
plot_alfven_results.py

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
  - Presenting an example with the results of the Alfvén Radius of a
  (sub)stellar object, as a function of the Magnetic Longitude, for an object
  of radius: R_obj = 4*Rsun
"""

import numpy as np
import pyqtgraph as pg

"""
Units used: cgs
---------------

Parameters used for the results presented in this example:
----------------------------------------------------------
- Radius of the (sub)stellar object (R_obj = R*):
    R_obj = 4 * Rsun
- beta = 60 
- Robj2Rsun = 4
- P_rot = 1 
- Bp = 1e4                   
- v_inf=600e3
- M_los = 1e-9
"""

# Magnetic longitude angles in radians [0 - 2PI]
magnetic_longitude_angles = [
 0.0, 0.08726646, 0.17453293, 0.26179939, 0.34906585,
 0.43633231, 0.52359878, 0.61086524, 0.6981317, 0.78539816,
 0.87266463, 0.95993109, 1.04719755, 1.13446401, 1.22173048,
 1.30899694, 1.3962634, 1.48352986, 1.57079633, 1.65806279,
 1.74532925, 1.83259571, 1.91986218, 2.00712864, 2.0943951,
 2.18166156, 2.26892803, 2.35619449, 2.44346095, 2.53072742,
 2.61799388, 2.70526034, 2.7925268, 2.87979327, 2.96705973,
 3.05432619, 3.14159265, 3.22885912, 3.31612558, 3.40339204,
 3.4906585, 3.57792497, 3.66519143, 3.75245789, 3.83972435,
 3.92699082, 4.01425728, 4.10152374, 4.1887902, 4.27605667,
 4.36332313, 4.45058959, 4.53785606, 4.62512252, 4.71238898,
 4.79965544, 4.88692191, 4.97418837, 5.06145483, 5.14872129,
 5.23598776, 5.32325422, 5.41052068, 5.49778714, 5.58505361,
 5.67232007, 5.75958653, 5.84685299, 5.93411946, 6.02138592,
 6.10865238, 6.19591884, 6.28318531]

# Rotation phase: [0-1]  ([0-1] -> [0º-360º])
magnetic_longitude_phase = np.array(magnetic_longitude_angles) / (2*np.pi)

# Rotation angle in degrees:  [0º-360º])
magnetic_longitude_degrees = (180/np.pi) * np.array(magnetic_longitude_angles)

# Alfvén Radius, in R_obj (R*) units, as a function of the magnetic longitude:
alfven_radius_array = [
 15.699, 15.644, 15.489, 15.257, 14.977, 14.676, 14.375, 14.086, 13.817,
 13.574, 13.358, 13.17,  13.008, 12.874, 12.765, 12.681, 12.622, 12.586,
 12.575, 12.586, 12.622, 12.681, 12.765, 12.874, 13.008, 13.17,  13.358,
 13.574, 13.817, 14.086, 14.375, 14.676, 14.977, 15.257, 15.489, 15.644,
 15.699, 15.644, 15.489, 15.257, 14.977, 14.676, 14.375, 14.086, 13.817,
 13.574, 13.358, 13.17,  13.008, 12.874, 12.765, 12.681, 12.622, 12.586,
 12.575, 12.586, 12.622, 12.681, 12.765, 12.874, 13.008, 13.17,  13.358,
 13.574, 13.817, 14.086, 14.375, 14.676, 14.977, 15.257, 15.489, 15.644,
 15.699]

# Average Alfvén Radius in R_obj (R*) units
Ra = np.average(alfven_radius_array)
print()
print("Average Alfvén Radius:")
print(" {:.4g}".format(Ra))
print()

app = pg.mkQApp()
# window = pg.plot(magnetic_longitude_angles, alfven_radius_array,
#                  pen="b", symbol='o')
# window.setWindowTitle("Alfvén = f(magnetic_longitude)")

window2 = pg.plot(magnetic_longitude_degrees, alfven_radius_array,
                  pen="b", symbol='o')
window2.setWindowTitle("Alfvén = f(magnetic_longitude_degrees)")
app.exec_()

