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

The objective(s) of this file are:
  - Present the results of the Alfvén Radius of a (sub)stellar object, as
    a function of the Magnetic Longitude
"""

import numpy as np
import pyqtgraph as pg

# Note: B units [Gauss]

Rsun = 6.96e8  # [m]

# Set radius of the (sub)stellar object:
R_obj = 4*Rsun

# rotation in radians
rotation_phase = [
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

# Rotation phase: [0-1] ([0-1] -> [0º-360º])
magnetic_longitude_angles = np.array(rotation_phase) / (2*np.pi)

# Alfvén Radius as a function of the magnetic longitude:
alfven_radius_array = [
 64979666585.0315, 64743646556.1279, 64076451850.8515, 63082171929.7342,
 61887738216.4647, 60607930655.9848, 59329021933.3496, 58107801477.1043,
 56977667599.4493, 55955911725.9531, 55049717632.9937, 54260396606.3796,
 53586142112.7451, 53023735257.0119, 52569565718.0500, 52220226645.6084,
 51972852383.2495, 51825304428.6803, 51776269227.4744, 51825304428.6803,
 51972852383.2495, 52220226645.6084, 52569565718.0500, 53023735257.0119,
 53586142112.7451, 54260396606.3796, 55049717632.9937, 55955911725.9531,
 56977667599.4493, 58107801477.1043, 59329021933.3496, 60607930655.9848,
 61887738216.4647, 63082171929.7342, 64076451850.8515, 64743646556.1279,
 64979666585.0315, 64743646556.1279, 64076451850.8515, 63082171929.7342,
 61887738216.4647, 60607930655.9848, 59329021933.3496, 58107801477.1043,
 56977667599.4493, 55955911725.9531, 55049717632.9937, 54260396606.3796,
 53586142112.7451, 53023735257.0119, 52569565718.0500, 52220226645.6084,
 51972852383.2495, 51825304428.6803, 51776269227.4744, 51825304428.6803,
 51972852383.2495, 52220226645.6084, 52569565718.0500, 53023735257.0119,
 53586142112.7451, 54260396606.3796, 55049717632.9937, 55955911725.9531,
 56977667599.4493, 58107801477.1043, 59329021933.3496, 60607930655.9848,
 61887738216.4647, 63082171929.7342, 64076451850.8515, 64743646556.1279,
 64979666585.0315]

alfven_radius_array = np.array(alfven_radius_array)

# Normalized Alfvén Radius, in R* units
alfven_radius_array_norm = np.round(alfven_radius_array / R_obj, 3)
Ra = np.average(alfven_radius_array_norm)

print(magnetic_longitude_angles)
print()
# Alfvén Radius in R_obj (R*) units
print(alfven_radius_array_norm)
print()
print("Average Alfvén Radius:")
print(Ra)
print()


app = pg.mkQApp()
pg.plot(magnetic_longitude_angles, alfven_radius_array_norm,
        pen="b", symbol='o')
app.exec_()

