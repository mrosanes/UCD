"""
radioemission.py

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

The objective(s) of this file is to execute the 3D model, implementing the
developments done in the other Python files of the project. This file
contains the function "main".
"""

import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLineEdit, QDialogButtonBox, QGroupBox, QLabel,
    QSpinBox, QCheckBox, QMainWindow, QFormLayout, QHBoxLayout, QVBoxLayout)

from ucd import UCD


def plot_3D(n=7, beta=0, rotation_angle=0, inclination=90,
            plot3d=True):
    ucd = UCD(
        n=n, beta=beta, rotation_angle=rotation_angle, inclination=inclination,
        plot3d=plot3d)
    # LoS grid points in different systems of coordinates
    points_LoS, points_LoS_in_B = ucd.LoS_cube()
    # Compute and Plot the UCD (or other (sub)stellar object) dipole
    # magnetic vector field and plot it together with the rotation and the
    # magnetic axes, and the different coordinate systems
    ucd.ucd_compute_and_plot(points_LoS_in_B, points_LoS)


def specific_intensities_2D(n=25, beta=0, rotation_angle=0, inclination=90,
                            plot3d=True):
    """
    Notes:
      - Create a UCD with a low grid sampling "n" per edge (eg: <13),
        to be able to plot the vectors without losing a relatively good
        visibility of the vectors
      - Create a UCD with a higher grid sampling "n" per edge (eg: >31 <101)
        in order to have a middle-magnetosphere with a better sampling
        resolution, but without going to too long computation times (for
        101 points per cube edge, ~2min for finding the points of the
        middle-magnetosphere)
    """
    start_time = time.time()
    ucd = UCD(
        n=n, beta=beta, rotation_angle=rotation_angle, inclination=inclination,
        plot3d=plot3d)
    # LoS grid points in different systems of coordinates
    points_LoS, points_LoS_in_B = ucd.LoS_cube()
    # Compute and Plot the UCD (or other (sub)stellar object) dipole
    # magnetic vector field and plot it together with the rotation and the
    # magnetic axes, and the different coordinate systems
    ucd.ucd_compute_and_plot(points_LoS_in_B, points_LoS)
    voxels_inner, voxels_middle = ucd.find_magnetosphere_regions()
    # ucd.plot_middlemag_in_slices(voxels_middle, marker_size=2)
    ucd.LoS_voxel_rays()
    ucd.compute_flux_density_LoS()
    print("Total Flux Density in the plane perpendicular to the LoS:")
    print("{:.4g}".format(ucd.total_flux_density_LoS))
    ucd.plot_2D_specific_intensity_LoS()
    end_time = time.time()
    print("\nUCD computations for %d elements per cube edge, took:\n"
          "%d seconds\n" % (ucd.n, end_time - start_time))


def flux_densities_1D(
        n=15, beta=0, inclination=90,
        plot3d=False):
    """
    Flux densities 1D in function of the rotation phase angles of the UCD (
    or other (sub)stellar object)
    """
    start_time_flux_densities = time.time()
    # Rotation phase angles from 0º to 360º (each 10º)
    rotation_phases = []
    for i in range(36):
        rotation_phases.append(i * 10)

    # Flux densities in function of the rotation phase angles
    flux_densities = []
    for rot_phase in rotation_phases:
        ucd = UCD(n=n,
                  beta=beta, inclination=inclination, rotation_angle=rot_phase,
                  plot3d=plot3d)
        points_LoS, points_LoS_in_B = ucd.LoS_cube()
        ucd.ucd_compute_and_plot(points_LoS_in_B, points_LoS)
        ucd.find_magnetosphere_regions()
        ucd.LoS_voxel_rays()
        ucd.compute_flux_density_LoS()
        # ucd.plot_2D_specific_intensity_LoS()
        flux_densities.append(np.round(ucd.total_flux_density_LoS, 3))

    end_time_flux_densities = time.time()
    print("Time to compute the 1D specific intensities graph along the\n"
          " rotation of the UCD (or other (sub)stellar object), "
          "took:\n%d seconds\n" % (
                  end_time_flux_densities - start_time_flux_densities))

    # 1D Plot of the flux densities in function of the rotation phase angles
    pg.plot(rotation_phases, flux_densities, pen="b", symbol='o')


class InputDialog(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()
        self.setWindowTitle("(Sub)Stellar Object Radio Emission")
        #######################################################################
        form_group_box_angles = QGroupBox()
        layout_angles = QFormLayout()

        # Angle between magnetic and rotation axes [degrees]
        self.beta = QSpinBox()
        self.beta.setMinimum(-360)
        self.beta.setValue(0)
        layout_angles.addRow(QLabel("beta [º]"), self.beta)

        # Rotation angle [degrees]
        self.rotation = QSpinBox()
        self.rotation.setMinimum(-360)
        self.rotation.setValue(0)
        layout_angles.addRow(QLabel("rotation [º]"), self.rotation)

        # Inclination of the rotation axis regarding the LoS [degrees]
        self.inclination = QSpinBox()
        self.inclination.setMinimum(-360)
        self.inclination.setValue(90)
        layout_angles.addRow(QLabel("inclination [º]"), self.inclination)

        form_group_box_angles.setLayout(layout_angles)

        #######################################################################
        form_group_box_middlemag = QGroupBox()
        layout_middlemag = QFormLayout()

        self.frequency = QLineEdit()
        self.frequency.setText("5")
        layout_middlemag.addRow(QLabel("Frequency [GHz]"), self.frequency)

        self.r_alfven = QLineEdit()
        self.r_alfven.setText("16")
        layout_middlemag.addRow(QLabel("R_alfven [R_obj]"), self.r_alfven)

        self.l_middlemag = QLineEdit()
        self.l_middlemag.setText("4")
        layout_middlemag.addRow(QLabel("l_middlemag [R_obj]"),
                                self.l_middlemag)

        self.acc_eff = QLineEdit()
        self.acc_eff.setText("0.002")
        layout_middlemag.addRow(QLabel("r_ne (acceleration efficiency)"),
                                self.acc_eff)

        self.delta = QLineEdit()
        self.delta.setText("2")
        layout_middlemag.addRow(QLabel("δ"), self.delta)

        form_group_box_middlemag.setLayout(layout_middlemag)

        #######################################################################
        form_group_box_innermag = QGroupBox()
        layout_innermag = QFormLayout()

        # Rotation Period of the (sub)stellar object [days]
        self.P_rot = QLineEdit()
        self.P_rot.setText("1")
        layout_innermag.addRow(QLabel("P_rot [days]"), self.P_rot)

        # Density of electrons of the plasma in the inner magnetosphere
        self.n_p = QLineEdit()
        self.n_p.setText("0")
        layout_innermag.addRow(QLabel("np"), self.n_p)

        # Plasma temperature in the inner magnetosphere [K]
        self.T_p = QLineEdit()
        self.T_p.setText("0")
        layout_innermag.addRow(QLabel("Tp [K]"), self.T_p)

        form_group_box_innermag.setLayout(layout_innermag)
        #######################################################################

        # n_3d, n_2d and n_1d: Number of voxels per cube side: use
        # odd numbers for n (it allows having one of the voxels in the middle
        # of the (sub)stellar object)
        v_layout_3d = QFormLayout()
        self.checkbox_3d = QCheckBox("3D Magnetic Field")
        v_layout_3d.addRow(self.checkbox_3d)
        self.n_3d = QSpinBox()
        self.n_3d.setValue(7)
        v_layout_3d.addRow(QLabel("n:"), self.n_3d)
        v_layout_3d.setContentsMargins(0, 0, 20, 0)

        v_layout_2d = QFormLayout()
        self.checkbox_2d = QCheckBox("2D Specific Intensities")
        self.checkbox_2d.setChecked(True)
        v_layout_2d.addRow(self.checkbox_2d)
        self.n_2d = QSpinBox()
        self.n_2d.setValue(25)
        v_layout_2d.addRow(QLabel("n:"), self.n_2d)
        v_layout_2d.setContentsMargins(0, 0, 20, 0)

        v_layout_1d = QFormLayout()
        self.checkbox_1d = QCheckBox("1D Flux Densities")
        v_layout_1d.addRow(self.checkbox_1d)
        self.n_1d = QSpinBox()
        self.n_1d.setValue(7)
        v_layout_1d.addRow(QLabel("n:"), self.n_1d)
        v_layout_1d.setContentsMargins(0, 0, 20, 0)

        h_layout = QHBoxLayout()
        h_layout.addLayout(v_layout_3d)
        h_layout.addLayout(v_layout_2d)
        h_layout.addLayout(v_layout_1d)

        #######################################################################

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addWidget(form_group_box_angles)
        main_layout.addWidget(form_group_box_middlemag)
        main_layout.addWidget(form_group_box_innermag)
        main_layout.addLayout(h_layout)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)

    def accept(self):
        self.setWindowTitle("[PROCESSING...]")
        try:
            frequency = float(self.frequency.text())
            r_alfven = float(self.r_alfven.text())
            l_middlemag = float(self.l_middlemag.text())
            acc_eff = float(self.acc_eff.text())
            delta = float(self.delta.text())
            n_p = float(self.n_p.text())
            T_p = float(self.T_p.text())
        except ValueError:
            raise Exception(
                "Only Float and Int values are accepted as inputs")

        launch_app(
            d3_checkbox=self.checkbox_3d,
            d2_checkbox=self.checkbox_2d,
            d1_checkbox=self.checkbox_1d,
            beta=self.beta.value(),
            rotation=self.rotation.value(),
            inclination=self.inclination.value(),
            n_3d=self.n_3d.value(),
            n_2d=self.n_2d.value(),
            n_1d=self.n_1d.value()
        )

        self.setWindowTitle(
            "(Sub)Stellar Object Radio Emission")

    def reject(self):
        self.close()


def launch_app(d3_checkbox=False, d2_checkbox=False, d1_checkbox=False,
               beta=0, rotation=0, inclination=90, n_3d=7, n_2d=25, n_1d=7):

    if d3_checkbox.isChecked():
        plot_3D(
            n=n_3d, beta=beta, rotation_angle=rotation, inclination=inclination,
            plot3d=True)

    if d2_checkbox.isChecked():
        specific_intensities_2D(
            n=n_2d,
            beta=beta, rotation_angle=rotation, inclination=inclination,
            plot3d=False)

    if d1_checkbox.isChecked():
        flux_densities_1D(
            n=n_1d, beta=beta, inclination=inclination, plot3d=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    input_dialog = InputDialog()
    input_dialog.show()
    sys.exit(app.exec())

