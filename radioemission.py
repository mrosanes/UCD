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
# Note: in what follows, OBJ and "(sub)stellar object" are used indistinctly

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLineEdit,
    QDialogButtonBox, QGroupBox, QLabel, QSpinBox, QCheckBox, QFormLayout,
    QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt

from alfven_radius.alfven_radius import (
    approximate_alfven_radius, averaged_alfven_radius)
from object.object_main_func import (
    plot_3D, specific_intensities_2D, flux_densities_1D)


# InitialGUI ##################################################################
class InitialGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("(Sub)Stellar Object Radio Emission")
        self.move(400, 250)

        group_box = QGroupBox()
        layout = QVBoxLayout()

        button_compute_Ra = QPushButton("Compute Alfvén Radius (Ra)", self)
        button_compute_Ra.setToolTip("Alfvén Radius is not known:"
                                     " compute Alfvén Radius (Ra)")
        button_compute_Ra.setDefault(True)

        button_specific_intensities = QPushButton(
            "Compute Specific Intensities and Flux Densities", self)
        button_specific_intensities.setToolTip(
            "Compute specific intensities & flux density\n"
            "(Alfvén Radius already known)")
        button_specific_intensities.setDefault(True)

        layout.addWidget(button_compute_Ra)
        layout.addWidget(button_specific_intensities)
        group_box.setLayout(layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box)

        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        button_compute_Ra.clicked.connect(
            self.launch_alfven_radius_gui)
        button_specific_intensities.clicked.connect(
            self.launch_radio_emission_gui)

    def launch_alfven_radius_gui(self):
        alfven_radius_input_dialog = AlfvenRadiusGUI(self)
        alfven_radius_input_dialog.show()

    def launch_radio_emission_gui(self):
        radio_emission_input_dialog = RadioEmissionGUI(self)
        radio_emission_input_dialog.show()


# AlfvenRadiusGUI #############################################################
class AlfvenRadiusGUI(QMainWindow):
    """This GUI is accessed from the InitialGUI by clicking on the
    corresponding PushButton"""
    def __init__(self, parent=InitialGUI):
        super(QWidget, self).__init__(parent)
        self.setWindowTitle("Alfven Radius GUI")
        self.setWindowModality(Qt.WindowModal)

        form_group_box_center = QGroupBox()
        layout_center = QFormLayout()

        # Angle between magnetic and rotation axes [degrees]
        self.beta = QSpinBox()
        self.beta.setMinimum(-180)
        self.beta.setMaximum(180)
        self.beta.setValue(60)
        self.beta.setToolTip("Angle of magnetic axis regarding the"
                             + " rotation axis (Range: [-180º - 180º])")
        layout_center.addRow(QLabel("beta [º]"), self.beta)

        # Angle of magnetic longitude for the computation of Ra [degrees]
        self.zeta = QSpinBox()
        self.zeta.setMinimum(0)
        self.zeta.setMaximum(360)
        self.zeta.setValue(0)
        self.zeta.setToolTip("Magnetic longitude angle"
                             + " (Range: [0º - 360º])")
        layout_center.addRow(QLabel("zeta [º]"), self.zeta)

        # Rotation Period of the (sub)stellar object [days]
        self.Robj2Rsun = QLineEdit()
        self.Robj2Rsun.setValidator(QDoubleValidator())
        self.Robj2Rsun.setText("4")
        self.Robj2Rsun.setToolTip("Robj compared to Rsun radius size factor")
        layout_center.addRow(QLabel("Robj2Rsun [-]"), self.Robj2Rsun)

        # Rotation Period of the (sub)stellar object [days]
        self.P_rot = QLineEdit()
        self.P_rot.setValidator(QDoubleValidator())
        self.P_rot.setText("1")
        self.P_rot.setToolTip("Rotation period of the (sub)stellar object;"
                              + " [days]")
        layout_center.addRow(QLabel("P_rot [days]"), self.P_rot)

        self.Bp = QLineEdit()
        self.Bp.setValidator(QIntValidator())
        self.Bp.setText("1e4")
        self.Bp.setToolTip("Magnetic field strength at the pole of"
                           + " the (sub)stellar object; [Gauss]")
        layout_center.addRow(QLabel("Bp [Gauss]"), self.Bp)

        self.vinf = QLineEdit()
        self.vinf.setValidator(QIntValidator())
        self.vinf.setText("600e3")
        self.vinf.setToolTip("(sub)stellar object wind velocity"
                             + " close to 'infinity'; [m/s]")
        layout_center.addRow(QLabel("vinf [m/s]"), self.vinf)

        self.M_los = QLineEdit()
        self.M_los.setValidator(QDoubleValidator())
        self.M_los.setText("1e-9")
        self.M_los.setToolTip("Mass loss rate from the (sub)stellar object;"
                              + " [Solar Masses / year]")
        layout_center.addRow(QLabel("M_los [Msun / year]"), self.M_los)

        form_group_box_center.setLayout(layout_center)

        # Checkboxes for choosing computation #################################
        # - Approximate Ra: faster computation, but not precise; Alfvén Radius
        #     computation done at the magnetic longitude zeta=0
        # - Averaged Ra + 1D Ra plot: more precise, but slower computation;
        #     Ra averaged over the range of magnetic longitudes [0º-360º],
        #     with one single value of Ra its 5º (computation can take
        #     several hours).
        v_layout = QFormLayout()
        self.checkbox_approximate_Ra = QCheckBox("Approximate Ra")
        self.checkbox_approximate_Ra.setToolTip(
            "Approximate Ra: faster computation, but not precise;\n"
            "Alfvén Radius computation done at the magnetic longitude"
            " zeta=0; NOTE: This computation can take a few minutes (< 5min).")
        self.checkbox_approximate_Ra.setChecked(True)
        self.checkbox_approximate_Ra.toggled.connect(self.unset_averaged_Ra)
        v_layout.addRow(self.checkbox_approximate_Ra)

        self.checkbox_averaged_Ra = QCheckBox("Averaged Ra")
        self.checkbox_averaged_Ra.setToolTip(
            "Computes the averaged Ra, thanks to the 1D Ra plot:\n "
            "more precise, but slower computation;\n"
            "Ra averaged over the range of magnetic longitudes [0º-360º],\n"
            "(one single value of Ra taken each 5º);\n "
            "NOTE: This computation can take several hours.")
        self.checkbox_averaged_Ra.toggled.connect(self.unset_approximate_Ra)
        v_layout.addRow(self.checkbox_averaged_Ra)

        bottom_box = QGroupBox()
        bottom_box.setLayout(v_layout)

        #######################################################################

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.close)

        main_layout = QVBoxLayout()
        main_layout.addWidget(form_group_box_center)
        main_layout.addWidget(bottom_box)
        main_layout.addWidget(button_box)

        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def unset_approximate_Ra(self):
        self.checkbox_approximate_Ra.setChecked(False)

    def unset_averaged_Ra(self):
        self.checkbox_averaged_Ra.setChecked(False)

    def accept(self):
        self.setWindowTitle("[PROCESSING...]")

        # Prepare inputs for functions which computes the Alfvén Radius
        beta = self.beta.value()
        zeta = self.zeta.value()
        Robj2Rsun = float(self.Robj2Rsun.text())
        P_rot = float(self.P_rot.text())
        Bp = float(self.Bp.text())
        vinf = float(self.vinf.text())
        M_los = float(self.M_los.text())

        # Launching application to compute approximate Alfvén Radius
        if self.checkbox_approximate_Ra.isChecked():
            approximate_alfven_radius(
                beta=beta, zeta=zeta, Robj2Rsun=Robj2Rsun, P_rot=P_rot,
                Bp=Bp, vinf=vinf, M_los=M_los)

        # Launching application to compute average Alfvén Radius and 1D plot
        # of the Alfvén Radius values along the magnetic longitude angle
        if self.checkbox_averaged_Ra.isChecked():
            averaged_alfven_radius(
                beta=beta, Robj2Rsun=Robj2Rsun, P_rot=P_rot,
                Bp=Bp, vinf=vinf, M_los=M_los)
        # End launching application ###########################################

        self.setWindowTitle("Alfven Radius Computation")


# RadioEmissionGUI ############################################################
class RadioEmissionGUI(QMainWindow):
    """This GUI is accessed from the InitialGUI by clicking on the
    corresponding PushButton"""
    def __init__(self, parent=InitialGUI):
        super(QWidget, self).__init__(parent)
        self.setWindowTitle("(Sub)Stellar Object Radio Emission")
        self.setWindowModality(Qt.WindowModal)

        #######################################################################
        form_group_box_angles = QGroupBox()
        layout_angles = QFormLayout()

        # Angle between magnetic and rotation axes [degrees]
        self.beta = QSpinBox()
        self.beta.setMinimum(-180)
        self.beta.setMaximum(180)
        self.beta.setValue(0)
        self.beta.setToolTip("Angle of magnetic axis regarding the"
                             + " rotation axis (Range: [-180º - 180º])")
        layout_angles.addRow(QLabel("beta [º]"), self.beta)

        # Rotation angle [degrees]
        self.rotation = QSpinBox()
        self.rotation.setMinimum(0)
        self.rotation.setMaximum(360)
        self.rotation.setValue(0)
        self.rotation.setToolTip("Rotation phase (Range: [0º - 360º])")
        layout_angles.addRow(QLabel("rotation [º]"), self.rotation)

        # Inclination of the rotation axis regarding the LoS [degrees]
        self.inclination = QSpinBox()
        self.inclination.setMinimum(-90)
        self.inclination.setMaximum(90)
        self.inclination.setValue(90)
        self.inclination.setToolTip("Angle of rotation axis regarding the LoS"
                                    " (Range: [-90º - 90º])")
        layout_angles.addRow(QLabel("inclination [º]"), self.inclination)

        form_group_box_angles.setLayout(layout_angles)

        #######################################################################
        form_group_box_center = QGroupBox()
        h_layout = QHBoxLayout()
        layout_center_1 = QFormLayout()
        layout_center_1.setContentsMargins(0, 0, 30, 0)
        layout_center_2 = QFormLayout()

        self.frequency = QLineEdit()
        self.frequency.setValidator(QDoubleValidator())
        self.frequency.setText("5")
        self.frequency.setToolTip("GyroFrequency of electrons")
        layout_center_1.addRow(QLabel("Frequency [GHz]"), self.frequency)

        self.Robj2Rsun = QLineEdit()
        self.Robj2Rsun.setValidator(QDoubleValidator())
        self.Robj2Rsun.setText("4")
        self.Robj2Rsun.setToolTip(
            "Radius of the (sub)stellar object compared to the radius "
            "of the Sun;\n(Dimensionless)")
        layout_center_1.addRow(QLabel("Robj2Rsun"), self.Robj2Rsun)

        self.Bp = QLineEdit()
        self.Bp.setValidator(QIntValidator())
        self.Bp.setText("3000")
        self.Bp.setToolTip("Magnetic field strength at the pole of"
                           + " the (sub)stellar object")
        layout_center_1.addRow(QLabel("Bp [Gauss]"), self.Bp)

        self.r_alfven = QLineEdit()
        self.r_alfven.setValidator(QDoubleValidator())
        self.r_alfven.setText("16")
        self.r_alfven.setToolTip("Averaged Alfvén Radius in R* units")
        layout_center_1.addRow(QLabel("R_alfven [R*]"), self.r_alfven)

        self.L = QLineEdit()
        self.L.setValidator(QIntValidator())
        self.L.setText("30")
        self.L.setToolTip("Length of the cubic grid sides in R* units")
        layout_center_1.addRow(QLabel("L [R*]"), self.L)

        self.l_middlemag = QLineEdit()
        self.l_middlemag.setValidator(QDoubleValidator())
        self.l_middlemag.setText("4")
        self.l_middlemag.setToolTip("Thickness of middle-magnetosphere"
                                    + " in R* units")
        layout_center_1.addRow(QLabel("l_middlemag [R*]"), self.l_middlemag)

        self.acc_eff = QLineEdit()
        self.acc_eff.setValidator(QDoubleValidator())
        self.acc_eff.setText("0.002")
        self.acc_eff.setToolTip("Acceleration efficiency of electrons in the"
                                + " middle-magnetosphere: r_ne = Ne / neA);\n"
                                + "(Dimensionless)")
        layout_center_2.addRow(QLabel("Acceleration Efficiency"), self.acc_eff)

        self.delta = QLineEdit()
        self.delta.setValidator(QDoubleValidator())
        self.delta.setText("2")
        self.delta.setToolTip("Spectral index of non-thermal electron"
                              + " energy distribution")
        layout_center_2.addRow(QLabel("δ"), self.delta)

        # Distance to the (sub)stellar object [cm]
        self.D = QLineEdit()
        self.D.setValidator(QDoubleValidator())
        self.D.setText("352")
        self.D.setToolTip("Distance to the (sub)stellar object [Pc]")
        layout_center_2.addRow(QLabel("Distance [Pc]"), self.D)

        # Rotation Period of the (sub)stellar object [days]
        self.P_rot = QLineEdit()
        self.P_rot.setValidator(QDoubleValidator())
        self.P_rot.setText("1")
        self.P_rot.setToolTip("Rotation period of the (sub)stellar object")
        layout_center_2.addRow(QLabel("P_rot [days]"), self.P_rot)

        # Density of electrons of the plasma in the inner magnetosphere
        self.n_p0 = QLineEdit()
        self.n_p0.setValidator(QDoubleValidator())
        self.n_p0.setText("0")
        self.n_p0.setToolTip("Plasma electron density, in inner-magnetosphere,"
                             + " at the stellar surface")
        layout_center_2.addRow(QLabel("np [cm^(−3)]"), self.n_p0)

        # Plasma temperature in the inner magnetosphere [K]
        self.T_p0 = QLineEdit()
        self.T_p0.setValidator(QIntValidator())
        self.T_p0.setText("0")
        self.T_p0.setToolTip("Plasma temperature in inner-magnetosphere,"
                             + " at the stellar surface")
        layout_center_2.addRow(QLabel("Tp [K]"), self.T_p0)

        h_layout.addLayout(layout_center_1)
        h_layout.addLayout(layout_center_2)
        form_group_box_center.setLayout(h_layout)

        # Points per cube side ################################################

        # n_3d, n_2d and n_1d: Number of voxels per cube side: use
        # odd numbers for n (it allows having one of the voxels in the middle
        # of the (sub)stellar object)
        v_layout_3d = QFormLayout()
        self.checkbox_3d = QCheckBox("3D Magnetic Field")
        self.checkbox_3d.setToolTip("3D plot of the magnetic vectorial field")
        v_layout_3d.addRow(self.checkbox_3d)
        self.n_3d = QSpinBox()
        self.n_3d.setValue(7)
        self.n_3d.setToolTip("number of points per cube side"
                             + " (3D magnetic field)")
        v_layout_3d.addRow(QLabel("n:"), self.n_3d)
        v_layout_3d.setContentsMargins(0, 0, 20, 0)

        v_layout_2d = QFormLayout()
        self.checkbox_2d = QCheckBox("2D Specific Intensities")
        self.checkbox_2d.setChecked(True)
        self.checkbox_2d.setToolTip("2D image of the specific intensities in"
                                    + " the plane perpendicular to the LoS\n"
                                    + " at the specific rotation phase"
                                    + " indicated by the user")
        v_layout_2d.addRow(self.checkbox_2d)
        self.n_2d = QSpinBox()
        self.n_2d.setValue(13)
        self.n_2d.setToolTip("Number of points per cube side"
                             + " (2D specific intensities)")
        v_layout_2d.addRow(QLabel("n:"), self.n_2d)
        v_layout_2d.setContentsMargins(0, 0, 20, 0)

        v_layout_1d = QFormLayout()
        self.checkbox_1d = QCheckBox("1D Flux Densities")
        self.checkbox_1d.setToolTip("1D Flux Densities (Light Curve) along a"
                                    + " complete rotation of 360º")
        v_layout_1d.addRow(self.checkbox_1d)
        self.n_1d = QSpinBox()
        self.n_1d.setValue(7)
        self.n_1d.setToolTip("Number of points per cube side"
                             + " (1D flux densities)")
        v_layout_1d.addRow(QLabel("n:"), self.n_1d)
        v_layout_1d.setContentsMargins(0, 0, 0, 0)

        bottom_box = QGroupBox()
        h_layout = QHBoxLayout()
        h_layout.addLayout(v_layout_3d)
        h_layout.addLayout(v_layout_2d)
        h_layout.addLayout(v_layout_1d)
        bottom_box.setLayout(h_layout)

        #######################################################################

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.close)

        main_layout = QVBoxLayout()
        main_layout.addWidget(form_group_box_angles)
        main_layout.addWidget(form_group_box_center)
        main_layout.addWidget(bottom_box)
        main_layout.addWidget(button_box)

        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def accept(self):
        self.setWindowTitle("[PROCESSING...]")

        frequency = float(self.frequency.text()) * 1e9  # [Hz]
        Robj2Rsun = float(self.Robj2Rsun.text())
        r_alfven = float(self.r_alfven.text())
        L = int(self.L.text())
        Bp = int(self.Bp.text())
        l_middlemag = float(self.l_middlemag.text())
        acc_eff = float(self.acc_eff.text())
        delta = float(self.delta.text())
        D = float(self.D.text())
        P_rot = float(self.P_rot.text())
        # n_p0 = float(self.n_p0.text())
        # T_p0 = int(self.T_p0.text())

        # Launching application with inputs entered by the user ###############
        if self.checkbox_3d.isChecked():
            plot_3D(
                L=L, n=self.n_3d.value(), beta=self.beta.value(),
                rotation_angle=self.rotation.value(),
                inclination=self.inclination.value(),
                Robj_Rsun_scale=Robj2Rsun, Bp=Bp, Pr=P_rot, D_pc=D,
                f=frequency, Ra=r_alfven, l_middlemag=l_middlemag, δ=delta,
                plot3d=True)

        if self.checkbox_2d.isChecked():
            specific_intensities_2D(
                L=L, n=self.n_2d.value(), beta=self.beta.value(),
                rotation_angle=self.rotation.value(),
                inclination=self.inclination.value(),
                Robj_Rsun_scale=Robj2Rsun, Bp=Bp, Pr=P_rot, D_pc=D,
                f=frequency, Ra=r_alfven, l_middlemag=l_middlemag, δ=delta,
                plot3d=False)

        if self.checkbox_1d.isChecked():
            flux_densities_1D(
                L=L, n=self.n_1d.value(), beta=self.beta.value(),
                inclination=self.inclination.value(),
                Robj_Rsun_scale=Robj2Rsun, Bp=Bp, Pr=P_rot, D_pc=D,
                f=frequency, Ra=r_alfven, l_middlemag=l_middlemag, δ=delta,
                plot3d=False)
        # End launching application ###########################################

        self.setWindowTitle(
            "(Sub)Stellar Object Radio Emission")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    initial_gui = InitialGUI()
    initial_gui.show()
    sys.exit(app.exec())

