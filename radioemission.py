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
    alfven_radius_at_given_zeta, averaged_alfven_radius)
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
        info = ("Angle of magnetic axis regarding the"
                + " rotation axis (Range: [-180º - 180º])")
        self.beta.setToolTip(info)
        beta_label = QLabel("β [º]")
        beta_label.setToolTip(info)
        layout_center.addRow(beta_label, self.beta)

        # Angle of magnetic longitude for the computation of Ra [degrees]
        self.zeta = QSpinBox()
        self.zeta.setMinimum(0)
        self.zeta.setMaximum(360)
        self.zeta.setValue(0)
        info = "Magnetic longitude angle (Range: [0º - 360º])"
        self.zeta.setToolTip(info)
        zeta_label = QLabel("ζ [º]")
        zeta_label.setToolTip(info)
        layout_center.addRow(zeta_label, self.zeta)

        # Rotation Period of the (sub)stellar object [days]
        self.Robj2Rsun = QLineEdit()
        self.Robj2Rsun.setValidator(QDoubleValidator())
        self.Robj2Rsun.setText("4")
        info = "Robj compared to Rsun radius size factor (dimensionless)"
        self.Robj2Rsun.setToolTip(info)
        Robj2Rsun_label = QLabel("Robj2Rsun [-]")
        Robj2Rsun_label.setToolTip(info)
        layout_center.addRow(Robj2Rsun_label, self.Robj2Rsun)

        # Rotation Period of the (sub)stellar object [days]
        self.P_rot = QLineEdit()
        self.P_rot.setValidator(QDoubleValidator())
        self.P_rot.setText("1")
        info = "Rotation period of the (sub)stellar object [days]"
        self.P_rot.setToolTip(info)
        P_rot_label = QLabel("P_rot [days]")
        P_rot_label.setToolTip(info)
        layout_center.addRow(P_rot_label, self.P_rot)

        self.Bp = QLineEdit()
        self.Bp.setValidator(QIntValidator())
        self.Bp.setText("1e4")
        info = ("Magnetic field strength at the pole of"
                + " the (sub)stellar object [Gauss]")
        self.Bp.setToolTip(info)
        Bp_label = QLabel("Bp [Gauss]")
        Bp_label.setToolTip(info)
        layout_center.addRow(Bp_label, self.Bp)

        self.v_inf = QLineEdit()
        self.v_inf.setValidator(QIntValidator())
        self.v_inf.setText("600")
        info = "(sub)stellar object wind velocity close to 'infinity' [km/s]"
        self.v_inf.setToolTip(info)
        v_inf_label = QLabel("v_inf [km/s]")
        v_inf_label.setToolTip(info)
        layout_center.addRow(v_inf_label, self.v_inf)

        self.M_los = QLineEdit()
        self.M_los.setValidator(QDoubleValidator())
        self.M_los.setText("1e-9")
        info = ("Mass loss rate from the (sub)stellar object;"
                + " [Solar Masses / year]")
        self.M_los.setToolTip(info)
        M_los_label = QLabel("M_los [Msun / year]")
        M_los_label.setToolTip(info)
        layout_center.addRow(M_los_label, self.M_los)

        form_group_box_center.setLayout(layout_center)

        # Checkboxes for choosing computation #################################
        v_layout = QFormLayout()
        self.checkbox_Ra_at_zeta = QCheckBox("Ra at a given ζ")
        self.checkbox_Ra_at_zeta.setToolTip(
            "Ra at a given ζ (zeta);\n"
            "Alfvén Radius computation done at the specific magnetic longitude"
            " (ζ);\nNOTE: This computation can take a few minutes (< 5min).")
        self.checkbox_Ra_at_zeta.setChecked(True)
        self.checkbox_Ra_at_zeta.toggled.connect(self.unset_averaged_Ra)
        v_layout.addRow(self.checkbox_Ra_at_zeta)

        self.checkbox_averaged_Ra = QCheckBox("Averaged Ra")
        self.checkbox_averaged_Ra.setToolTip(
            "Computes the averaged Ra, thanks to a 1D Ra plot:\n"
            "more precise, but slower computation;\n"
            "Ra averaged over the range of magnetic longitudes [0º-360º],\n"
            "(one single value of Ra taken each 10º);\n "
            "NOTE: This computation can take from 15min to some hours.")
        self.checkbox_averaged_Ra.toggled.connect(self.unset_Ra_at_zeta)
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

    def unset_Ra_at_zeta(self):
        self.checkbox_Ra_at_zeta.setChecked(False)

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
        v_inf = float(self.v_inf.text())
        M_los = float(self.M_los.text())

        # Launching application to compute Alfvén Radius at a given magnetic
        # longitude zeta (ζ)
        if self.checkbox_Ra_at_zeta.isChecked():
            alfven_radius_at_given_zeta(
                beta=beta, zeta=zeta, Robj2Rsun=Robj2Rsun, P_rot=P_rot,
                Bp=Bp, v_inf=v_inf, M_los=M_los)

        # Launching application to compute average Alfvén Radius and 1D plot
        # of the Alfvén Radius values along the magnetic longitude angle
        if self.checkbox_averaged_Ra.isChecked():
            averaged_alfven_radius(
                beta=beta, Robj2Rsun=Robj2Rsun, P_rot=P_rot,
                Bp=Bp, v_inf=v_inf, M_los=M_los)
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
        info = ("Angle of magnetic axis regarding the"
                + " rotation axis (Range: [-180º - 180º])")
        self.beta.setToolTip(info)
        beta_label = QLabel("β [º]")
        beta_label.setToolTip(info)
        layout_angles.addRow(beta_label, self.beta)

        # Rotation angle [degrees]
        self.rotation = QSpinBox()
        self.rotation.setMinimum(0)
        self.rotation.setMaximum(360)
        self.rotation.setValue(0)
        info = "Rotation phase (Range: [0º - 360º])"
        self.rotation.setToolTip(info)
        rotation_label = QLabel("rotation [º]")
        rotation_label.setToolTip(info)
        layout_angles.addRow(rotation_label, self.rotation)

        # Inclination of the rotation axis regarding the LoS [degrees]
        self.inclination = QSpinBox()
        self.inclination.setMinimum(-90)
        self.inclination.setMaximum(90)
        self.inclination.setValue(90)
        info = ("Angle between (sub)stellar object rotation axis and the LoS"
                + " (Range: [-90º - 90º])")
        self.inclination.setToolTip(info)
        inclination_label = QLabel("inclination [º]")
        inclination_label.setToolTip(info)
        layout_angles.addRow(inclination_label, self.inclination)

        form_group_box_angles.setLayout(layout_angles)

        #######################################################################
        form_group_box_center = QGroupBox()
        h_layout = QHBoxLayout()
        layout_center_1 = QFormLayout()
        layout_center_1.setContentsMargins(0, 0, 30, 0)
        layout_center_2 = QFormLayout()

        self.L = QLineEdit()
        self.L.setValidator(QIntValidator())
        self.L.setText("30")
        info = "Length of the cubic grid sides. Units: [R*]"
        self.L.setToolTip(info)
        L_label = QLabel("L [R*]")
        L_label.setToolTip(info)
        layout_center_1.addRow(L_label, self.L)

        self.frequency = QLineEdit()
        self.frequency.setValidator(QDoubleValidator())
        self.frequency.setText("5")
        info = "GyroFrequency of electrons [GHz]"
        self.frequency.setToolTip(info)
        frequency_label = QLabel("Frequency [GHz]")
        frequency_label.setToolTip(info)
        layout_center_1.addRow(frequency_label, self.frequency)

        self.Robj2Rsun = QLineEdit()
        self.Robj2Rsun.setValidator(QDoubleValidator())
        self.Robj2Rsun.setText("4")
        info = ("Radius of the (sub)stellar object compared to the radius"
                + " of the Sun;\n(Dimensionless)")
        self.Robj2Rsun.setToolTip(info)
        Robj2Rsun_label = QLabel("Robj2Rsun")
        Robj2Rsun_label.setToolTip(info)
        layout_center_1.addRow(Robj2Rsun_label, self.Robj2Rsun)

        self.Bp = QLineEdit()
        self.Bp.setValidator(QIntValidator())
        self.Bp.setText("3000")
        info = ("Magnetic field strength at the pole of"
                + " the (sub)stellar object [Gauss]")
        self.Bp.setToolTip(info)
        Bp_label = QLabel("Bp [Gauss]")
        Bp_label.setToolTip(info)
        layout_center_1.addRow(Bp_label, self.Bp)

        self.r_alfven = QLineEdit()
        self.r_alfven.setValidator(QDoubleValidator())
        self.r_alfven.setText("16")
        info = "Averaged Alfvén Radius. Units: [R*]"
        self.r_alfven.setToolTip(info)
        r_alfven_label = QLabel("R_alfven [R*]")
        r_alfven_label.setToolTip(info)
        layout_center_1.addRow(r_alfven_label, self.r_alfven)

        self.l_middlemag = QLineEdit()
        self.l_middlemag.setValidator(QDoubleValidator())
        self.l_middlemag.setText("4")
        info = "Thickness of middle-magnetosphere. Units: [R*]"
        self.l_middlemag.setToolTip(info)
        l_middlemag_label = QLabel("l_middlemag [R*]")
        l_middlemag_label.setToolTip(info)
        layout_center_1.addRow(l_middlemag_label, self.l_middlemag)

        self.acc_eff = QLineEdit()
        self.acc_eff.setValidator(QDoubleValidator())
        self.acc_eff.setText("0.002")
        info = ("Acceleration efficiency of electrons in the"
                + " middle-magnetosphere: r_ne = Ne / neA);\n(Dimensionless)")
        self.acc_eff.setToolTip(info)
        acc_eff_label = QLabel("Acceleration Efficiency")
        acc_eff_label.setToolTip(info)
        layout_center_1.addRow(acc_eff_label, self.acc_eff)

        self.delta = QLineEdit()
        self.delta.setValidator(QDoubleValidator())
        self.delta.setText("2")
        info = "Spectral index of non-thermal electron energy distribution"
        self.delta.setToolTip(info)
        delta_label = QLabel("δ")
        delta_label.setToolTip(info)
        layout_center_2.addRow(delta_label, self.delta)

        # Distance to the (sub)stellar object [cm]
        self.D = QLineEdit()
        self.D.setValidator(QDoubleValidator())
        self.D.setText("352")
        info = "Distance from Earth to the studied (sub)stellar object [Pc]"
        self.D.setToolTip(info)
        D_label = QLabel("Distance [Pc]")
        D_label.setToolTip(info)
        layout_center_2.addRow(D_label, self.D)

        # Rotation Period of the (sub)stellar object [days]
        self.P_rot = QLineEdit()
        self.P_rot.setValidator(QDoubleValidator())
        self.P_rot.setText("1")
        info = "Rotation period of the (sub)stellar object [days]"
        self.P_rot.setToolTip(info)
        Prot_label = QLabel("P_rot [days]")
        Prot_label.setToolTip(info)
        layout_center_2.addRow(Prot_label, self.P_rot)

        self.v_inf = QLineEdit()
        self.v_inf.setValidator(QIntValidator())
        self.v_inf.setText("600")
        info = "(sub)stellar object wind velocity close to 'infinity' [km/s]"
        self.v_inf.setToolTip(info)
        vinf_label = QLabel("v_inf [km/s]")
        vinf_label.setToolTip(info)
        layout_center_2.addRow(vinf_label, self.v_inf)

        # Density of electrons of the plasma in the inner magnetosphere
        self.n_p0 = QLineEdit()
        self.n_p0.setValidator(QDoubleValidator())
        self.n_p0.setText("0")
        info = ("Plasma electron density, in inner-magnetosphere,"
                + " at the (sub)stellar object surface [cm^(−3)]")
        self.n_p0.setToolTip(info)
        np_label = QLabel("np [cm^(−3)]")
        np_label.setToolTip(info)
        layout_center_2.addRow(np_label, self.n_p0)

        # Plasma temperature in the inner magnetosphere [K]
        self.T_p0 = QLineEdit()
        self.T_p0.setValidator(QIntValidator())
        self.T_p0.setText("0")
        info = ("Plasma temperature in inner-magnetosphere,"
                + " at the (sub)stellar object surface [K]")
        self.T_p0.setToolTip(info)
        Tp_label = QLabel("Tp [K]")
        Tp_label.setToolTip(info)
        layout_center_2.addRow(Tp_label, self.T_p0)

        h_layout.addLayout(layout_center_1)
        h_layout.addLayout(layout_center_2)
        form_group_box_center.setLayout(h_layout)

        # Points per cube side ################################################

        # n_3d, n_2d and n_1d: Number of voxels per cube side: n shall be odd
        # (n_3d, n_2d, n_1d) allowing having one of the voxels in the
        # middle of the (sub)stellar object (which is removed for the
        # computations)

        box_3d = QGroupBox()
        v_layout_3d = QFormLayout()
        self.checkbox_3d = QCheckBox("3D Magnetic Field")
        self.checkbox_3d.setToolTip("3D plot of the magnetic vectorial field")
        v_layout_3d.addRow(self.checkbox_3d)
        self.n_3d = QSpinBox()
        self.n_3d.setMinimum(3)
        self.n_3d.setValue(7)
        self.n_3d.setSingleStep(2)
        self.n_3d.editingFinished.connect(
            lambda: self.on_value_changed(self.n_3d))
        info = ("Number of points per cube side (3D magnetic field"
                " computation)\nOnly odd values accepted")
        self.n_3d.setToolTip(info)
        label_3d = QLabel("n:")
        label_3d.setToolTip(info)
        v_layout_3d.addRow(label_3d, self.n_3d)
        box_3d.setLayout(v_layout_3d)

        box_2d = QGroupBox()
        v_layout_2d = QFormLayout()
        self.checkbox_2d = QCheckBox("2D Specific Intensities")
        self.checkbox_2d.setChecked(True)
        self.checkbox_2d.setToolTip("2D image of the specific intensities in"
                                    + " the plane perpendicular to the LoS\n"
                                    + " at the specific rotation phase"
                                    + " indicated by the user")
        v_layout_2d.addRow(self.checkbox_2d)
        self.n_2d = QSpinBox()
        self.n_2d.setMinimum(3)
        self.n_2d.setMaximum(1000)
        self.n_2d.setValue(21)
        self.n_2d.setSingleStep(2)
        self.n_2d.editingFinished.connect(
            lambda: self.on_value_changed(self.n_2d))
        info = ("Number of points per cube side (2D specific intensities"
                + " computation)\nOnly odd values accepted")
        self.n_2d.setToolTip(info)
        label_2d = QLabel("n:")
        label_2d.setToolTip(info)
        v_layout_2d.addRow(label_2d, self.n_2d)
        box_2d.setLayout(v_layout_2d)

        box_1d = QGroupBox()
        v_layout_1d = QFormLayout()
        self.checkbox_1d = QCheckBox("1D Flux Densities")
        self.checkbox_1d.setToolTip("1D Flux Densities (Light Curve) along a"
                                    + " complete rotation of 360º")
        v_layout_1d.addRow(self.checkbox_1d)
        self.n_1d = QSpinBox()
        self.n_1d.setMinimum(3)
        self.n_1d.setMaximum(1000)
        self.n_1d.setValue(7)
        self.n_1d.setSingleStep(2)
        self.n_1d.editingFinished.connect(
            lambda: self.on_value_changed(self.n_1d))
        info = ("Number of points per cube side (1D flux densities"
                + " computation)\nOnly odd values accepted")
        self.n_1d.setToolTip(info)
        label_1d = QLabel("n:")
        label_1d.setToolTip(info)
        v_layout_1d.addRow(label_1d, self.n_1d)
        box_1d.setLayout(v_layout_1d)

        h_layout = QHBoxLayout()
        h_layout.addWidget(box_3d)
        h_layout.addWidget(box_2d)
        h_layout.addWidget(box_1d)

        #######################################################################

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.close)

        main_layout = QVBoxLayout()
        main_layout.addWidget(form_group_box_angles)
        main_layout.addWidget(form_group_box_center)
        main_layout.addLayout(h_layout)
        main_layout.addWidget(button_box)

        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def on_value_changed(self, n):
        val = n.value()
        if val % 2 == 0:
            if n == self.n_1d:
                n_label = "'n' (for flux densities computation)"
                self.n_1d.setValue(val - 1)
            elif n == self.n_2d:
                n_label = "'n' (for 2D specific intensities computation)"
                self.n_2d.setValue(val - 1)
            elif n == self.n_3d:
                n_label = "'n' (for 3D magnetic vector field computation)"
                self.n_3d.setValue(val - 1)
            print("\nOnly odd values allowed:\n"
                  + n_label + " has been set to: " + str(val-1) + "\n")

    def accept(self):
        self.setWindowTitle("[PROCESSING...]")

        L = int(self.L.text())
        frequency = float(self.frequency.text()) * 1e9  # [Hz]
        Robj2Rsun = float(self.Robj2Rsun.text())
        r_alfven = float(self.r_alfven.text())
        Bp = int(self.Bp.text())
        l_middlemag = float(self.l_middlemag.text())
        acc_eff = float(self.acc_eff.text())
        delta = float(self.delta.text())
        D = float(self.D.text())
        P_rot = float(self.P_rot.text())
        v_inf = int(self.v_inf.text())
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
                r_ne=acc_eff, v_inf=v_inf, plot3d=True)

        if self.checkbox_2d.isChecked():
            specific_intensities_2D(
                L=L, n=self.n_2d.value(), beta=self.beta.value(),
                rotation_angle=self.rotation.value(),
                inclination=self.inclination.value(),
                Robj_Rsun_scale=Robj2Rsun, Bp=Bp, Pr=P_rot, D_pc=D,
                f=frequency, Ra=r_alfven, l_middlemag=l_middlemag, δ=delta,
                r_ne=acc_eff, v_inf=v_inf, plot3d=False)

        if self.checkbox_1d.isChecked():
            flux_densities_1D(
                L=L, n=self.n_1d.value(), beta=self.beta.value(),
                inclination=self.inclination.value(),
                Robj_Rsun_scale=Robj2Rsun, Bp=Bp, Pr=P_rot, D_pc=D,
                f=frequency, Ra=r_alfven, l_middlemag=l_middlemag, δ=delta,
                r_ne=acc_eff, v_inf=v_inf, plot3d=False)
        # End launching application ###########################################

        self.setWindowTitle(
            "(Sub)Stellar Object Radio Emission")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    initial_gui = InitialGUI()
    initial_gui.show()
    sys.exit(app.exec())

