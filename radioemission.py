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
    QApplication, QMainWindow, QWidget, QPushButton, QLineEdit, QLabel,
    QSpinBox, QCheckBox, QComboBox, QDialogButtonBox, QGroupBox, QFormLayout,
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
        self.setWindowTitle("Radio Emission opening GUI")
        self.move(400, 250)

        group_box = QGroupBox()
        layout = QVBoxLayout()

        button_compute_Ra = QPushButton("Compute Alfvén Radius and neA",
                                        self)
        info = ("Computation of the Alfvén Radius (Ra) and the \ndensity of"
                + " electrons at the Alfvén Radius (neA)")
        button_compute_Ra.setToolTip(info)
        button_compute_Ra.setDefault(True)

        button_specific_intensities = QPushButton(
            "Compute Specific Intensities and Flux Densities", self)
        button_specific_intensities.setToolTip(
            "Compute specific intensities & flux density\n"
            "(Alfvén Radius and neA already known)")
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
        self.setWindowTitle("Alfvén Radius GUI")
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
        self.Bp.setValidator(QDoubleValidator())
        self.Bp.setText("1e4")
        info = ("Magnetic field strength at the pole of"
                + " the (sub)stellar object [Gauss]")
        self.Bp.setToolTip(info)
        Bp_label = QLabel("Bp [Gauss]")
        Bp_label.setToolTip(info)
        layout_center.addRow(Bp_label, self.Bp)

        self.v_inf = QLineEdit()
        self.v_inf.setValidator(QDoubleValidator())
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
        self.checkbox_Ra_at_zeta = QCheckBox("Ra and neA at a given ζ")
        self.checkbox_Ra_at_zeta.setToolTip(
            "Ra and neA at a given ζ (zeta);\n"
            "Alfvén Radius (Ra) and density of electrons at the Alfvén"
            " Radius (neA) computed at the specific magnetic longitude"
            " (ζ);\nNOTE: This computation can take a few minutes (< 5min).")
        self.checkbox_Ra_at_zeta.setChecked(True)
        self.checkbox_Ra_at_zeta.toggled.connect(self.unset_averaged_Ra)
        v_layout.addRow(self.checkbox_Ra_at_zeta)

        self.checkbox_averaged_Ra = QCheckBox("Averaged Ra, B(Ra) and neA")
        self.checkbox_averaged_Ra.setToolTip(
            "Compute the averaged Alfvén Radius (Ra), thanks to a 1D Ra plot,"
            "\n (more precise, but slower computation);\n"
            "Ra averaged over the range of magnetic longitudes [0º-360º],\n"
            " (one single value of Ra taken each 10º);\n "
            "It also computes the 1D plots and averages for B(Ra) and neA:\n"
            " magnetic field and density of electrons at the Alfvén Radius;\n"
            "NOTE: This computation can take long (from ~15min to 1 hour)")
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

        self.setWindowTitle("Alfvén Radius GUI")


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

        # Inclination of the rotation axis regarding the LoS [degrees]
        self.inclination = QSpinBox()
        self.inclination.setMinimum(0)
        self.inclination.setMaximum(180)
        self.inclination.setValue(25)
        info = ("Angle between (sub)stellar object rotation axis and the LoS"
                + " (Range: [0º - 180º])")
        self.inclination.setToolTip(info)
        inclination_label = QLabel("inclination [º]")
        inclination_label.setToolTip(info)
        layout_angles.addRow(inclination_label, self.inclination)

        # Angle between magnetic and rotation axes [degrees]
        self.beta = QSpinBox()
        self.beta.setMinimum(-180)
        self.beta.setMaximum(180)
        self.beta.setValue(65)
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
        info = "(Sub)stellar object rotation angle (Range: [0º - 360º])"
        self.rotation.setToolTip(info)
        rotation_label = QLabel("rotation [º]")
        rotation_label.setToolTip(info)
        layout_angles.addRow(rotation_label, self.rotation)

        # Rotation phase offset [degrees]
        self.rotation_phase = QSpinBox()
        self.rotation_phase.setMinimum(0)
        self.rotation_phase.setMaximum(360)
        self.rotation_phase.setValue(180)
        info = ("Rotation phase offset added to the angle of rotation.\n"
                + "Useful to match different phase conventions used"
                + " by different authors or in different published results.\n"
                + "(Range: [0º - 360º])")
        self.rotation_phase.setToolTip(info)
        rotation_phase_label = QLabel("rotation offset [º]")
        rotation_phase_label.setToolTip(info)
        layout_angles.addRow(rotation_phase_label, self.rotation_phase)

        form_group_box_angles.setLayout(layout_angles)

        #######################################################################
        form_group_box_center = QGroupBox()
        h_layout = QHBoxLayout()
        layout_center_1 = QFormLayout()
        layout_center_1.setContentsMargins(0, 0, 30, 0)
        layout_center_2 = QFormLayout()

        self.L = QLineEdit()
        self.L.setValidator(QIntValidator())
        self.L.setText("40")
        info = "Length of the cubic grid sides (FOV length). Units: [R*]"
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
        self.Bp.setValidator(QDoubleValidator())
        self.Bp.setText("7700")
        info = ("Magnetic field strength at the pole of"
                + " the (sub)stellar object [Gauss]")
        self.Bp.setToolTip(info)
        Bp_label = QLabel("Bp [Gauss]")
        Bp_label.setToolTip(info)
        layout_center_1.addRow(Bp_label, self.Bp)

        self.r_alfven = QLineEdit()
        self.r_alfven.setValidator(QDoubleValidator())
        self.r_alfven.setText("15")
        info = "Averaged Alfvén Radius (Ra). Units: [R*]"
        self.r_alfven.setToolTip(info)
        r_alfven_label = QLabel("Ra [R*]")
        r_alfven_label.setToolTip(info)
        layout_center_1.addRow(r_alfven_label, self.r_alfven)

        self.l_middlemag = QLineEdit()
        self.l_middlemag.setValidator(QDoubleValidator())
        self.l_middlemag.setText("7")
        info = ("Length of the current sheets created just after the Alfvén"
                + " surface (equatorial thickness of the magnetic shell"
                + " [Trigilio04]). Units: [R*]")
        self.l_middlemag.setToolTip(info)
        l_middlemag_label = QLabel("l_middlemag [R*]")
        l_middlemag_label.setToolTip(info)
        layout_center_1.addRow(l_middlemag_label, self.l_middlemag)

        # Distance to the (sub)stellar object [cm]
        self.D = QLineEdit()
        self.D.setValidator(QDoubleValidator())
        self.D.setText("373")
        info = "Distance from Earth to the studied (sub)stellar object [pc]"
        self.D.setToolTip(info)
        D_label = QLabel("Distance [pc]")
        D_label.setToolTip(info)
        layout_center_1.addRow(D_label, self.D)

        self.neA = QLineEdit()
        self.neA.setValidator(QDoubleValidator())
        self.neA.setText("3e6")
        info = ("neA (or ne,A): Density of electrons at the Alfvén Radius.\n"
                + "Units: [cm^(-3)]")
        self.neA.setToolTip(info)
        neA = QLabel("neA (e- density at Ra)")
        neA.setToolTip(info)
        layout_center_2.addRow(neA, self.neA)

        self.acc_eff = QLineEdit()
        self.acc_eff.setValidator(QDoubleValidator())
        self.acc_eff.setText("0.0001")
        info = ("Acceleration efficiency of electrons in the"
                + " middle magnetosphere: r_ne = Ne / neA);\n(Dimensionless)")
        self.acc_eff.setToolTip(info)
        acc_eff_label = QLabel("Acceleration Efficiency")
        acc_eff_label.setToolTip(info)
        layout_center_2.addRow(acc_eff_label, self.acc_eff)

        self.delta = QLineEdit()
        self.delta.setValidator(QDoubleValidator())
        self.delta.setText("2")
        info = "Spectral index of non-thermal electron energy distribution"
        self.delta.setToolTip(info)
        delta_label = QLabel("δ")
        delta_label.setToolTip(info)
        layout_center_2.addRow(delta_label, self.delta)

        self.checkbox_innermag = QCheckBox("Inner Magnetosphere Contribution")
        self.checkbox_innermag.setChecked(True)
        info = ("Take into account the inner magnetosphere "
                + " contribution to absorption and emission")
        self.checkbox_innermag.setToolTip(info)
        layout_center_2.addRow(self.checkbox_innermag)

        # Density of electrons of the plasma in the inner magnetosphere
        self.n_p0 = QLineEdit()
        self.n_p0.setValidator(QDoubleValidator())
        self.n_p0.setText("3e9")
        info = ("Plasma electron density, in inner magnetosphere,"
                + " at the (sub)stellar object surface [cm^(−3)]")
        self.n_p0.setToolTip(info)
        np_label = QLabel("np [cm^(−3)]")
        np_label.setToolTip(info)
        layout_center_2.addRow(np_label, self.n_p0)

        # Plasma temperature in the inner magnetosphere [K]
        self.T_p0 = QLineEdit()
        self.T_p0.setValidator(QDoubleValidator())
        self.T_p0.setText("1e5")
        info = ("Plasma temperature in inner magnetosphere,"
                + " at the (sub)stellar object surface [K]")
        self.T_p0.setToolTip(info)
        Tp_label = QLabel("Tp [K]")
        Tp_label.setToolTip(info)
        layout_center_2.addRow(Tp_label, self.T_p0)

        h_layout.addLayout(layout_center_1)
        h_layout.addLayout(layout_center_2)
        form_group_box_center.setLayout(h_layout)

        # Points per cube side ################################################

        # n_3d, n_2d and n_1d: Number of voxels per cube side

        box_3d = QGroupBox()
        v_layout_3d = QFormLayout()
        self.checkbox_3d = QCheckBox("3D Magnetic Field")
        self.checkbox_3d.setToolTip("3D plot of the magnetic vectorial field")
        v_layout_3d.addRow(self.checkbox_3d)
        self.n_3d = QSpinBox()
        self.n_3d.setMinimum(3)
        self.n_3d.setValue(7)
        info = "Number of points per cube side (3D magnetic field plot)"
        self.n_3d.setToolTip(info)
        label_3d = QLabel("n:")
        label_3d.setToolTip(info)
        v_layout_3d.addRow(label_3d, self.n_3d)

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
        self.n_2d.setValue(31)
        info = ("Number of points per cube side (2D specific intensities"
                + " computation)")
        self.n_2d.setToolTip(info)
        label_2d = QLabel("n:")
        label_2d.setToolTip(info)
        v_layout_2d.addRow(label_2d, self.n_2d)

        # Scale colors of 2D plot
        self.scale_2D_colors = QLineEdit()
        self.scale_2D_colors.setValidator(QIntValidator())
        self.scale_2D_colors.setText("1")
        info = "Scale grey colors to make the 2D plot more visible"
        self.scale_2D_colors.setToolTip(info)
        scale_2D_label = QLabel("Scale colors")
        scale_2D_label.setToolTip(info)
        v_layout_2d.addRow(scale_2D_label, self.scale_2D_colors)

        # Linear or logarithmic color map for specific intensities 2D plot
        self.colormap = QComboBox()
        self.colormap.addItem("Linear")
        self.colormap.addItem("Logarithmic")
        info = ("Color Map: Linear or logarithmic color map for"
                + " specific intensities 2D plot")
        self.colormap.setToolTip(info)
        colormap = QLabel("Color Map")
        colormap.setToolTip(info)
        v_layout_2d.addRow(colormap, self.colormap)

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
        info = ("Number of points per cube side (1D flux densities"
                + " computation)")
        self.n_1d.setToolTip(info)
        label_1d = QLabel("n:")
        label_1d.setToolTip(info)
        v_layout_1d.addRow(label_1d, self.n_1d)

        # Rotation angle step [º] for flux densitiy 1D plot
        self.step_angle_1D = QLineEdit()
        self.step_angle_1D.setValidator(QIntValidator())
        self.step_angle_1D.setText("10")
        info = ("Rotation angle step [º]: abscissa axis step for the flux"
                + " densities 1D plot\n"
                + "(Plot from 0º to 360º in steps of 'Stepº')")
        self.step_angle_1D.setToolTip(info)
        step_angle_label = QLabel("Step [º]")
        step_angle_label.setToolTip(info)
        v_layout_1d.addRow(step_angle_label, self.step_angle_1D)

        self.checkbox_use_symmetry = QCheckBox("Use symmetry on 180º")
        info = ("Use the symmetry of flux densities 1D plot at 180º rotation "
                + "angle, to extrapolate the curve from 180º to 360º;\n"
                + "Computation time is cut in half")
        self.checkbox_use_symmetry.setChecked(True)
        self.checkbox_use_symmetry.setToolTip(info)
        v_layout_1d.addRow(self.checkbox_use_symmetry)

        h_layout = QHBoxLayout()
        box_3d.setLayout(v_layout_3d)
        h_layout.addWidget(box_3d)

        box_2d.setLayout(v_layout_2d)
        h_layout.addWidget(box_2d)

        box_1d.setLayout(v_layout_1d)
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

    def accept(self):
        self.setWindowTitle("[PROCESSING...]")

        L = int(self.L.text())
        frequency = float(self.frequency.text()) * 1e9  # [Hz]
        Robj2Rsun = float(self.Robj2Rsun.text())
        r_alfven = float(self.r_alfven.text())
        Bp = float(self.Bp.text())
        l_middlemag = float(self.l_middlemag.text())
        neA = float(self.neA.text())
        acc_eff = float(self.acc_eff.text())
        delta = float(self.delta.text())
        D = float(self.D.text())
        inner_contrib = self.checkbox_innermag.isChecked()
        n_p0 = float(self.n_p0.text())
        T_p0 = float(self.T_p0.text())

        # Launching application with inputs entered by the user ###############
        if self.checkbox_3d.isChecked():
            plot_3D(
                L=L, n=self.n_3d.value(), beta=self.beta.value(),
                rotation_angle=self.rotation.value(),
                inclination=self.inclination.value(),
                Robj_Rsun_scale=Robj2Rsun, Bp=Bp, D_pc=D,
                f=frequency, Ra=r_alfven, l_middlemag=l_middlemag, δ=delta,
                r_ne=acc_eff, plot3d=True)

        if self.checkbox_2d.isChecked():
            scale_colors = int(self.scale_2D_colors.text())
            colormap = self.colormap.currentText()
            specific_intensities_2D(
                L=L, n=self.n_2d.value(), inclination=self.inclination.value(),
                beta=self.beta.value(), rotation_angle=self.rotation.value(),
                rotation_offset=self.rotation_phase.value(),
                Robj_Rsun_scale=Robj2Rsun, Bp=Bp, D_pc=D,
                f=frequency, Ra=r_alfven, l_middlemag=l_middlemag, δ=delta,
                neA=neA, r_ne=acc_eff, inner_contrib=inner_contrib, n_p0=n_p0,
                T_p0=T_p0, scale_colors=scale_colors, colormap=colormap,
                plot3d=False)

        if self.checkbox_1d.isChecked():
            use_symmetry = self.checkbox_use_symmetry.isChecked()
            step_angle = int(self.step_angle_1D.text())
            flux_densities_1D(
                L=L, n=self.n_1d.value(), inclination=self.inclination.value(),
                beta=self.beta.value(),
                rotation_offset=self.rotation_phase.value(),
                Robj_Rsun_scale=Robj2Rsun, Bp=Bp, D_pc=D,
                f=frequency, Ra=r_alfven, l_middlemag=l_middlemag, δ=delta,
                neA=neA, r_ne=acc_eff, rotation_angle_step=step_angle,
                inner_contrib=inner_contrib, n_p0=n_p0, T_p0=T_p0,
                use_symmetry=use_symmetry, plot3d=False)
        self.setWindowTitle("(Sub)Stellar Object Radio Emission")
        # End launching application ###########################################


if __name__ == '__main__':
    app = QApplication(sys.argv)
    initial_gui = InitialGUI()
    initial_gui.show()
    sys.exit(app.exec())

