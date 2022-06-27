"""
Usage of the isosurface function to convert a scalar field
(a hydrogen orbital) into a mesh for 3D display.
"""
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

app = pg.mkQApp("Isosurface")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle("Isosurface")

w.setCameraPosition(distance=40)

#############################################
# Add LoS Reference Grid to 3D plot
g = gl.GLGridItem()
g.scale(2, 2, 1)
w.addItem(g)


#############################################
# Add Dipole Field to 3D plot
# Define a scalar field from which we will generate an isosurface
def psi(i, j, k, offset=(25, 25, 50)):
    x = i - offset[0]
    y = j - offset[1]
    z = k - offset[2]
    th = np.arctan2(z, np.hypot(x, y))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    a0 = 1
    ps = (1./81.) * 1./(6.*np.pi)**0.5 * (1./a0)**(3/2) * (r/a0)**2 * (
            np.exp(-r / (3*a0)) * (3 * np.cos(th) ** 2))
    return ps


print("Generating scalar field..")
data = np.abs(np.fromfunction(psi, (50, 50, 100)))

print("Generating isosurface..")
verts, faces = pg.isosurface(data, data.max()/4.)
md = gl.MeshData(vertexes=verts, faces=faces)

colors = np.ones((md.faceCount(), 4), dtype=float)
colors[:, 3] = 0.2
colors[:, 2] = np.linspace(0, 1, colors.shape[0])
md.setFaceColors(colors)

m1 = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
m1.setGLOptions('additive')

w.addItem(m1)
m1.translate(-25, -25, -50)

#############################################
# Add Magnetic Field Dipole Axis to 3D plot
xx = 0
yx = 0
zx = -20

xy = 0
yy = 0
zy = 20

Xdot = (xx, yx, zx)
Ydot = (xy, yy, zy)

pts = np.array([Xdot, Ydot])
dipole_axis = gl.GLLinePlotItem(pos=pts, width=10, antialias=False)
dipole_axis.setData(color="yellow")
w.addItem(dipole_axis)

#############################################
# Add Star Rotation Axis to 3D plot
xx = 0
yx = 0
zx = -20

xy = 0
yy = 0
zy = 20

Xdot = (xx, yx, zx)
Ydot = (xy, yy, zy)

pts = np.array([Xdot, Ydot])
rotation_axis = gl.GLLinePlotItem(pos=pts, width=10, antialias=False)
rotation_axis.setData(color="purple")
w.addItem(rotation_axis)


m1.rotate(20, 0, 1, 0)
dipole_axis.rotate(20, 0, 1, 0)

# rotation_axis(0, 0, 1, 0)


#############################################
def start():
    QtGui.QApplication.instance().exec_()


def update():
    angle = 1
    for i in range(72):
        angle = angle + 5
        m1.rotate(i, 0, 0, 1)
        dipole_axis.rotate(i, 0, 0, 1)


def animation():
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(100)
    start()


if __name__ == '__main__':
    animation()
