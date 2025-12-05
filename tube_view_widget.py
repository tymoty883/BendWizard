from PyQt5.QtWidgets import QOpenGLWidget, QInputDialog
from OpenGL.GL import *
import numpy as np
from tube_utils import compute_clearance, optimize_inner_path
from color_utils import ColorUtils

class TubeViewWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.outer_centerline = None
        self.outer_radii = None
        self.inner_path = None
        self.inner_radius = None
        self.clearance = None
        self.highlight_points = None

    def fit_inner_tube(self):
        # Ask user for radius
        radius, ok = QInputDialog.getDouble(self, "Inner Tube Radius", "Enter radius (m):", 0.1, 0.01, 10.0, 3)
        if not ok:
            return
        self.inner_radius = radius
        # Compute optimized path
        self.inner_path = optimize_inner_path(self.outer_centerline, self.outer_radii, self.inner_radius)
        # Compute clearance
        self.clearance = compute_clearance(self.inner_path, self.outer_centerline, self.outer_radii, self.inner_radius)
        self.update()

    def paintGL(self):
        # Draw outer tube (centerline)
        if self.outer_centerline is not None:
            glColor4f(0.3, 0.3, 1.0, 1.0)
            glBegin(GL_LINE_STRIP)
            for pt in self.outer_centerline:
                glVertex3f(*pt)
            glEnd()
        # Draw inner tube (optimized path)
        if self.inner_path is not None:
            for i, pt in enumerate(self.inner_path):
                color = ColorUtils.get_gradient_color(self.clearance[i])
                glColor4f(*color)
                glPushMatrix()
                glTranslatef(*pt)
                glutSolidSphere(self.inner_radius, 8, 8)
                glPopMatrix()
