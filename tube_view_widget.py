"""
OpenGL-based widget for rendering pipe and borehole geometry and interactions.
Handles all 3D visualization, camera control, and user interaction.
"""

import math
import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QColor, QFont, QPainter
from OpenGL.GL import *
from OpenGL.GLU import *
import constants
from tube_generator import TubeGenerator
from color_utils import ColorUtils

class TubeViewWidget(QOpenGLWidget):
    """
    3D OpenGL widget for visualizing pipe-borehole interaction.
    Provides camera controls, rendering, and user interaction.
    """

    def set_pipe_stress_data(self, stress_data):
        self.pipe_stress_data = stress_data
        self.update() 

    def set_radius_classification_lowest_step(self, lowest_step: float) -> None:
        """Set custom lowest threshold for borehole curvature color classes."""
        self.radius_classification_lowest_step = float(lowest_step)
        self.update()

    def _get_custom_radius_thresholds(self):
        """Build fixed class thresholds shifted by user-selected lowest threshold."""
        if self.radius_classification_lowest_step is None:
            return None

        base = float(self.radius_classification_lowest_step)
        default_base = float(constants.STRESS_COLOR_THRESHOLD_VERY_LOW)
        offsets = (
            float(constants.STRESS_COLOR_THRESHOLD_VERY_LOW) - default_base,
            float(constants.STRESS_COLOR_THRESHOLD_LOW) - default_base,
            float(constants.STRESS_COLOR_THRESHOLD_MEDIUM) - default_base,
            float(constants.STRESS_COLOR_THRESHOLD_HIGH) - default_base,
        )
        return tuple(base + offset for offset in offsets)




    def __init__(self, parent=None):
        super(TubeViewWidget, self).__init__(parent)
        self.setMinimumSize(800, 600)
        self.pipe_stress_data = None

        
        self.centerline = None
        self.world_centerline = None
        self.drill_centerline = None
        self.outer_tube_segments = []
        self.drill_tube_segments = []
        self.rings = []
        self.distance_labels = []
        self.segment_labels = []
        self.segment_radii = []
        self.radius_scale_min = None
        self.radius_scale_max = None
        self.radius_average = None
        self.radius_outlier_threshold = None
        self.radius_quantile_thresholds = None
        self.radius_quantile_edges = None
        self.radius_classification_lowest_step = None
        self.render_origin = np.array([0.0, 0.0, 0.0], dtype=float)
        self.render_scale = 1.0
        
        # Camera settings
        self.camera_distance = 1000
        self.camera_elevation = 20
        self.camera_azimuth = -35
        self.center_xyz = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # Preset view points
        self.start_point = None
        self.middle_point = None
        self.end_point = None
        
        self.last_mouse_pos = None
        self.setMouseTracking(True)
        
        # Visibility flags
        self.show_drill_tube = False
        self.show_outer_tube = False
        self.show_distance_labels = True
        self.show_segment_numbers = False
        self.show_force_labels = True  # <--- NEW for force labels
        
        # FPS counter setup
        self.fps = 0
        self.frame_count = 0
        self.last_fps_update = QTime.currentTime()
        
        # Update FPS every second
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.updateFPS)
        self.fps_timer.start(1000)  # Update every 1000ms (1 second)

    def _compute_render_transform(self, centerline: np.ndarray) -> None:
        """Compute translation and scale to render large-coordinate data in a local frame."""
        self.render_origin = np.mean(centerline, axis=0)
        extent = np.max(np.max(centerline, axis=0) - np.min(centerline, axis=0))
        target_extent = 2000.0
        if extent > 0:
            self.render_scale = min(1.0, target_extent / extent)
        else:
            self.render_scale = 1.0

    def _to_render_coords(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates (e.g. UTM) to local render coordinates."""
        return (points - self.render_origin) * self.render_scale


        
    def set_centerline(self, centerline, reset_view=True):
        self.centerline = centerline
        
        # Calculate the center of the centerline for initial camera positioning
        if len(centerline) > 0:
            center = np.mean(centerline, axis=0)
            
            # Set preset view points
            self.start_point = centerline[0]
            self.middle_point = centerline[len(centerline) // 2]
            self.end_point = centerline[-1]
            
            # Only update center_xyz on first load or when explicitly requested
            if not hasattr(self, 'center_xyz') or reset_view:
                self.center_xyz = center
                # Set initial view to middle point only on first load
                self.view_middle_point()
            
            self.max_xyz = np.max(centerline, axis=0)
            self.min_xyz = np.min(centerline, axis=0)
            self.tube_size = np.max(self.max_xyz - self.min_xyz)
            self.default_distance = max(1000, self.tube_size * 1.5)
            
            # Only set camera distance on first load
            if not hasattr(self, 'camera_distance') or reset_view:
                self.camera_distance = self.default_distance
            
    def initializeGL(self):
        glClearColor(0.5, 0.8, 1.0, 1.0)  # Sky blue background
        
        # Enable higher quality rendering
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Enable depth testing with proper configuration for better precision
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)  # Less than or equal for better precision
        glDepthMask(GL_TRUE)  # Enable writing to the depth buffer
        
        # Set higher precision for depth buffer
        glClearDepth(1.0)
        glDepthRange(0.0, 1.0)
        
        # Keep polygon offset disabled globally; enable only for specific mesh passes.
        glDisable(GL_POLYGON_OFFSET_FILL)
        
        # Disable face culling to allow seeing inside the tubes
        glDisable(GL_CULL_FACE)
        
        # Enable smooth shading for better appearance
        glShadeModel(GL_SMOOTH)
        
        # Configure anti-aliasing
        
        glEnable(GL_MULTISAMPLE)  # Enable multisampling if available
        
        # Enable lighting with better settings
        glEnable(GL_LIGHTING)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)  # Proper two-sided lighting
        
        # Configure main light
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        
        # Add a second light from another direction
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [-1, -1, 1, 0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.5, 0.5, 0.5, 1])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.3, 0.3, 0.3, 1])
        
        # Enable color material for simpler material setup
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Enable blending for transparent objects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable line smoothing
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Enable automatic normalization of normals
        glEnable(GL_NORMALIZE)
        
        # GL_POLYGON_SMOOTH together with blending frequently causes shimmering.
        glDisable(GL_POLYGON_SMOOTH)
        
    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        self.aspect_ratio = width / height if height > 0 else 1.0
        self.updateProjection()
        
    def updateProjection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Tight near/far planes improve depth buffer precision and reduce z-fighting.
        scene_extent = max(100.0, float(getattr(self, 'tube_size', 1000.0)))
        near_plane = max(0.5, self.camera_distance / 50.0)
        far_plane = self.camera_distance + (scene_extent * 8.0)
        if far_plane <= near_plane:
            far_plane = near_plane + 1000.0
        
        gluPerspective(60, self.aspect_ratio, near_plane, far_plane)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Increment frame counter for FPS calculation
        self.frame_count += 1
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Set up camera
        eye_x = self.center_xyz[0] + self.camera_distance * np.cos(np.radians(self.camera_azimuth)) * np.cos(np.radians(self.camera_elevation))
        eye_y = self.center_xyz[1] + self.camera_distance * np.sin(np.radians(self.camera_azimuth)) * np.cos(np.radians(self.camera_elevation))
        eye_z = self.center_xyz[2] + self.camera_distance * np.sin(np.radians(self.camera_elevation))
        
        gluLookAt(
            eye_x, eye_y, eye_z,                           # Eye position
            self.center_xyz[0], self.center_xyz[1], self.center_xyz[2],  # Look at point
            0, 0, 1                                        # Up vector
        )
        
        # Store current modelview and projection matrices for use in 2D text rendering
        self.modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Draw grid
        self.drawGrid()
        
        # Draw centerline
        if self.centerline is not None and hasattr(self, 'centerline_display_list'):
            glCallList(self.centerline_display_list)
        
        # Ensure depth testing is properly configured
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_TRUE)
        
        # Draw outer tube segments first (opaque objects)
        if hasattr(self, 'outer_tube_display_list') and self.show_outer_tube:
            glCallList(self.outer_tube_display_list)
        
        # Draw drill tube segments using display lists if visible and available
        if hasattr(self, 'drill_tube_display_list') and self.drill_tube_display_list is not None and self.show_drill_tube:
            glCallList(self.drill_tube_display_list)


         # Draw pipe segment model
        if hasattr(self, 'pipe_display_list') and self.pipe_display_list is not None:
            glCallList(self.pipe_display_list)
   
            
        # Draw rings using display lists
        if hasattr(self, 'rings_display_list'):
            glCallList(self.rings_display_list)

      
    def drawGrid(self):
        if not hasattr(self, 'center_xyz') or not hasattr(self, 'min_xyz') or not hasattr(self, 'max_xyz'):
            return
            
        glDisable(GL_LIGHTING)
        
        # Calculate grid boundaries based on object extents
        x_min, y_min, z_min = self.min_xyz
        x_max, y_max, z_max = self.max_xyz
        
        # Add some padding around the objects (in scaled render units)
        padding = 200 * self.render_scale
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        
        # Snap to grid lines
        grid_step = max(200 * self.render_scale, 1e-3)
        x_min = math.floor(x_min / grid_step) * grid_step
        x_max = math.ceil(x_max / grid_step) * grid_step
        y_min = math.floor(y_min / grid_step) * grid_step
        y_max = math.ceil(y_max / grid_step) * grid_step
        
        # Set grid height slightly below the minimum Z
        grid_z = z_min - 50
        
        # Draw horizontal grid
        glColor3f(1.0, 1.0, 1.0)  # White grid
        glLineWidth(1.0)
        
        glBegin(GL_LINES)
        # Draw lines along X axis
        for i in np.arange(x_min, x_max + grid_step, grid_step):
            glVertex3f(i, y_min, grid_z)
            glVertex3f(i, y_max, grid_z)
        
        # Draw lines along Y axis
        for i in np.arange(y_min, y_max + grid_step, grid_step):
            glVertex3f(x_min, i, grid_z)
            glVertex3f(x_max, i, grid_z)
        glEnd()
        
        # Draw vertical grid on XZ plane (optional)
        draw_vertical_grid = False
        if draw_vertical_grid:
            glColor4f(0.8, 0.8, 1.0, 0.6)  # Light blue grid
            
            grid_y = y_min - 50
            
            glBegin(GL_LINES)
            # Draw lines along X axis at different Z levels
            for i in range(int(x_min), int(x_max) + 1, grid_step):
                glVertex3f(i, grid_y, z_min)
                glVertex3f(i, grid_y, z_max)
            
            # Draw lines along Z axis at different X positions
            for i in range(int(z_min), int(z_max) + 1, grid_step):
                glVertex3f(x_min, grid_y, i)
                glVertex3f(x_max, grid_y, i)
            glEnd()
            
        glEnable(GL_LIGHTING)
        
    def drawCenterline(self):
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.0)  # Yellow
        glLineWidth(2.0)
        
        glBegin(GL_LINE_STRIP)
        for point in self.centerline:
            glVertex3f(point[0], point[1], point[2])
        glEnd()
        
        glEnable(GL_LIGHTING)
        
    def drawMesh(self, vertices, faces, normals, color):
        # Set the color for the mesh
        glColor4f(color[0], color[1], color[2], color[3])
        
        # Ensure depth testing is enabled with proper settings
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        
        # Enable polygon offset to prevent z-fighting
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(2.0, 4.0)  # Increased offset values
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Use material properties for better lighting and reflections
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [color[0]*0.3, color[1]*0.3, color[2]*0.3, color[3]])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [color[0], color[1], color[2], color[3]])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.8, 0.8, 0.8, 1.0])  # Increased specular reflection
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 128.0)  # Higher shininess for sharper reflections
        
        # Draw the mesh using immediate mode for compatibility
        glBegin(GL_TRIANGLES)
        for face in faces:
            for idx in face:
                glNormal3f(normals[idx][0], normals[idx][1], normals[idx][2])
                glVertex3f(vertices[idx][0], vertices[idx][1], vertices[idx][2])
        glEnd()
        
        # Disable polygon offset after drawing
        glDisable(GL_POLYGON_OFFSET_FILL)
        
    def drawRing(self, points, color):
        glDisable(GL_LIGHTING)
        # Keep depth testing enabled so rings are properly occluded
        glEnable(GL_DEPTH_TEST)
        
        # Draw thicker black lines for better visibility
        glColor3f(0.0, 0.0, 0.0)  # Pure black
        glLineWidth(2.0)  # Thicker and smoother line
        
        glBegin(GL_LINE_LOOP)
        for point in points:
            glVertex3f(point[0], point[1], point[2])
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        pass
        
    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return
            
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            # Rotate camera (reversed direction for more intuitive control)
            self.camera_azimuth -= dx * 0.5
            self.camera_elevation += dy * 0.5
            
            # Clamp elevation to avoid gimbal lock
            self.camera_elevation = max(-89, min(89, self.camera_elevation))
        elif event.buttons() & Qt.RightButton:
            # Pan camera
            pan_speed = max(self.camera_distance / 1000, 0.5)  # Minimum pan speed for responsiveness
            right = np.array([np.cos(np.radians(self.camera_azimuth + 90)), np.sin(np.radians(self.camera_azimuth + 90)), 0])
            up = np.array([0, 0, 1])
            
            self.center_xyz -= right * dx * pan_speed
            self.center_xyz += up * dy * pan_speed
        elif event.buttons() & Qt.MiddleButton:
            # Zoom camera
            zoom_speed = 0.01
            min_step = 10.0
            new_distance = self.camera_distance * (1 - dy * zoom_speed)
            # Enforce minimum step
            if abs(new_distance - self.camera_distance) < min_step:
                if new_distance < self.camera_distance:
                    new_distance = self.camera_distance - min_step
                else:
                    new_distance = self.camera_distance + min_step
            self.camera_distance = max(0.1, new_distance)
        
        self.last_mouse_pos = event.pos()
        self.update()
        
    def wheelEvent(self, event):
        # Zoom with mouse wheel
        zoom_factor = 1.1
        min_step = 10.0
        old_distance = self.camera_distance
        if event.angleDelta().y() > 0:
            # Zoom in
            new_distance = self.camera_distance / zoom_factor
            # Enforce minimum step
            if abs(new_distance - self.camera_distance) < min_step:
                new_distance = self.camera_distance - min_step
            self.camera_distance = max(1, new_distance)
        else:
            # Zoom out
            new_distance = self.camera_distance * zoom_factor
            if abs(new_distance - self.camera_distance) < min_step:
                new_distance = self.camera_distance + min_step
            self.camera_distance = max(1, new_distance)
        self.update()

    def updateFPS(self):
        current_time = QTime.currentTime()
        elapsed = self.last_fps_update.msecsTo(current_time)
        
        if elapsed > 0:
            self.fps = self.frame_count * 1000.0 / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
            
    def toggle_drill_tube(self):
        if not hasattr(self, 'drill_tube_display_list') or self.drill_tube_display_list is None:
            print("No drill tube data available to display")
            return
        
        self.show_drill_tube = not self.show_drill_tube
        self.update()
        print(f"Drill tube {'visible' if self.show_drill_tube else 'hidden'}")
        
    def toggle_outer_tube(self):
        self.show_outer_tube = not self.show_outer_tube
        self.update()
        print(f"Borehole {'visible' if self.show_outer_tube else 'hidden'}")
        
    def toggle_distance_labels(self) -> None:
        """Toggle distance labels visibility."""
        self.show_distance_labels = not self.show_distance_labels
        self.update()
        print(f"Distance labels {'visible' if self.show_distance_labels else 'hidden'}")

    def toggle_force_labels(self) -> None:
        """Toggle contact force labels visibility."""
        self.show_force_labels = not self.show_force_labels
        self.update()
        print(f"Force labels {'visible' if self.show_force_labels else 'hidden'}")
        
    def toggle_segment_numbers(self) -> None:
        """Toggle segment number labels visibility."""
        self.show_segment_numbers = not self.show_segment_numbers
        self.update()
        print(f"Segment numbers {'visible' if self.show_segment_numbers else 'hidden'}")
        
    def view_start_point(self):
        if self.start_point is not None:
            # Set camera to look at start point
            self.center_xyz = self.start_point.copy()
            
            # Calculate direction vector from start to second point
            if len(self.centerline) > 1:
                direction = self.centerline[1] - self.centerline[0]
                direction = direction / np.linalg.norm(direction)
                
                # Set camera position to look along the tube direction
                # Calculate azimuth and elevation from direction vector
                azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
                elevation = np.degrees(np.arcsin(direction[2]))
                
                # Position camera to face the direction of the tube
                self.camera_azimuth = azimuth - 180  # Look back at the start point
                self.camera_elevation = -elevation
                self.camera_distance = 200  # Closer view
            
            self.update()
            print("Viewing entry point")
    
    def view_middle_point(self):
        if self.middle_point is not None:
            # Set camera to look at middle point
            self.center_xyz = self.middle_point.copy()
            
            # Reset to a standard viewing angle
            self.camera_azimuth = -35
            self.camera_elevation = 20
            self.camera_distance = 1000
            
            self.update()
            print("Viewing middle point")
    
    def view_end_point(self):
        if self.end_point is not None:
            # Set camera to look at end point
            self.center_xyz = self.end_point.copy()
            
            # Calculate direction vector from second-to-last to last point
            if len(self.centerline) > 1:
                direction = self.centerline[-1] - self.centerline[-2]
                direction = direction / np.linalg.norm(direction)
                
                # Set camera position to look along the tube direction
                # Calculate azimuth and elevation from direction vector
                azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
                elevation = np.degrees(np.arcsin(direction[2]))
                
                # Position camera to face the direction of the tube
                self.camera_azimuth = azimuth  # Look along the tube direction
                self.camera_elevation = -elevation
                self.camera_distance = 200  # Closer view
            
            self.update()
            print("Viewing end point")

    def update(self):
        # Call the parent class's update method to properly refresh the widget
        super(TubeViewWidget, self).update()
        
    def get_distance_labels(self):
        """Generate distance labels based on the current zoom level"""
        if not hasattr(self, 'world_centerline') or self.world_centerline is None or len(self.world_centerline) < 2:
            return []
            
        # Determine interval based on zoom level
        if self.camera_distance < 100:
            interval = 10  # Show labels every 10m when very close
        elif self.camera_distance < 300:
            interval = 20  # Show labels every 20m when close
        elif self.camera_distance < 700:
            interval = 50  # Show labels every 50m when at medium distance
        else:
            interval = 100  # Show labels every 100m when far away
            
        # Calculate total distance along centerline
        total_distance = 0
        distances = [0]
        for i in range(1, len(self.world_centerline)):
            segment_length = np.linalg.norm(self.world_centerline[i] - self.world_centerline[i-1])
            total_distance += segment_length
            distances.append(total_distance)
            
        # Generate labels at specified interval
        labels = []
        current_distance = 0
        tube_radius = getattr(self, 'bore_radius', 56 * 0.0254 / 2) * self.render_scale
        
        while current_distance <= total_distance:
            # Find segment containing this distance
            for i in range(1, len(distances)):
                if distances[i-1] <= current_distance <= distances[i]:
                    # Interpolate position
                    t = (current_distance - distances[i-1]) / (distances[i] - distances[i-1])
                    world_pos = self.world_centerline[i-1] + t * (self.world_centerline[i] - self.world_centerline[i-1])
                    pos = self._to_render_coords(world_pos)
                    
                    # Add offset above centerline
                    offset = np.array([0, 0, tube_radius * 3])
                    label_pos = pos + offset
                    
                    labels.append((label_pos, current_distance))
                    break
            
            current_distance += interval
            
        return labels 

    def load_tube_data(self, centerline, radii, drill_centerline=None, contact_points=None, contact_positions=None, contact_forces=None, bore_radius=None, pipe_radius=None):
        self.contact_positions = contact_positions  # zaktualizowane 
        # Smooth the contact forces for better visualization
        self.contact_forces = np.array(contact_forces) if contact_forces is not None else np.zeros(len(centerline))
        if len(self.contact_forces) > 2:
            window = 5
            kernel = np.ones(window)/window
            self.smoothed_contact_forces = np.convolve(self.contact_forces, kernel, mode='same')
        else:
            self.smoothed_contact_forces = self.contact_forces

        # Build a local render transform so large UTM coordinates don't hurt depth precision.
        self.world_centerline = np.array(centerline, copy=True)
        self._compute_render_transform(self.world_centerline)
        render_centerline = self._to_render_coords(np.asarray(centerline))
        render_drill_centerline = self._to_render_coords(np.asarray(drill_centerline)) if drill_centerline is not None else None

        # Reset view only on the first data load; keep current camera on subsequent recalculations.
        is_first_load = self.centerline is None
        self.set_centerline(render_centerline, reset_view=is_first_load)

        self.radii = radii
        self.drill_centerline = drill_centerline
        self.contact_points = contact_points

        self.outer_tube_segments = []
        self.pipe_segments = []
        self.drill_tube_segments = []
        self.rings = []
        self.distance_labels = []
        self.segment_labels = []
        self.segment_radii = radii
        self.contact_points = contact_points  # n

        self.radius_scale_min = None
        self.radius_scale_max = None
        self.radius_average = None
        self.radius_outlier_threshold = None
        self.radius_quantile_thresholds = None
        self.radius_quantile_edges = None
        if self.segment_radii is not None and len(self.segment_radii) > 0:
            radii_array = np.asarray(self.segment_radii, dtype=float)
            valid_radii = radii_array[~np.isnan(radii_array)]
            if len(valid_radii) > 0:
                self.radius_average = float(np.mean(valid_radii))
                self.radius_outlier_threshold = 2.0 * self.radius_average
                non_outlier_radii = valid_radii[valid_radii <= self.radius_outlier_threshold]
                if len(non_outlier_radii) == 0:
                    non_outlier_radii = valid_radii

                self.radius_scale_min = float(np.min(non_outlier_radii))
                self.radius_scale_max = float(np.max(non_outlier_radii))
                quantiles = np.quantile(non_outlier_radii, [0.2, 0.4, 0.6, 0.8])
                self.radius_quantile_thresholds = tuple(float(q) for q in quantiles)
                self.radius_quantile_edges = (
                    float(np.min(non_outlier_radii)),
                    float(quantiles[0]),
                    float(quantiles[1]),
                    float(quantiles[2]),
                    float(quantiles[3]),
                    float(np.max(non_outlier_radii)),
                )

        if pipe_radius is None:
            pipe_radius = 0.6096
        self.pipe_radius = pipe_radius
        render_pipe_radius = pipe_radius * self.render_scale

        n_points_circle = 12 

        # Use dynamic bore_radius for outer tube
        if bore_radius is None:
            bore_radius = 56 * 0.0254 / 2

        self.bore_radius = bore_radius
        render_bore_radius = bore_radius * self.render_scale

        # Delete old display lists
        for attr in ['centerline_display_list', 'outer_tube_display_list',
                    'pipe_display_list', 'drill_tube_display_list', 'rings_display_list']:
            if hasattr(self, attr):
                glDeleteLists(getattr(self, attr), 1)

        # --- Tworzenie zewnętrznego otworu (Outer tube) ---
        tube1_radius = render_bore_radius
        n_points_circle = 12 
        
        for i in range(len(render_centerline) - 1):
            seg = render_centerline[i:i+2]
            vertices, faces, normals = TubeGenerator.compute_tube_geometry(seg, radius=tube1_radius, n_points=n_points_circle)
            # Use radius for segment (if available) to determine color
            if self.segment_radii is not None and len(self.segment_radii) > i and self.segment_radii[i] is not None:
                custom_thresholds = self._get_custom_radius_thresholds()
                active_thresholds = custom_thresholds if custom_thresholds is not None else self.radius_quantile_thresholds
                color = ColorUtils.get_gradient_color(
                    self.segment_radii[i],
                    self.radius_scale_min,
                    self.radius_scale_max,
                    self.radius_average,
                    active_thresholds,
                )
            else:
                color = (0.5, 0.5, 0.5, 0.5)  # Default gray if no radius
            self.outer_tube_segments.append((vertices, faces, normals, color))
        
        pipe_line = render_drill_centerline if render_drill_centerline is not None else render_centerline
        pipe_n_points_circle = n_points_circle

        for i in range(len(pipe_line) - 1):
            seg = pipe_line[i:i+2]
            vertices, faces, normals = TubeGenerator.compute_tube_geometry(seg, radius=render_pipe_radius, n_points=pipe_n_points_circle)
            
            # Gradient color based on smoothed contact force
            force = 11 * (self.smoothed_contact_forces[i] + self.smoothed_contact_forces[i+1])
            print(force)
            if force > 0:
                color = (1.0, 1.0 - force, 1.0 - force, 1.0)
                print(color)
            else:
                color = (1.0, 1.0, 1.0, 1.0)
            
            self.pipe_segments.append((vertices, faces, normals, color))

        # === Drill tube model (if any) ===
        if render_drill_centerline is not None and len(render_drill_centerline) > 1:
            tube2_radius = render_pipe_radius
            drill_n_points = max(16, min(32, int(150000 / len(render_drill_centerline))))
            print(f"Using {drill_n_points} points per circle for drill tube")

            for i in range(len(render_drill_centerline) - 1):
                seg = render_drill_centerline[i:i+2]
                vertices, faces, normals = TubeGenerator.compute_tube_geometry(seg, radius=tube2_radius, n_points=drill_n_points)
                self.drill_tube_segments.append((vertices, faces, normals, (0.8, 0, 0.8, 1.0)))  # magenta

        # === Rings for cross-section markers ===
        n_ring_points = 16  # Increased for smoother circles
        ring_radius = tube1_radius * 1.01
        def compute_rotation_minimizing_frames(centerline):
            n = len(centerline)
            tangents = np.zeros_like(centerline)
            tangents[1:-1] = centerline[2:] - centerline[:-2]
            tangents[0] = centerline[1] - centerline[0]
            tangents[-1] = centerline[-1] - centerline[-2]
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
            normals = np.zeros_like(centerline)
            binormals = np.zeros_like(centerline)
            up = np.array([0, 0, 1], dtype=float)
            if np.allclose(tangents[0], up):
                up = np.array([1, 0, 0], dtype=float)
            normals[0] = np.cross(tangents[0], up)
            if np.linalg.norm(normals[0]) < 1e-8:
                normals[0] = np.array([1, 0, 0], dtype=float)
            normals[0] /= np.linalg.norm(normals[0])
            binormals[0] = np.cross(tangents[0], normals[0])
            for i in range(1, n):
                v = tangents[i-1]
                w = tangents[i]
                if np.linalg.norm(v - w) < 1e-6:
                    normals[i] = normals[i-1]
                else:
                    axis = np.cross(v, w)
                    if np.linalg.norm(axis) < 1e-6:
                        normals[i] = normals[i-1]
                    else:
                        axis /= np.linalg.norm(axis)
                        angle = np.arccos(np.clip(np.dot(v, w), -1.0, 1.0))
                        # Rodrigues' rotation formula
                        K = np.array([[0, -axis[2], axis[1]],
                                      [axis[2], 0, -axis[0]],
                                      [-axis[1], axis[0], 0]])
                        R = (
                            np.eye(3) +
                            np.sin(angle) * K +
                            (1 - np.cos(angle)) * np.dot(K, K)
                        )
                        normals[i] = np.dot(R, normals[i-1])
                normals[i] /= np.linalg.norm(normals[i])
                binormals[i] = np.cross(tangents[i], normals[i])
            return normals, binormals

        tangents = np.zeros_like(render_centerline)
        tangents[1:-1] = render_centerline[2:] - render_centerline[:-2]
        tangents[0] = render_centerline[1] - render_centerline[0]
        tangents[-1] = render_centerline[-1] - render_centerline[-2]
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        normals, binormals = compute_rotation_minimizing_frames(render_centerline)
        theta = np.linspace(0, 2 * np.pi, n_ring_points, endpoint=True)

        for i in range(1, len(render_centerline)-1):
            center = render_centerline[i]
            n = normals[i]
            b = binormals[i]
            ring_points = center + ring_radius * (np.cos(theta)[:, None] * n + np.sin(theta)[:, None] * b)
            self.rings.append((ring_points, (0, 0, 0)))

        # === Segment numbers ===
        for i in range(len(render_centerline) - 1):
            midpoint = (render_centerline[i] + render_centerline[i+1]) / 2
            self.segment_labels.append((midpoint, i))

        # === Distance labels ===
        if len(render_centerline) > 1:
            total_distance = 0
            distances = [0]
            for i in range(1, len(render_centerline)):
                total_distance += np.linalg.norm(render_centerline[i] - render_centerline[i-1])
                distances.append(total_distance)

            interval = 100  # meters
            current_distance = 0
            while current_distance <= total_distance:
                for i in range(1, len(distances)):
                    if distances[i-1] <= current_distance <= distances[i]:
                        t = (current_distance - distances[i-1]) / (distances[i] - distances[i-1])
                        pos = render_centerline[i-1] + t * (render_centerline[i] - render_centerline[i-1])
                        offset = np.array([0, 0, tube1_radius * 3])
                        self.distance_labels.append((pos + offset, current_distance))
                        break
                current_distance += interval

        # === Generate display lists ===
        self.createDisplayLists()
        self.update()
        if drill_centerline is not None:
            print("deflected_centerline offset:", np.max(np.linalg.norm(np.array(drill_centerline) - np.array(centerline), axis=1)))
        else:
            print("drill_centerline is None")


        
    def createDisplayLists(self):
        # Create centerline display list
        self.centerline_display_list = glGenLists(1)
        glNewList(self.centerline_display_list, GL_COMPILE)
        self.drawCenterline()
        glEndList()
        
        # Create outer tube display list
        self.outer_tube_display_list = glGenLists(1)
        glNewList(self.outer_tube_display_list, GL_COMPILE)
        # Enable depth writing for outer tube
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        
        # Apply a small offset to prevent z-fighting
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        
        for i, segment in enumerate(self.outer_tube_segments):
            vertices, faces, normals, _ = segment

            # Use gradient color based on segment radius
            if self.segment_radii is not None and i < len(self.segment_radii) and self.segment_radii[i] is not None:
                custom_thresholds = self._get_custom_radius_thresholds()
                active_thresholds = custom_thresholds if custom_thresholds is not None else self.radius_quantile_thresholds
                color = ColorUtils.get_gradient_color(
                    self.segment_radii[i],
                    self.radius_scale_min,
                    self.radius_scale_max,
                    self.radius_average,
                    active_thresholds,
                )
            else:
                color = (0.5, 0.5, 0.5, 1.0)  # Default gray if no radius
            self.drawMesh(vertices, faces, normals, color)

            
        glDisable(GL_POLYGON_OFFSET_FILL)
        glEndList()

        # Create drill tube display list only if there are drill tube segments
        if self.drill_tube_segments:
            self.drill_tube_display_list = glGenLists(1)
            glNewList(self.drill_tube_display_list, GL_COMPILE)
            # Ensure depth testing is properly set for drill tube
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            
            # Apply a different offset for drill tube to prevent z-fighting with outer tube
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(2.0, 2.0)  # Larger offset than outer tube
            
            for segment in self.drill_tube_segments:
                vertices, faces, normals, color = segment
                self.drawMesh(vertices, faces, normals, color)
                
            glDisable(GL_POLYGON_OFFSET_FILL)
            glEndList()
        else:
            # Set to None to indicate no drill tube to render
            self.drill_tube_display_list = None
        

        if self.pipe_segments:
            self.pipe_display_list = glGenLists(1)
            glNewList(self.pipe_display_list, GL_COMPILE)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.5, 1.5)

            for vertices, faces, normals, color in self.pipe_segments:
                self.drawMesh(vertices, faces, normals, color)

            glDisable(GL_POLYGON_OFFSET_FILL)
            glEndList()
        else:
            self.pipe_display_list = None



        # Create rings display list
        self.rings_display_list = glGenLists(1)
        glNewList(self.rings_display_list, GL_COMPILE)
        
        # Enable depth testing and writing for proper occlusion
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        
        # Use polygon offset to prevent z-fighting with tubes
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(1.0, 1.0)
        
        for ring in self.rings:
            points, color = ring
            self.drawRing(points, color)
            
        glDisable(GL_POLYGON_OFFSET_LINE)
        glEndList() 

    def paintEvent(self, event):
        super(TubeViewWidget, self).paintEvent(event)
        
        # Draw FPS counter, distance labels, and segment numbers
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        
        # Draw FPS counter
        painter.setPen(Qt.white)
        painter.drawText(10, 20, f"FPS: {self.fps:.1f}")

        if self.show_outer_tube:
            self._draw_borehole_color_legend(painter)

        
        # Draw distance labels if enabled
        if self.show_distance_labels and hasattr(self, 'modelview_matrix') and hasattr(self, 'projection_matrix'):
            # Get dynamically generated labels based on current zoom
            dynamic_labels = self.get_distance_labels()
            
            for label_pos, distance in dynamic_labels:
                # Project 3D point to 2D screen coordinates
                viewport = glGetIntegerv(GL_VIEWPORT)
                win_x, win_y, win_z = gluProject(
                    label_pos[0], label_pos[1], label_pos[2],
                    self.modelview_matrix,
                    self.projection_matrix,
                    viewport
                )
                
                # Only draw if the point is in front of the camera (win_z < 1.0)
                if win_z < 1.0:
                    # Convert OpenGL y-coordinate (bottom-up) to Qt y-coordinate (top-down)
                    win_y = viewport[3] - win_y
                    
                    # Draw distance label with black outline for better visibility
                    text = f"{distance:.0f}m"
                    
                    # Draw text outline
                    painter.setPen(Qt.black)
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        painter.drawText(int(win_x) + dx, int(win_y) + dy, text)
                    
                    # Draw text
                    painter.setPen(Qt.white)
                    painter.drawText(int(win_x), int(win_y), text)
        
        # Draw segment numbers if enabled
        if self.show_segment_numbers and hasattr(self, 'modelview_matrix') and hasattr(self, 'projection_matrix'):
            for label_pos, segment_num in self.segment_labels:
                # Project 3D point to 2D screen coordinates
                viewport = glGetIntegerv(GL_VIEWPORT)
                win_x, win_y, win_z = gluProject(
                    label_pos[0], label_pos[1], label_pos[2],
                    self.modelview_matrix,
                    self.projection_matrix,
                    viewport
                )
                
                # Only draw if the point is in front of the camera (win_z < 1.0)
                if win_z < 1.0:
                    # Convert OpenGL y-coordinate (bottom-up) to Qt y-coordinate (top-down)
                    win_y = viewport[3] - win_y
                    
                    # Draw segment label with black outline for better visibility
                    text = f"{segment_num + 1}"
                    
                    # Draw text outline
                    painter.setPen(Qt.black)
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        painter.drawText(int(win_x) + dx, int(win_y) + dy, text)
                    
                    # Draw text
                    painter.setPen(Qt.yellow)  # Yellow for segment numbers
                    painter.drawText(int(win_x), int(win_y), text)
        
        # Draw labels for 5 segments with max contact forces (excluding first 20)
        if self.show_force_labels and hasattr(self, 'smoothed_contact_forces') and hasattr(self, 'segment_labels'):
            contact_forces = np.array(self.smoothed_contact_forces)
            indices = np.arange(min(len(contact_forces), len(self.segment_labels)))
            for idx in indices:
                force_val = contact_forces[idx]
                label_pos, segment_num = self.segment_labels[idx]
                # Project 3D point to 2D screen coordinates
                viewport = glGetIntegerv(GL_VIEWPORT)
                win_x, win_y, win_z = gluProject(
                    label_pos[0], label_pos[1], label_pos[2],
                    self.modelview_matrix,
                    self.projection_matrix,
                    viewport
                )
                if win_z < 1.0:
                    win_y = viewport[3] - win_y
                    display_force = force_val * 1000.0
                    if display_force < 1.0:
                        continue

                    text = f"{display_force:.0f}"
                    painter.setPen(Qt.black)
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        painter.drawText(int(win_x) + dx, int(win_y) + dy - 34, text)
                    painter.setPen(Qt.red)
                    painter.drawText(int(win_x), int(win_y) - 34, text)

        painter.end()

    def _draw_borehole_color_legend(self, painter: QPainter) -> None:
        """Draw curvature-radius color legend when Borehole visualization is enabled."""
        legend_width = 240
        legend_height = 154
        margin = 12
        x0 = self.width() - legend_width - margin
        y0 = margin

        painter.fillRect(x0, y0, legend_width, legend_height, QColor(0, 0, 0, 140))
        painter.setPen(Qt.white)
        painter.drawText(x0 + 10, y0 + 20, "Curvature Radius")

        custom_thresholds = self._get_custom_radius_thresholds()
        if custom_thresholds is not None:
            labels = [
                f"< {custom_thresholds[0]:.1f} m",
                f"{custom_thresholds[0]:.1f} - {custom_thresholds[1]:.1f} m",
                f"{custom_thresholds[1]:.1f} - {custom_thresholds[2]:.1f} m",
                f"{custom_thresholds[2]:.1f} - {custom_thresholds[3]:.1f} m",
                f">= {custom_thresholds[3]:.1f} m",
            ]
        elif self.radius_quantile_edges is not None:
            bounds = self.radius_quantile_edges
            labels = [
                f"< {bounds[1]:.1f} m",
                f"{bounds[1]:.1f} - {bounds[2]:.1f} m",
                f"{bounds[2]:.1f} - {bounds[3]:.1f} m",
                f"{bounds[3]:.1f} - {bounds[4]:.1f} m",
                f">= {bounds[4]:.1f} m",
            ]
        else:
            labels = [
                "< 600 m",
                "600 - 800 m",
                "800 - 1000 m",
                "1000 - 1600 m",
                ">= 1600 m",
            ]

        swatches = [
            ((1.0, 0.0, 0.0), labels[0]),
            ((1.0, 0.65, 0.0), labels[1]),
            ((1.0, 1.0, 0.0), labels[2]),
            ((0.0, 1.0, 0.0), labels[3]),
            ((0.0, 0.39, 0.0), labels[4]),
        ]

        row_y = y0 + 40
        for color_rgb, label in swatches:
            color = QColor(
                int(color_rgb[0] * 255),
                int(color_rgb[1] * 255),
                int(color_rgb[2] * 255),
                220,
            )
            painter.fillRect(x0 + 10, row_y - 10, 16, 12, color)
            painter.setPen(Qt.white)
            painter.drawText(x0 + 34, row_y, label)
            row_y += 24
