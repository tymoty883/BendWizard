"""
Main application window for the Pipe-Borehole Visualization App.
Handles GUI setup, user interaction, and data orchestration.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar, QPushButton, QFileDialog,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QSplitter, QLabel, QLineEdit
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt

from tube_view_widget import TubeViewWidget
from data_loader import DataLoader
from tube_utils import simulate_pipe_deflection


def analyze_pipe_stress(centerline, outer_diameter_mm=1016, thickness_mm=20, E=206e9):
    """
    Analiza naprężeń wzdłuż rury DN1016x20 w oparciu o tor otworu.
    Zwraca listę słowników z parametrami dla każdego segmentu.
    """
    D = outer_diameter_mm / 1000  # m
    t = thickness_mm / 1000        # m
    r_i = D/2 - t
    r_o = D/2

    # Moment bezwładności powłoki
    I = (np.pi / 64) * (r_o**4 - r_i**4)
    y = r_o  # maksymalna odległość od osi (dla σ = M*y/I)

    results = []

    for i in range(1, len(centerline)-1):
        p1 = centerline[i-1]
        p2 = centerline[i]
        p3 = centerline[i+1]

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # Obliczenie krzywizny z trzech punktów (promień okręgu przez 3 pkt)
        s = (a + b + c) / 2
        area = np.sqrt(max(s*(s - a)*(s - b)*(s - c), 0))
        if area == 0:
            R = float('inf')
        else:
            R = (a * b * c) / (4.0 * area)

        if R == float('inf') or R == 0:
            M = 0
            sigma = 0
        else:
            M = E * I / R
            sigma = M * y / I  # czyli E * y / R

        results.append({
            'segment': i,
            'radius_m': R,
            'moment_Nm': M,
            'stress_Pa': sigma,
            'critical': sigma > 415e6
        })

    return results


class MainWindow(QMainWindow):
    OUTER_TUBE_DIAMETERS = [i for i in range(42, 66, 2)]  # 42, 44, ..., 64
    
    def __init__(self):
        super().__init__()
        self.centerline = None
        self.pipe_stress_data = None
        self.current_centerline_data = None  # Store the current centerline data for recalculation
        self.iterations = 60 # Default number of iterations
        self.outer_tube_diameter_inch = 56  # Default value, can be changed by user

        self.setWindowTitle('Bend Wizard')
        self.resize(1000, 800)
        
        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create a horizontal splitter for the main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Create the 3D view
        self.view = TubeViewWidget()
        splitter.addWidget(self.view)
        
        # Create the tab widget for tube data
        tab_widget = QTabWidget()
        
        # Borehole data tab
        borehole_tab = QWidget()
        borehole_layout = QVBoxLayout(borehole_tab)
        borehole_layout.setContentsMargins(5, 5, 5, 5)
        
        # Borehole info table
        self.borehole_info_table = QTableWidget()
        self.borehole_info_table.setColumnCount(2)
        self.borehole_info_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.borehole_info_table.horizontalHeader().setStretchLastSection(True)
        self.borehole_info_table.setMaximumHeight(200)  # Set maximum height
        self.borehole_info_table.setColumnWidth(0, 120)  # Set first column width
        self.borehole_info_table.setColumnWidth(1, 120)  # Set second column width
        borehole_layout.addWidget(self.borehole_info_table, 1)  # Add stretch factor
        
        # Borehole data table
        self.borehole_data_table = QTableWidget()
        self.borehole_data_table.setColumnCount(2)
        self.borehole_data_table.setHorizontalHeaderLabels(['Segment', 'Radius (m)'])
        self.borehole_data_table.horizontalHeader().setStretchLastSection(True)
        self.borehole_data_table.setMaximumHeight(600)  # Set maximum height
        self.borehole_data_table.setColumnWidth(0, 80)   # Segment column

        borehole_layout.addWidget(self.borehole_data_table, 2)  # Add stretch factor
        
        # Add tabs to tab widget
        tab_widget.addTab(borehole_tab, "Borehole Data")
        
        # Add the tab widget to the splitter
        splitter.addWidget(tab_widget)

        # tab styku
        self.contact_table = QTableWidget()
        self.contact_table.setColumnCount(2)
        self.contact_table.setHorizontalHeaderLabels(['Distance [m]', 'Angle [deg]'])
        self.contact_table.horizontalHeader().setStretchLastSection(True)
        tab_widget.addTab(self.contact_table, "Contact points")

        
        # Set initial sizes for the splitter (give more space to the 3D view)
        splitter.setSizes([int(self.width() * 0.8), int(self.width() * 0.2)])
        
        # Add the splitter to the main layout
        main_layout.addWidget(splitter)
        
        # Create navigation toolbar
        toolbar = QToolBar("View Controls")
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        
        # Add load data button
        load_data_btn = QPushButton("Load Data Files")
        load_data_btn.clicked.connect(self.load_data_files)
        toolbar.addWidget(load_data_btn)

        # Add outer tube diameter selector
        toolbar.addSeparator()
        from PyQt5.QtWidgets import QComboBox, QLabel
        self.diameter_label = QLabel("Outer Tube Diameter [in]:")
        toolbar.addWidget(self.diameter_label)
        self.diameter_combo = QComboBox()
        for d in self.OUTER_TUBE_DIAMETERS:
            self.diameter_combo.addItem(str(d))
        self.diameter_combo.setCurrentText(str(self.outer_tube_diameter_inch))
        self.diameter_combo.currentTextChanged.connect(self.on_outer_tube_diameter_changed)
        toolbar.addWidget(self.diameter_combo)

        # Add a separator in the toolbar
        toolbar.addSeparator()
        
        """# Add toggle button for drill tube visibility
        toggle_drill_btn = QPushButton("Toggle 1600m Radius Model")
        toggle_drill_btn.setCheckable(True)
        toggle_drill_btn.setChecked(True)  # Initially visible
        toggle_drill_btn.clicked.connect(self.view.toggle_drill_tube)
        toolbar.addWidget(toggle_drill_btn)"""
        
        # Add toggle button for outer tube visibility
        toggle_outer_btn = QPushButton("Toggle Outer Tube")
        toggle_outer_btn.setCheckable(True)
        toggle_outer_btn.setChecked(False)  # Initially visible
        toggle_outer_btn.clicked.connect(self.view.toggle_outer_tube)
        toolbar.addWidget(toggle_outer_btn)
        
        # Add toggle button for distance labels
        toggle_labels_btn = QPushButton("Toggle Distance Labels")
        toggle_labels_btn.setCheckable(True)
        toggle_labels_btn.setChecked(True)  # Initially visible
        toggle_labels_btn.clicked.connect(self.view.toggle_distance_labels)
        toolbar.addWidget(toggle_labels_btn)

        # Add toggle button for force labels
        toggle_force_btn = QPushButton("Toggle Contact Labels")
        toggle_force_btn.setCheckable(True)
        toggle_force_btn.setChecked(True)  # Initially visible
        toggle_force_btn.clicked.connect(self.view.toggle_force_labels)
        toolbar.addWidget(toggle_force_btn)
        
        # Add toggle button for segment numbers
        toggle_segment_btn = QPushButton("Toggle Segment Numbers")
        toggle_segment_btn.setCheckable(True)
        toggle_segment_btn.setChecked(False)  # Initially hidden
        toggle_segment_btn.clicked.connect(self.view.toggle_segment_numbers)
        toolbar.addWidget(toggle_segment_btn)
        
        # Add iterations input
        toolbar.addSeparator()
        
        # Add label for iterations
        iterations_label = QLabel("Iterations:")
        toolbar.addWidget(iterations_label)
        
        # Add line edit for direct iterations input
        self.iterations_input = QLineEdit(str(self.iterations))
        self.iterations_input.setFixedWidth(50)
        self.iterations_input.setAlignment(Qt.AlignRight)
        self.iterations_input.setValidator(QIntValidator(1, 1000, self))  # Allow 1-1000 iterations
        self.iterations_input.returnPressed.connect(self.update_iterations_from_input)
        self.iterations_input.editingFinished.connect(self.update_iterations_from_input)
        toolbar.addWidget(self.iterations_input)
        
        # Add view preset buttons
        toolbar.addSeparator()

        start_view_btn = QPushButton("View Start")
        start_view_btn.clicked.connect(self.view.view_start_point)
        toolbar.addWidget(start_view_btn)

        middle_view_btn = QPushButton("View Middle")
        middle_view_btn.clicked.connect(self.view.view_middle_point)
        toolbar.addWidget(middle_view_btn)

        end_view_btn = QPushButton("View End")
        end_view_btn.clicked.connect(self.view.view_end_point)
        toolbar.addWidget(end_view_btn)

        self.setCentralWidget(central_widget)

    def on_outer_tube_diameter_changed(self, value):
        """Handle changes to the outer tube diameter from the dropdown."""
        try:
            self.outer_tube_diameter_inch = int(value)
        except ValueError:
            self.outer_tube_diameter_inch = 56  # fallback
        self.update_pipe_deflection()
    
        
    def load_data_files(self):
        # Ask user to select main data file
        main_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Main Data File",
            "",
            "Excel Files (*.xlsx *.xlsm);;All Files (*)"
        )
        
        if not main_file:
            return  # User canceled
        
        # Ask user to select drill path file
        """drill_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Drill Path File",
            "",
            "Excel Files (*.xlsx *.xlsm);;All Files (*)"
        )
        
        if not drill_file:
            return  # User canceled"""
        
        try:
            centerline, radii = DataLoader.load_main_data(main_file)
            
            # Analiza i krzywizny
            self._update_tube_info(self.borehole_info_table, main_file, centerline, radii, "Borehole")
            self._display_borehole_radii(centerline, radii, self.borehole_data_table)

            # Store centerline data for recalculation
            self.current_centerline_data = {
                'centerline': centerline,
                'radii': radii,
                'main_file': main_file
            }
            
            # Calculate initial pipe deflection
            self.update_pipe_deflection()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            print(f"Error loading data: {e}")
            
    def update_iterations_from_input(self):
        """Handle iterations input changes"""
        try:
            value = int(self.iterations_input.text())
            if 1 <= value <= 1000:  # Validate range
                self.iterations = value
                self.update_pipe_deflection()
            else:
                # Reset to last valid value if out of range
                self.iterations_input.setText(str(self.iterations))
        except ValueError:
            # Reset to last valid value if input is invalid
            self.iterations_input.setText(str(self.iterations))

    def update_pipe_deflection(self):
        """Recalculate pipe deflection with current parameters"""
        if not self.current_centerline_data:
            return

        # Store current view parameters
        camera_pos = None
        if hasattr(self.view, 'camera_distance') and hasattr(self.view, 'camera_azimuth') and hasattr(self.view, 'camera_elevation'):
            camera_pos = {
                'distance': self.view.camera_distance,
                'azimuth': self.view.camera_azimuth,
                'elevation': self.view.camera_elevation
            }

        try:
            centerline = self.current_centerline_data['centerline']
            radii = self.current_centerline_data['radii']

            # Use selected outer tube diameter (inches) converted to meters radius
            bore_diameter_inch = self.outer_tube_diameter_inch
            bore_radius = bore_diameter_inch * 0.0254 / 2
            tube_radius = 40 * 0.0254 / 2  # Still hardcoded, can be made dynamic

            deflected_centerline, contact_points, contact_positions, contact_forces = simulate_pipe_deflection(
                centerline, tube_radius=tube_radius, bore_radius=bore_radius, stiffness=0.1, iterations=self.iterations
            )

            # Update the view with new deflection data and contact forces
            self.view.load_tube_data(centerline, radii, deflected_centerline, contact_points, contact_positions, contact_forces,
                                    bore_radius=bore_radius)

            # Update contact points visualization
            contact_indices = [i for i, val in enumerate(contact_points) if val]
            from tube_utils import compute_contact_angles
            contact_results = compute_contact_angles(centerline, contact_indices, contact_positions)

            # Update the contact table
            self.contact_table.setRowCount(len(contact_results))
            for i, (distance, angle) in enumerate(contact_results):
                self.contact_table.setItem(i, 0, QTableWidgetItem(f"{distance:.2f}"))
                self.contact_table.setItem(i, 1, QTableWidgetItem(f"{angle:.2f}"))

            # Update the view
            self.view.update()

            # Run pipe stress analysis with the new deflection
            self.pipe_stress_data = analyze_pipe_stress(deflected_centerline)
            self.view.set_pipe_stress_data(self.pipe_stress_data)
            for d in self.pipe_stress_data:
                print(d)

            # Restore camera position if it was stored
            if camera_pos:
                self.view.camera_distance = camera_pos['distance']
                self.view.camera_azimuth = camera_pos['azimuth']
                self.view.camera_elevation = camera_pos['elevation']
                self.view.update()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update pipe deflection: {str(e)}")
            print(f"Error updating pipe deflection: {e}")

    def _update_tube_info(self, table, file_path, centerline, radii, tube_name):
        """Update the info table for a tube with statistics and file info"""
        table.setRowCount(7)

        # File info
        table.setItem(0, 0, QTableWidgetItem("Source File"))
        table.setItem(0, 1, QTableWidgetItem(file_path))

        # Segment info
        table.setItem(1, 0, QTableWidgetItem("Total Segments"))
        table.setItem(1, 1, QTableWidgetItem(str(len(centerline)-1)))

        # Calculate statistics
        if radii is not None and len(radii) > 0:
            try:
                # Convert to numpy array if it isn't already
                radii_array = np.asarray(radii)

                # Filter out None and invalid values
                valid_radii = radii_array[~np.isnan(radii_array)]

                if len(valid_radii) > 0:
                    min_radius = np.min(valid_radii)
                    max_radius = np.max(valid_radii)
                    avg_radius = np.mean(valid_radii)

                    table.setItem(2, 0, QTableWidgetItem("Min Radius"))
                    table.setItem(2, 1, QTableWidgetItem(f"{min_radius:.2f} m"))
                    table.setItem(3, 0, QTableWidgetItem("Max Radius"))
                    table.setItem(3, 1, QTableWidgetItem(f"{max_radius:.2f} m"))
                table.setItem(4, 0, QTableWidgetItem("Avg Radius"))
                table.setItem(4, 1, QTableWidgetItem(f"{avg_radius:.2f} m"))

                # Calculate and display histogram
                hist, _ = np.histogram(valid_radii, bins=[0,800,900,1000,2000])
                table.setItem(5, 0, QTableWidgetItem("Radius Histogram"))
                table.setItem(5, 1, QTableWidgetItem(str(hist.tolist())))
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update pipe deflection: {str(e)}")
                print(f"Error updating pipe deflection: {e}")
        
    def _update_tube_info(self, table, file_path, centerline, radii, tube_name):
        """Update the info table for a tube with statistics and file info"""
        table.setRowCount(7)
        
        # File info
        table.setItem(0, 0, QTableWidgetItem("Source File"))
        table.setItem(0, 1, QTableWidgetItem(file_path))
        
        # Segment info
        table.setItem(1, 0, QTableWidgetItem("Total Segments"))
        table.setItem(1, 1, QTableWidgetItem(str(len(centerline)-1)))
        
        # Calculate statistics
        if radii is not None and len(radii) > 0:
            try:
                # Convert to numpy array if it isn't already
                radii_array = np.asarray(radii)
                
                # Filter out None and invalid values
                valid_radii = radii_array[~np.isnan(radii_array)]
                
                if len(valid_radii) > 0:
                    min_radius = np.min(valid_radii)
                    max_radius = np.max(valid_radii)
                    avg_radius = np.mean(valid_radii)
                    
                    table.setItem(2, 0, QTableWidgetItem("Min Radius"))
                    table.setItem(2, 1, QTableWidgetItem(f"{min_radius:.2f} m"))
                    table.setItem(3, 0, QTableWidgetItem("Max Radius"))
                    table.setItem(3, 1, QTableWidgetItem(f"{max_radius:.2f} m"))
                    table.setItem(4, 0, QTableWidgetItem("Avg Radius"))
                    table.setItem(4, 1, QTableWidgetItem(f"{avg_radius:.2f} m"))
                    
                    # Calculate and display histogram
                    hist, _ = np.histogram(valid_radii, bins=[0,800,900,1000,2000])
                    table.setItem(5, 0, QTableWidgetItem("Radius Histogram"))
                    table.setItem(5, 1, QTableWidgetItem(str(hist.tolist())))
                else:
                    self._set_table_error(table, 2, 5, "No valid radius data")
                    
            except Exception as e:
                print(f"Error processing radii: {e}")
                self._set_table_error(table, 2, 5, f"Error: {str(e)}")
        else:
            self._set_table_error(table, 2, 5, "No radius data available")
        
    
    def _set_table_error(self, table, start_row, end_row, message):
        """Helper method to set error message in table rows"""
        for row in range(start_row, end_row + 1):
            table.setItem(row, 0, QTableWidgetItem(""))
            table.setItem(row, 1, QTableWidgetItem(""))
        table.setItem(start_row, 0, QTableWidgetItem("Error"))
        table.setItem(start_row, 1, QTableWidgetItem(message))
    
    def _display_borehole_radii(self, centerline, radii, table):
        """Display only segment number and input radius in the borehole table."""
        n_segments = len(centerline) - 1
        table.setRowCount(n_segments)
        for i in range(n_segments):
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            if radii is not None and len(radii) > i and radii[i] is not None:
                table.setItem(i, 1, QTableWidgetItem(f"{radii[i]:.2f}"))
            else:
                table.setItem(i, 1, QTableWidgetItem("-"))