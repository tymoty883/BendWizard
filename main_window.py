"""
Main application window for the Pipe-Borehole Visualization App.
Handles GUI setup, user interaction, and data orchestration.
"""

from typing import Optional, Dict, List
import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar, QPushButton, QFileDialog,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QSplitter, QLabel, QLineEdit,
    QDialog, QDialogButtonBox, QFormLayout, QComboBox, QHBoxLayout, QToolButton,
    QMenu
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt

import constants
from tube_view_widget import TubeViewWidget
from data_loader import DataLoader, DataValidationError
from tube_utils import simulate_pipe_deflection


def analyze_pipe_stress(
    centerline: np.ndarray,
    outer_diameter_mm: float = constants.PIPE_OUTER_DIAMETER_MM,
    thickness_mm: float = constants.PIPE_WALL_THICKNESS_MM,
    E: float = constants.STEEL_YOUNGS_MODULUS_PA
) -> List[Dict]:
    """
    Calculate bending stress along pipe centerline for each segment.
    
    Uses moment of inertia formula for thin-walled cylindrical shells.
    Formula: σ = M × y / I = E × y / R
    where M = bending moment, I = moment of inertia, R = curvature radius
    
    Args:
        centerline: Nx3 array of centerline points
        outer_diameter_mm: Outer pipe diameter (default: DN1016 = 1016 mm)
        thickness_mm: Wall thickness (default: 20 mm)
        E: Young's modulus for steel (default: 206 GPa)
        
    Returns:
        List of dicts with keys: segment, radius_m, moment_Nm, stress_Pa, critical
        
    Raises:
        ValueError: If inputs are invalid
    """
    if centerline.ndim != 2 or centerline.shape[1] != 3 or len(centerline) < 3:
        raise ValueError("Centerline must be Nx3 array with N >= 3")
    
    D = outer_diameter_mm / 1000.0  # Convert to meters
    t = thickness_mm / 1000.0
    r_i = D / 2.0 - t  # Inner radius
    r_o = D / 2.0  # Outer radius

    # Moment of inertia for thin-walled cylindrical shell
    I = (np.pi / 64.0) * (r_o**4 - r_i**4)
    y = r_o  # Maximum distance from neutral axis (surface stress)

    results = []

    for i in range(1, len(centerline) - 1):
        p1 = centerline[i - 1]
        p2 = centerline[i]
        p3 = centerline[i + 1]

        # Compute segment lengths
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # Compute curvature radius from three points (circle through 3 points)
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        
        if area == 0:
            R = float('inf')
        else:
            R = (a * b * c) / (4.0 * area)

        # Calculate moment and stress
        if R == float('inf') or R == 0:
            M = 0.0
            sigma = 0.0
        else:
            M = E * I / R
            sigma = M * y / I

        results.append({
            'segment': i,
            'radius_m': R,
            'moment_Nm': M,
            'stress_Pa': sigma,
            'critical': sigma > constants.STRESS_CRITICAL_LIMIT_PA
        })

    return results


class ToolbarPopupPanel(QWidget):
    """Popup panel anchored to a toolbar button."""

    def __init__(self, title: str, button: QToolButton, parent=None):
        super().__init__(parent, Qt.Popup)
        self.button = button

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(8)

        title_label = QLabel(f"{title}:")
        root_layout.addWidget(title_label)

        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)
        root_layout.addLayout(self.content_layout)

    def addWidget(self, widget: QWidget) -> None:
        """Add a control to the popup panel."""
        self.content_layout.addWidget(widget)

    def show_below_button(self) -> None:
        """Show the popup directly below its owning toolbar button."""
        global_pos = self.button.mapToGlobal(self.button.rect().bottomLeft())
        self.move(global_pos)
        self.show()

    def hideEvent(self, event) -> None:
        """Reset the button state when the popup closes."""
        self.button.setChecked(False)
        super().hideEvent(event)


class ExcelImportDialog(QDialog):
    """Prompt for worksheet and column selections before importing Excel data."""

    FIELD_ORDER = ["Northing", "Easting", "Elev", "Radius"]

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.column_selectors: Dict[str, QComboBox] = {}

        self.setWindowTitle("Excel Import Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)
        file_label = QLabel(f"File: {os.path.basename(file_path)}")
        file_label.setWordWrap(True)
        layout.addWidget(file_label)

        form_layout = QFormLayout()
        self.sheet_combo = QComboBox()
        self.sheet_combo.currentTextChanged.connect(self._populate_column_selectors)
        form_layout.addRow("Worksheet:", self.sheet_combo)

        for field in self.FIELD_ORDER:
            combo = QComboBox()
            self.column_selectors[field] = combo
            form_layout.addRow(f"{field} column:", combo)

        layout.addLayout(form_layout)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.sheet_names = DataLoader.get_excel_sheet_names(file_path)
        if not self.sheet_names:
            raise DataValidationError("No worksheets found in the selected Excel file")

        self.sheet_combo.addItems(self.sheet_names)

        first_valid_sheet = None
        for sheet_name in self.sheet_names:
            columns = DataLoader.get_excel_columns(self.file_path, sheet_name)
            if columns:
                first_valid_sheet = sheet_name
                break

        if first_valid_sheet is not None:
            self.sheet_combo.setCurrentText(first_valid_sheet)
        else:
            self._populate_column_selectors(self.sheet_combo.currentText())

    def _populate_column_selectors(self, sheet_name: str) -> None:
        """Reload the available columns for the selected worksheet."""
        columns = DataLoader.get_excel_columns(self.file_path, sheet_name)
        has_columns = bool(columns)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(has_columns)

        if not has_columns:
            self.status_label.setText(
                f"Worksheet '{sheet_name}' does not contain readable columns. Select a different worksheet."
            )
            for selector in self.column_selectors.values():
                selector.blockSignals(True)
                selector.clear()
                selector.setEnabled(False)
                selector.blockSignals(False)
            return

        self.status_label.setText("")

        chosen_columns: List[str] = []
        for field in self.FIELD_ORDER:
            selector = self.column_selectors[field]
            selector.blockSignals(True)
            selector.clear()
            selector.addItems(columns)
            selector.setEnabled(True)
            selector.setCurrentText(self._suggest_column(field, columns, chosen_columns))
            selector.blockSignals(False)
            chosen_columns.append(selector.currentText())

    def _suggest_column(self, field_name: str, columns: List[str], chosen_columns: List[str]) -> str:
        """Choose a sensible default for a required field."""
        lowered_field = field_name.lower()
        exact_matches = [column for column in columns if str(column).strip().lower() == lowered_field]
        for column in exact_matches:
            if column not in chosen_columns:
                return column

        partial_matches = [column for column in columns if lowered_field in str(column).strip().lower()]
        for column in partial_matches:
            if column not in chosen_columns:
                return column

        for column in columns:
            if column not in chosen_columns:
                return column

        return columns[0]

    def get_selection(self) -> Dict[str, str]:
        """Return the selected worksheet and column mapping."""
        return {
            "sheet_name": self.sheet_combo.currentText(),
            "column_mapping": {
                field: selector.currentText()
                for field, selector in self.column_selectors.items()
            }
        }

    def accept(self) -> None:
        """Validate column selections before closing."""
        selection = self.get_selection()
        selected_columns = list(selection["column_mapping"].values())
        if len(set(selected_columns)) != len(selected_columns):
            QMessageBox.warning(self, "Invalid Selection", "Each field must use a different Excel column.")
            return

        super().accept()


class MainWindow(QMainWindow):
    """Main application window with UI orchestration and data coordination."""
    
    OUTER_TUBE_DIAMETERS = constants.OUTER_TUBE_DIAMETERS_INCHES
    
    def __init__(self):
        super().__init__()
        self.centerline: Optional[np.ndarray] = None
        self.pipe_stress_data: Optional[List[Dict]] = None
        self.current_centerline_data: Optional[Dict] = None
        self.iterations: int = constants.DEFAULT_ITERATIONS
        self.outer_tube_diameter_inch: int = constants.DEFAULT_OUTER_TUBE_DIAMETER_INCH
        self.tube_radius_m: float = 0.6096
        self.tube_radius_unit: str = "mm"
        self.display_popup: Optional[ToolbarPopupPanel] = None
        self.geometry_popup: Optional[ToolbarPopupPanel] = None

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
        self.borehole_data_table.setHorizontalHeaderLabels(['Segment', 'Curvature Radius (m)'])
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
        self.group_toolbar = QToolBar("View Control Groups")
        self.group_toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.group_toolbar)

        def create_menu_button(title: str) -> QToolButton:
            button = QToolButton(self)
            button.setText(title)
            button.setPopupMode(QToolButton.InstantPopup)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.group_toolbar.addWidget(button)
            return button

        def create_popup_button(title: str, handler) -> QToolButton:
            button = QToolButton(self)
            button.setText(title)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.clicked.connect(handler)
            self.group_toolbar.addWidget(button)
            return button

        file_button = create_menu_button("File")
        display_button = create_popup_button("Display", self.toggle_display_popup)
        geometry_button = create_popup_button("Geometry", self.toggle_geometry_popup)
        view_button = create_menu_button("View")

        file_group = None
        view_group = None
        self.display_popup = ToolbarPopupPanel("Display", display_button, self)
        self.geometry_popup = ToolbarPopupPanel("Geometry", geometry_button, self)
        
        # Add load data button
        file_menu = QMenu(self)
        file_menu.addAction("Load Data Files", self.load_data_files)
        file_button.setMenu(file_menu)

        # Add outer tube diameter selector
        self.diameter_label = QLabel("Outer Tube Diameter [in]:")
        self.geometry_popup.addWidget(self.diameter_label)
        self.diameter_combo = QComboBox()
        for d in self.OUTER_TUBE_DIAMETERS:
            self.diameter_combo.addItem(str(d))
        self.diameter_combo.setCurrentText(str(self.outer_tube_diameter_inch))
        self.diameter_combo.currentTextChanged.connect(self.on_outer_tube_diameter_changed)
        self.geometry_popup.addWidget(self.diameter_combo)
        
        # Add toggle button for outer tube visibility
        toggle_outer_btn = QPushButton("Toggle Outer Tube")
        toggle_outer_btn.setCheckable(True)
        toggle_outer_btn.setChecked(False)  # Initially visible
        toggle_outer_btn.clicked.connect(self.view.toggle_outer_tube)
        self.display_popup.addWidget(toggle_outer_btn)
        
        # Add toggle button for distance labels
        toggle_labels_btn = QPushButton("Toggle Distance Labels")
        toggle_labels_btn.setCheckable(True)
        toggle_labels_btn.setChecked(True)  # Initially visible
        toggle_labels_btn.clicked.connect(self.view.toggle_distance_labels)
        self.display_popup.addWidget(toggle_labels_btn)

        # Add toggle button for force labels
        toggle_force_btn = QPushButton("Toggle Contact Labels")
        toggle_force_btn.setCheckable(True)
        toggle_force_btn.setChecked(True)  # Initially visible
        toggle_force_btn.clicked.connect(self.view.toggle_force_labels)
        self.display_popup.addWidget(toggle_force_btn)
        
        # Add toggle button for segment numbers
        toggle_segment_btn = QPushButton("Toggle Segment Numbers")
        toggle_segment_btn.setCheckable(True)
        toggle_segment_btn.setChecked(False)  # Initially hidden
        toggle_segment_btn.clicked.connect(self.view.toggle_segment_numbers)
        self.display_popup.addWidget(toggle_segment_btn)

        # Add pipe diameter input with unit conversion
        tube_radius_label = QLabel("Pipe Diameter:")
        self.geometry_popup.addWidget(tube_radius_label)

        self.tube_radius_input = QLineEdit()
        self.tube_radius_input.setFixedWidth(80)
        self.tube_radius_input.setAlignment(Qt.AlignRight)
        self.tube_radius_input.setValidator(QDoubleValidator(0.0001, 100000.0, 4, self))
        self.tube_radius_input.returnPressed.connect(self.update_tube_radius_from_input)
        self.tube_radius_input.editingFinished.connect(self.update_tube_radius_from_input)
        self.geometry_popup.addWidget(self.tube_radius_input)

        self.tube_radius_unit_combo = QComboBox()
        self.tube_radius_unit_combo.addItems(["mm", "in"])
        self.tube_radius_unit_combo.setCurrentText(self.tube_radius_unit)
        self.tube_radius_unit_combo.currentTextChanged.connect(self.on_tube_radius_unit_changed)
        self.geometry_popup.addWidget(self.tube_radius_unit_combo)

        self._sync_tube_radius_input_text()
        
        # Add label for iterations
        iterations_label = QLabel("Iterations:")
        self.geometry_popup.addWidget(iterations_label)
        
        # Add line edit for direct iterations input
        self.iterations_input = QLineEdit(str(self.iterations))
        self.iterations_input.setFixedWidth(50)
        self.iterations_input.setAlignment(Qt.AlignRight)
        self.iterations_input.setValidator(QIntValidator(1, 1000, self))  # Allow 1-1000 iterations
        self.geometry_popup.addWidget(self.iterations_input)

        apply_iterations_btn = QPushButton("Apply")
        apply_iterations_btn.clicked.connect(self.update_iterations_from_input)
        self.geometry_popup.addWidget(apply_iterations_btn)

        view_menu = QMenu(self)
        view_menu.addAction("View Start", self.view.view_start_point)
        view_menu.addAction("View Middle", self.view.view_middle_point)
        view_menu.addAction("View End", self.view.view_end_point)
        view_button.setMenu(view_menu)

        self.setCentralWidget(central_widget)

    def _close_toolbar_popups(self, except_popup: Optional[ToolbarPopupPanel] = None) -> None:
        """Close any open toolbar popup except the specified one."""
        for popup in (self.display_popup, self.geometry_popup):
            if popup is not None and popup is not except_popup:
                popup.hide()

    def toggle_display_popup(self, checked: bool) -> None:
        """Open or close the display settings popup."""
        if checked and self.display_popup is not None:
            self._close_toolbar_popups(except_popup=self.display_popup)
            self.display_popup.show_below_button()
        elif self.display_popup is not None:
            self.display_popup.hide()

    def toggle_geometry_popup(self, checked: bool) -> None:
        """Open or close the geometry settings popup."""
        if checked and self.geometry_popup is not None:
            self._close_toolbar_popups(except_popup=self.geometry_popup)
            self.geometry_popup.show_below_button()
        elif self.geometry_popup is not None:
            self.geometry_popup.hide()

    def on_outer_tube_diameter_changed(self, value):
        """Handle changes to the outer tube diameter from the dropdown."""
        try:
            self.outer_tube_diameter_inch = int(value)
        except ValueError:
            self.outer_tube_diameter_inch = 56  # fallback
        self.update_pipe_deflection()

    def _format_radius_value(self, value: float) -> str:
        """Format numeric radius values for display in line edit."""
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _sync_tube_radius_input_text(self) -> None:
        """Update pipe diameter input text from internal radius value and selected unit."""
        diameter_m = self.tube_radius_m * 2.0
        if self.tube_radius_unit == "in":
            display_value = diameter_m / 0.0254
        else:
            display_value = diameter_m * 1000.0
        self.tube_radius_input.setText(self._format_radius_value(display_value))

    def on_tube_radius_unit_changed(self, unit: str) -> None:
        """Switch displayed unit for pipe diameter while preserving physical value."""
        if unit not in ("mm", "in"):
            return
        self.tube_radius_unit = unit
        self._sync_tube_radius_input_text()

    def update_tube_radius_from_input(self) -> None:
        """Validate and apply pipe diameter entered by the user."""
        text = self.tube_radius_input.text().strip()
        try:
            value = float(text)
            if value <= 0:
                raise ValueError

            # Convert user-entered diameter to internal radius in meters.
            if self.tube_radius_unit == "in":
                self.tube_radius_m = (value * 0.0254) / 2.0
            else:
                self.tube_radius_m = (value / 1000.0) / 2.0

            self._sync_tube_radius_input_text()
            self.update_pipe_deflection()
        except ValueError:
            self._sync_tube_radius_input_text()
            QMessageBox.warning(self, "Invalid Input", "Pipe diameter must be a positive number")
    
        
    def load_data_files(self) -> None:
        """Load Excel file with survey data and initialize visualization."""
        main_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Main Data File",
            "",
            "Excel Files (*.xlsx *.xlsm);;All Files (*)"
        )
        
        if not main_file:
            return  # User canceled
        
        try:
            import_dialog = ExcelImportDialog(main_file, self)
            if import_dialog.exec_() != QDialog.Accepted:
                return

            selection = import_dialog.get_selection()

            # Load and validate data
            centerline, radii = DataLoader.load_main_data(
                main_file,
                sheet_name=selection["sheet_name"],
                column_mapping=selection["column_mapping"],
            )
            
            # Update UI with data information
            self._update_tube_info(self.borehole_info_table, main_file, centerline, radii, "Borehole")
            self._display_borehole_radii(centerline, radii, self.borehole_data_table)

            # Store for recalculation on parameter changes
            self.current_centerline_data = {
                'centerline': centerline,
                'radii': radii,
                'main_file': main_file,
                'sheet_name': selection["sheet_name"],
                'column_mapping': selection["column_mapping"]
            }
            
            # Run simulation and visualization
            self.update_pipe_deflection()
            
        except DataValidationError as e:
            QMessageBox.critical(
                self,
                "Data Validation Error",
                f"Invalid data format:\n\n{str(e)}\n\nPlease check your Excel file."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Data",
                f"Failed to load file:\n\n{str(e)}"
            )
            
    def update_iterations_from_input(self) -> None:
        """Validate and apply iterations input from UI."""
        try:
            value = int(self.iterations_input.text())
            if constants.MIN_ITERATIONS <= value <= constants.MAX_ITERATIONS:
                self.iterations = value
                self.update_pipe_deflection()
            else:
                # Reset to last valid value if out of range
                self.iterations_input.setText(str(self.iterations))
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Iterations must be between {constants.MIN_ITERATIONS} and {constants.MAX_ITERATIONS}"
                )
        except ValueError:
            # Reset to last valid value if input is invalid
            self.iterations_input.setText(str(self.iterations))
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer")

    def update_pipe_deflection(self) -> None:
        """Recalculate pipe deflection with current parameters and update visualization."""
        if not self.current_centerline_data:
            return

        # Store current camera state to preserve user view
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
            tube_radius = self.tube_radius_m
            pipe_outer_diameter_mm = tube_radius * 2.0 * 1000.0

            deflected_centerline, contact_points, contact_positions, contact_forces = simulate_pipe_deflection(
                centerline, tube_radius=tube_radius, bore_radius=bore_radius, stiffness=0.1, iterations=self.iterations
            )

            # Update the view with new deflection data and contact forces
            self.view.load_tube_data(centerline, radii, deflected_centerline, contact_points, contact_positions, contact_forces,
                                    bore_radius=bore_radius, pipe_radius=tube_radius)

            # Update contact points visualization
            from tube_utils import compute_contact_angles

            # contact_positions is already indexed correctly by simulate_pipe_deflection
            # Extract indices from contact_points boolean array for consistency
            contact_indices = [i for i, val in enumerate(contact_points) if val]
            
            # Only compute angles if we have contact points
            if contact_indices and contact_positions:
                contact_results = compute_contact_angles(centerline, contact_indices, contact_positions)
            else:
                contact_results = []
            # Update the contact table
            self.contact_table.setRowCount(len(contact_results))
            for i, (distance, angle) in enumerate(contact_results):
                self.contact_table.setItem(i, 0, QTableWidgetItem(f"{distance:.2f}"))
                self.contact_table.setItem(i, 1, QTableWidgetItem(f"{angle:.2f}"))

            # Update the view
            self.view.update()

            # Run pipe stress analysis with the new deflection
            self.pipe_stress_data = analyze_pipe_stress(
                deflected_centerline,
                outer_diameter_mm=pipe_outer_diameter_mm,
            )
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

    def _update_tube_info(self, table: QTableWidget, file_path: str, centerline: np.ndarray, radii: np.ndarray, tube_name: str) -> None:
        """
        Update the info table with statistics and file information.
        
        Args:
            table: QTableWidget to update
            file_path: Path to source file
            centerline: Nx3 array of centerline points
            radii: N-array of bore radii
            tube_name: Name/label for this tube
        """
        table.setRowCount(7)
        
        # File info
        table.setItem(0, 0, QTableWidgetItem("Source File"))
        table.setItem(0, 1, QTableWidgetItem(file_path))
        
        # Segment info
        table.setItem(1, 0, QTableWidgetItem("Total Segments"))
        table.setItem(1, 1, QTableWidgetItem(str(len(centerline) - 1)))
        
        # Calculate and display radius statistics
        if radii is not None and len(radii) > 0:
            try:
                radii_array = np.asarray(radii)
                valid_radii = radii_array[~np.isnan(radii_array)]
                
                if len(valid_radii) > 0:
                    min_radius = np.min(valid_radii)
                    max_radius = np.max(valid_radii)
                    avg_radius = np.mean(valid_radii)
                    
                    table.setItem(2, 0, QTableWidgetItem("Min Curvature Radius"))
                    table.setItem(2, 1, QTableWidgetItem(f"{min_radius:.2f} m"))
                    table.setItem(3, 0, QTableWidgetItem("Max Curvature Radius"))
                    table.setItem(3, 1, QTableWidgetItem(f"{max_radius:.2f} m"))
                    table.setItem(4, 0, QTableWidgetItem("Avg CurvatureRadius"))
                    table.setItem(4, 1, QTableWidgetItem(f"{avg_radius:.2f} m"))
                    
                else:
                    self._set_table_error(table, 2, 5, "No valid radius data")
                    
            except Exception as e:
                print(f"Error processing radii: {e}")
                self._set_table_error(table, 2, 5, f"Error: {str(e)}")
        else:
            self._set_table_error(table, 2, 5, "No radius data available")
    
    def _set_table_error(self, table: QTableWidget, start_row: int, end_row: int, message: str) -> None:
        """Helper method to display error message in table rows."""
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