"""
Application-wide constants and configuration parameters.
Extracted from magic numbers to improve maintainability and flexibility.
"""

# ============================================================================
# PIPE MATERIAL PROPERTIES (DN1016 Pipe Specifications)
# ============================================================================

# Outer diameter of the pipe (DN1016) in millimeters
PIPE_OUTER_DIAMETER_MM = 609.6

# Wall thickness in millimeters
PIPE_WALL_THICKNESS_MM = 20.0

# Converted to meters for internal calculations
PIPE_OUTER_DIAMETER_M = PIPE_OUTER_DIAMETER_MM / 1000.0
PIPE_RADIUS_M = PIPE_OUTER_DIAMETER_M / 2.0  # ~0.508 m

# Young's modulus for steel (Pa)
STEEL_YOUNGS_MODULUS_PA = 206e9

# Yield stress limit for steel (Pa) - critical threshold
STRESS_CRITICAL_LIMIT_PA = 415e6


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Deflection simulation stiffness parameter (0.0 = rigid, 1.0 = fully flexible)
DEFLECTION_STIFFNESS = 0.1

# Enable the physics-based solver by default; set False to use legacy behavior.
USE_PHYSICS_DEFLECTION_SOLVER = True

# Physics solver defaults
PHYSICS_CONTACT_STIFFNESS = 25.0
PHYSICS_FRICTION_COEFFICIENT = 0.2
PHYSICS_STEP_SIZE = 0.02
PHYSICS_CONVERGENCE_TOLERANCE = 1e-4
PHYSICS_GRAVITY_WEIGHT = 0.0
PHYSICS_AXIAL_FORCE_WEIGHT = 0.0

# Default number of iterations for deflection simulation
DEFAULT_ITERATIONS = 60

# Minimum and maximum iterations allowed by user
MIN_ITERATIONS = 1
MAX_ITERATIONS = 1000

# Force scaling for visualization (multiplier)
CONTACT_FORCE_VISUALIZATION_SCALE = 10.0


# ============================================================================
# MESH GENERATION PARAMETERS
# ============================================================================

# Default number of points for circle tessellation in tube mesh
DEFAULT_CIRCLE_TESSELLATION_POINTS = 12

# Minimum and maximum tessellation points
MIN_CIRCLE_POINTS = 8
MAX_CIRCLE_POINTS = 32

# Ring tessellation points for cross-section markers
RING_TESSELLATION_POINTS = 16

# Ring radius multiplier (1.01 = slightly larger than tube radius)
RING_RADIUS_MULTIPLIER = 1.01


# ============================================================================
# UI/VISUALIZATION PARAMETERS
# ============================================================================

# Available borehole diameters (in inches)
BOREHOLE_DIAMETERS_INCHES = [i for i in range(26, 64, 2)]

# Default borehole diameter (in inches)
DEFAULT_BOREHOLE_DIAMETER_INCH = 56

# Backward-compatible aliases used by existing UI code.
OUTER_TUBE_DIAMETERS_INCHES = BOREHOLE_DIAMETERS_INCHES
DEFAULT_OUTER_TUBE_DIAMETER_INCH = DEFAULT_BOREHOLE_DIAMETER_INCH

# Window initial size (pixels)
WINDOW_DEFAULT_WIDTH = 1000
WINDOW_DEFAULT_HEIGHT = 800

# Splitter initial size ratio (3D view : data panel)
SPLITTER_3D_VIEW_RATIO = 0.8
SPLITTER_DATA_PANEL_RATIO = 0.2

# OpenGL rendering settings
GL_CLEAR_COLOR = (0.5, 0.8, 1.0, 1.0)  # Sky blue background

# Camera settings (spherical coordinates)
DEFAULT_CAMERA_DISTANCE = 1000
DEFAULT_CAMERA_AZIMUTH = -35
DEFAULT_CAMERA_ELEVATION = 20
CAMERA_ELEVATION_MIN = -89
CAMERA_ELEVATION_MAX = 89

# Camera pan speed multiplier
CAMERA_PAN_SPEED_MULTIPLIER = 1.0
CAMERA_PAN_SPEED_MINIMUM = 0.5

# Zoom parameters
CAMERA_ZOOM_FACTOR = 1.1
CAMERA_ZOOM_MINIMUM_STEP = 10.0
CAMERA_DISTANCE_MIN = 0.1
CAMERA_DISTANCE_MAX = 10000

# Mouse sensitivity
MOUSE_ROTATION_SENSITIVITY = 0.5

# Distance label parameters (in meters)
DISTANCE_LABEL_INTERVAL_VERY_CLOSE = 10
DISTANCE_LABEL_INTERVAL_CLOSE = 20
DISTANCE_LABEL_INTERVAL_MEDIUM = 50
DISTANCE_LABEL_INTERVAL_FAR = 100

DISTANCE_LABEL_ZOOM_THRESHOLD_VERY_CLOSE = 100
DISTANCE_LABEL_ZOOM_THRESHOLD_CLOSE = 300
DISTANCE_LABEL_ZOOM_THRESHOLD_MEDIUM = 700


# ============================================================================
# RENDERING SETTINGS
# ============================================================================

# OpenGL polygon offset for Z-fighting prevention
GL_POLYGON_OFFSET_BOREHOLE = (1.0, 1.0)
GL_POLYGON_OFFSET_PIPE = (2.5, 2.5)
GL_POLYGON_OFFSET_RINGS = (1.0, 1.0)

# Grid settings
GRID_STEP_SIZE = 200  # Meters
GRID_LINE_WIDTH = 1.0
GRID_LINE_COLOR = (1.0, 1.0, 1.0)  # White

# Centerline settings
CENTERLINE_LINE_WIDTH = 2.0
CENTERLINE_COLOR = (1.0, 1.0, 0.0)  # Yellow

# Ring (cross-section) settings
RING_LINE_WIDTH = 2.0
RING_LINE_COLOR = (0.0, 0.0, 0.0)  # Black

# Text overlay settings
FPS_TEXT_POSITION = (10, 20)
FPS_TEXT_COLOR = (255, 255, 255)  # White in Qt
TEXT_OUTLINE_COLOR = (0, 0, 0)  # Black in Qt
TEXT_OUTLINE_OFFSET = 1  # Pixels


# ============================================================================
# COLOR MAPPING FOR STRESS VISUALIZATION
# ============================================================================

# Stress color gradient thresholds (in meters for radius mapping)
STRESS_COLOR_THRESHOLD_VERY_LOW = 600  # Red
STRESS_COLOR_THRESHOLD_LOW = 800  # Orange
STRESS_COLOR_THRESHOLD_MEDIUM = 1000  # Yellow
STRESS_COLOR_THRESHOLD_HIGH = 1600  # Green

# RGBA color values for stress visualization
STRESS_COLOR_VERY_LOW = (1, 0, 0, 0.5)  # Red
STRESS_COLOR_LOW = (1, 0.65, 0, 0.5)  # Orange
STRESS_COLOR_MEDIUM = (1, 1, 0, 0.5)  # Yellow
STRESS_COLOR_HIGH = (0, 1, 0, 0.5)  # Light Green
STRESS_COLOR_VERY_HIGH = (0, 0.39, 0, 0.5)  # Dark Green

# Contact force coloring (white → red gradient)
CONTACT_FORCE_BASE_COLOR = (1.0, 1.0, 1.0, 1.0)  # White (no contact)
CONTACT_FORCE_MAX_COLOR = (1.0, 0.0, 0.0, 1.0)  # Red (high contact)


# ============================================================================
# FILE I/O SETTINGS
# ============================================================================

# Excel sheet name for survey data
EXCEL_SHEET_NAME = "Sheet1"

# Required columns in Excel file (for main_data load)
REQUIRED_COLUMNS_MAIN_DATA = ["Easting", "Northing", "Elev", "Radius"]

# Required columns for project data (optional Radius column)
REQUIRED_COLUMNS_PROJECT_DATA = ["Easting", "Northing", "Elev"]

# Maximum file size to load (in MB) - prevent DoS on very large files
MAX_FILE_SIZE_MB = 500

# Minimum number of survey points required
MIN_SURVEY_POINTS = 3

# Maximum number of survey points allowed
MAX_SURVEY_POINTS = 100000


# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_FILE_NOT_FOUND = "File not found: {}"
ERROR_INVALID_EXCEL_FORMAT = "Invalid Excel file format. Please ensure the file is a valid .xlsx or .xlsm file."
ERROR_MISSING_COLUMNS = "Missing required columns: {}. Expected columns: {}"
ERROR_INVALID_DATA_TYPE = "Invalid data type in column '{}'. Expected numeric values."
ERROR_INSUFFICIENT_POINTS = f"Insufficient survey points. Minimum {MIN_SURVEY_POINTS} points required, got {{}}."
ERROR_EXCESSIVE_POINTS = f"Too many survey points. Maximum {MAX_SURVEY_POINTS} allowed, got {{}}."
ERROR_COORDINATE_OUT_OF_BOUNDS = "Survey coordinate out of plausible bounds (consider checking input data)."
ERROR_EMPTY_DATASET = "No valid data points after filtering."
ERROR_DEFLECTION_SIMULATION = "Error during pipe deflection simulation: {}"
ERROR_STRESS_ANALYSIS = "Error during stress analysis: {}"
ERROR_MESH_GENERATION = "Error during mesh generation: {}"

# ============================================================================
# LOGGING
# ============================================================================

# Logging levels and formats (for future structured logging implementation)
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"
