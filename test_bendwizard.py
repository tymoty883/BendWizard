"""
Unit tests for BendWizard core modules.

Tests focus on:
- Data loading and validation (data_loader.py)
- Geometry and numerical computations (tube_utils.py)
- Mesh generation (tube_generator.py)
- Color mapping (color_utils.py)
"""

import unittest
import numpy as np
import tempfile
import os
import pandas as pd
from pathlib import Path

# Import modules to test
import constants
from data_loader import DataLoader, DataValidationError
from tube_utils import compute_tnb_frame, simulate_pipe_deflection, compute_contact_angles
from tube_generator import TubeGenerator
from color_utils import ColorUtils


class TestDataLoader(unittest.TestCase):
    """Test data loading and validation."""
    
    def setUp(self):
        """Create temporary Excel file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_file = os.path.join(self.temp_dir, "valid_data.xlsx")
        self.create_valid_excel_file()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def create_valid_excel_file(self):
        """Create a valid test Excel file with survey data."""
        df = pd.DataFrame({
            "Easting": [100000.0, 100100.0, 100200.0, 100300.0, 100400.0],
            "Northing": [200000.0, 200100.0, 200200.0, 200300.0, 200400.0],
            "Elev": [50.0, 52.0, 55.0, 53.0, 51.0],
            "Radius": [800.0, 850.0, 900.0, 880.0, 820.0]
        })
        df.to_excel(self.valid_file, sheet_name="Sheet1", index=False)
    
    def test_load_valid_main_data(self):
        """Test loading valid data with radii."""
        centerline, radii = DataLoader.load_main_data(self.valid_file)
        
        self.assertEqual(centerline.shape[0], 5)
        self.assertEqual(centerline.shape[1], 3)
        self.assertEqual(len(radii), 5)
        
        # Check data integrity
        self.assertTrue(np.all(np.isfinite(centerline)))
        self.assertTrue(np.all(np.isfinite(radii)))
        self.assertTrue(np.all(radii > 0))

    def test_get_excel_sheet_names_and_columns(self):
        """Test workbook metadata discovery for import selection."""
        sheet_names = DataLoader.get_excel_sheet_names(self.valid_file)
        columns = DataLoader.get_excel_columns(self.valid_file, "Sheet1")

        self.assertEqual(sheet_names, ["Sheet1"])
        self.assertEqual(columns, ["Easting", "Northing", "Elev", "Radius"])

    def test_load_main_data_with_custom_sheet_and_mapping(self):
        """Test loading from a user-selected sheet with custom column names."""
        mapped_file = os.path.join(self.temp_dir, "mapped_data.xlsx")
        mapped_df = pd.DataFrame({
            "X Coord": [100000.0, 100100.0, 100200.0, 100300.0, 100400.0],
            "Y Coord": [200000.0, 200100.0, 200200.0, 200300.0, 200400.0],
            "Height": [50.0, 52.0, 55.0, 53.0, 51.0],
            "Hole Radius": [800.0, 850.0, 900.0, 880.0, 820.0]
        })

        with pd.ExcelWriter(mapped_file) as writer:
            pd.DataFrame({"Ignore": [1, 2, 3]}).to_excel(writer, sheet_name="Summary", index=False)
            mapped_df.to_excel(writer, sheet_name="Survey", index=False)

        centerline, radii = DataLoader.load_main_data(
            mapped_file,
            sheet_name="Survey",
            column_mapping={
                "Easting": "X Coord",
                "Northing": "Y Coord",
                "Elev": "Height",
                "Radius": "Hole Radius",
            },
        )

        self.assertEqual(centerline.shape, (5, 3))
        self.assertEqual(len(radii), 5)
        np.testing.assert_array_equal(centerline[:, 0], mapped_df["X Coord"].to_numpy())
        np.testing.assert_array_equal(centerline[:, 1], mapped_df["Y Coord"].to_numpy())
        np.testing.assert_array_equal(centerline[:, 2], mapped_df["Height"].to_numpy())
        np.testing.assert_array_equal(radii, mapped_df["Hole Radius"].to_numpy())

    def test_load_main_data_ignores_placeholder_missing_values(self):
        """Test placeholder markers like '-' are treated as missing rows."""
        placeholder_file = os.path.join(self.temp_dir, "placeholder_data.xlsx")
        df = pd.DataFrame({
            "Easting": [100000.0, 100100.0, 100200.0, 100300.0],
            "Northing": [200000.0, 200100.0, 200200.0, 200300.0],
            "Elev": [50.0, 52.0, 55.0, 53.0],
            "Radius": [800.0, "-", 900.0, 880.0],
        })
        df.to_excel(placeholder_file, sheet_name="Sheet1", index=False)

        centerline, radii = DataLoader.load_main_data(placeholder_file)

        self.assertEqual(centerline.shape, (3, 3))
        np.testing.assert_array_equal(radii, np.array([800.0, 900.0, 880.0]))
    
    def test_load_project_data(self):
        """Test loading data without radii requirement."""
        centerline = DataLoader.load_project_data(self.valid_file)
        
        self.assertEqual(centerline.shape[0], 5)
        self.assertEqual(centerline.shape[1], 3)
    
    def test_file_not_found(self):
        """Test error handling for missing file."""
        with self.assertRaises(DataValidationError):
            DataLoader.load_main_data("/nonexistent/file.xlsx")
    
    def test_missing_columns(self):
        """Test error handling for missing required columns."""
        # Create file without Radius column
        df = pd.DataFrame({
            "Easting": [100000.0, 100100.0],
            "Northing": [200000.0, 200100.0],
            "Elev": [50.0, 52.0]
        })
        bad_file = os.path.join(self.temp_dir, "missing_column.xlsx")
        df.to_excel(bad_file, sheet_name="Sheet1", index=False)
        
        with self.assertRaises(DataValidationError):
            DataLoader.load_main_data(bad_file)
    
    def test_insufficient_points(self):
        """Test error handling for too few survey points."""
        df = pd.DataFrame({
            "Easting": [100000.0, 100100.0],
            "Northing": [200000.0, 200100.0],
            "Elev": [50.0, 52.0],
            "Radius": [800.0, 850.0]
        })
        small_file = os.path.join(self.temp_dir, "small_data.xlsx")
        df.to_excel(small_file, sheet_name="Sheet1", index=False)
        
        with self.assertRaises(DataValidationError):
            DataLoader.load_main_data(small_file)
    
    def test_negative_radii(self):
        """Test error handling for invalid (negative) radii."""
        df = pd.DataFrame({
            "Easting": [100000.0, 100100.0, 100200.0, 100300.0, 100400.0],
            "Northing": [200000.0, 200100.0, 200200.0, 200300.0, 200400.0],
            "Elev": [50.0, 52.0, 55.0, 53.0, 51.0],
            "Radius": [800.0, 850.0, -100.0, 880.0, 820.0]  # Negative value
        })
        bad_file = os.path.join(self.temp_dir, "negative_radii.xlsx")
        df.to_excel(bad_file, sheet_name="Sheet1", index=False)
        
        with self.assertRaises(DataValidationError):
            DataLoader.load_main_data(bad_file)


class TestTubeUtils(unittest.TestCase):
    """Test pipe geometry and simulation utilities."""
    
    def setUp(self):
        """Create test centerline data."""
        # Simple straight line (3 points)
        self.simple_centerline = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0]
        ])
        
        # Curved line (bent centerline)
        self.curved_centerline = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 5.0, 0.0],
            [30.0, 10.0, 0.0],
            [40.0, 10.0, 0.0]
        ])
    
    def test_compute_tnb_frame_straight_line(self):
        """Test TNB frame computation on straight line."""
        tangents, normals, binormals = compute_tnb_frame(self.simple_centerline)
        
        # Check shapes
        self.assertEqual(tangents.shape, self.simple_centerline.shape)
        self.assertEqual(normals.shape, self.simple_centerline.shape)
        self.assertEqual(binormals.shape, self.simple_centerline.shape)
        
        # Check unit vectors
        tangent_norms = np.linalg.norm(tangents, axis=1)
        normal_norms = np.linalg.norm(normals, axis=1)
        binormal_norms = np.linalg.norm(binormals, axis=1)
        
        np.testing.assert_array_almost_equal(tangent_norms, np.ones(3), decimal=6)
        np.testing.assert_array_almost_equal(normal_norms, np.ones(3), decimal=6)
        np.testing.assert_array_almost_equal(binormal_norms, np.ones(3), decimal=6)
    
    def test_compute_tnb_frame_orthogonality(self):
        """Test that TNB vectors are orthogonal."""
        tangents, normals, binormals = compute_tnb_frame(self.curved_centerline)
        
        # T · N = 0, T · B = 0, N · B = 0
        for i in range(len(tangents)):
            tn_dot = np.dot(tangents[i], normals[i])
            tb_dot = np.dot(tangents[i], binormals[i])
            nb_dot = np.dot(normals[i], binormals[i])
            
            self.assertAlmostEqual(tn_dot, 0.0, places=6)
            self.assertAlmostEqual(tb_dot, 0.0, places=6)
            self.assertAlmostEqual(nb_dot, 0.0, places=6)
    
    def test_simulate_pipe_deflection_invalid_input(self):
        """Test error handling for invalid simulation inputs."""
        with self.assertRaises(ValueError):
            # Invalid array shape
            simulate_pipe_deflection(np.array([1, 2, 3]), 0.5, 1.0, 0.1, 10)
        
        with self.assertRaises(ValueError):
            # Negative radius
            simulate_pipe_deflection(self.simple_centerline, -0.5, 1.0, 0.1, 10)
        
        with self.assertRaises(ValueError):
            # Invalid stiffness
            simulate_pipe_deflection(self.simple_centerline, 0.5, 1.0, 2.0, 10)
    
    def test_simulate_pipe_deflection_no_contact(self):
        """Test deflection simulation with no contact (bore >> pipe)."""
        deflected, contact_pts, contact_pos, contact_forces = simulate_pipe_deflection(
            self.curved_centerline,
            tube_radius=0.1,
            bore_radius=10.0,
            stiffness=0.1,
            iterations=10
        )
        
        # No contact should occur
        self.assertTrue(np.sum(contact_pts) == 0)
        self.assertTrue(len(contact_pos) == 0)
        self.assertTrue(np.all(contact_forces == 0))
    
    def test_compute_contact_angles_valid(self):
        """Test contact angle computation."""
        contact_indices = [1, 3]
        contact_positions = [
            np.array([10.0, 0.5, 0.0]),
            np.array([30.0, 10.5, 0.0])
        ]
        
        results = compute_contact_angles(self.curved_centerline, contact_indices, contact_positions)
        
        self.assertEqual(len(results), 2)
        for distance, angle in results:
            self.assertGreaterEqual(distance, 0)
            self.assertGreaterEqual(angle, -180)
            self.assertLessEqual(angle, 180)
    
    def test_compute_contact_angles_mismatch(self):
        """Test error handling for mismatched contact data."""
        with self.assertRaises(ValueError):
            compute_contact_angles(
                self.curved_centerline,
                contact_indices=[1, 2],
                contact_positions=[np.array([10, 0, 0])]  # Mismatched length
            )


class TestTubeGenerator(unittest.TestCase):
    """Test mesh generation."""
    
    def setUp(self):
        """Create test centerline."""
        self.simple_centerline = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0]
        ])
    
    def test_compute_tube_geometry_shape(self):
        """Test mesh generation produces correct vertex and face counts."""
        n_points = 12
        vertices, faces, normals = TubeGenerator.compute_tube_geometry(
            self.simple_centerline,
            radius=1.0,
            n_points=n_points
        )
        
        # Check vertex count: (centerline_points × circle_points)
        expected_vertex_count = len(self.simple_centerline) * n_points
        self.assertEqual(vertices.shape[0], expected_vertex_count)
        self.assertEqual(vertices.shape[1], 3)
        
        # Check face count: 2 triangles per quad per segment
        expected_face_count = 2 * (len(self.simple_centerline) - 1) * n_points
        self.assertEqual(len(faces), expected_face_count)
    
    def test_compute_tube_geometry_normals_unit(self):
        """Test that vertex normals are unit vectors."""
        vertices, faces, normals = TubeGenerator.compute_tube_geometry(
            self.simple_centerline,
            radius=1.0,
            n_points=12
        )
        
        # Check all normals are unit vectors
        normal_magnitudes = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(normal_magnitudes, np.ones(len(normals)), decimal=6)
    
    def test_compute_tube_geometry_invalid_input(self):
        """Test error handling for invalid inputs."""
        with self.assertRaises(ValueError):
            # Less than 2 points
            TubeGenerator.compute_tube_geometry(np.array([[0, 0, 0]]), 1.0, 12)


class TestColorUtils(unittest.TestCase):
    """Test stress visualization color mapping."""
    
    def test_gradient_color_thresholds(self):
        """Test color mapping across thresholds."""
        # Test each threshold range
        color_very_low = ColorUtils.get_gradient_color(500)
        self.assertEqual(color_very_low, constants.STRESS_COLOR_VERY_LOW)
        
        color_low = ColorUtils.get_gradient_color(700)
        self.assertEqual(color_low, constants.STRESS_COLOR_LOW)
        
        color_medium = ColorUtils.get_gradient_color(900)
        self.assertEqual(color_medium, constants.STRESS_COLOR_MEDIUM)
        
        color_high = ColorUtils.get_gradient_color(1200)
        self.assertEqual(color_high, constants.STRESS_COLOR_HIGH)
        
        color_very_high = ColorUtils.get_gradient_color(1700)
        self.assertEqual(color_very_high, constants.STRESS_COLOR_VERY_HIGH)
    
    def test_color_is_rgba(self):
        """Test that all color returns are valid RGBA tuples."""
        test_values = [0, 500, 1000, 1500, 2000, 3000]
        
        for val in test_values:
            color = ColorUtils.get_gradient_color(val)
            
            # Check is 4-tuple
            self.assertEqual(len(color), 4)
            
            # Check RGBA values in valid range [0, 1]
            for component in color:
                self.assertGreaterEqual(component, 0.0)
                self.assertLessEqual(component, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_file = os.path.join(self.temp_dir, "integration_test.xlsx")
        
        # Create reasonable survey data
        df = pd.DataFrame({
            "Easting": np.linspace(100000, 100100, 50),
            "Northing": np.linspace(200000, 200050, 50),
            "Elev": np.linspace(50, 100, 50),
            "Radius": np.random.uniform(800, 1000, 50)
        })
        df.to_excel(self.valid_file, sheet_name="Sheet1", index=False)
    
    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """Test complete data → geometry → visualization pipeline."""
        # Load data
        centerline, radii = DataLoader.load_main_data(self.valid_file)
        
        # Generate mesh
        vertices, faces, normals = TubeGenerator.compute_tube_geometry(
            centerline,
            radius=0.6096,
            n_points=12
        )
        
        # Simulate deflection
        deflected, contact_pts, contact_pos, contact_forces = simulate_pipe_deflection(
            centerline,
            tube_radius=0.6096,
            bore_radius=np.mean(radii),
            stiffness=0.1,
            iterations=10
        )
        
        # Check outputs
        self.assertEqual(deflected.shape, centerline.shape)
        self.assertEqual(len(contact_pts), len(centerline))
        self.assertEqual(len(contact_forces), len(centerline))
        self.assertGreaterEqual(len(vertices), len(centerline) * 8)
        self.assertGreater(len(faces), 0)


if __name__ == '__main__':
    unittest.main()
