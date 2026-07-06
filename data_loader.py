"""
Data loading module with input validation and error handling.
Loads survey data from Excel files with comprehensive validation.
"""

from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd
import constants


class DataValidationError(Exception):
    """Raised when input data validation fails."""
    pass


class DataLoader:
    """
    Loads and validates borehole survey data from Excel files.
    
    Provides methods to load project data (without radii) and main data (with radii),
    with comprehensive validation including schema checking, data type validation,
    and plausibility bounds checking.
    """

    MISSING_VALUE_MARKERS = {"", "-", "--"}

    @staticmethod
    def validate_file_exists(file_path: str) -> None:
        """
        Validate that the file exists and is accessible.
        
        Args:
            file_path: Path to Excel file
            
        Raises:
            DataValidationError: If file doesn't exist or is inaccessible
        """
        if not os.path.exists(file_path):
            raise DataValidationError(constants.ERROR_FILE_NOT_FOUND.format(file_path))
        
        # Check file size to prevent DoS
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > constants.MAX_FILE_SIZE_MB:
            raise DataValidationError(
                f"File too large: {file_size_mb:.1f} MB. Maximum allowed: {constants.MAX_FILE_SIZE_MB} MB"
            )

    @staticmethod
    def get_excel_sheet_names(file_path: str) -> List[str]:
        """Return available sheet names for an Excel workbook."""
        try:
            DataLoader.validate_file_exists(file_path)
            with pd.ExcelFile(file_path) as workbook:
                return list(workbook.sheet_names)
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(f"Excel parsing error: {str(e)}") from e

    @staticmethod
    def get_excel_columns(file_path: str, sheet_name: str) -> List[str]:
        """Return column names for a sheet in an Excel workbook."""
        try:
            DataLoader.validate_file_exists(file_path)
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
            return [str(column) for column in df.columns]
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(f"Excel parsing error: {str(e)}") from e

    @staticmethod
    def _read_excel_sheet(file_path: str, sheet_name: str) -> pd.DataFrame:
        """Read a specific sheet from an Excel file."""
        return pd.read_excel(file_path, sheet_name=sheet_name)

    @staticmethod
    def _normalize_column_mapping(column_mapping: Dict[str, str], required_fields: List[str]) -> Dict[str, str]:
        """Validate and normalize requested column mappings."""
        normalized_mapping: Dict[str, str] = {}
        for field in required_fields:
            column_name = column_mapping.get(field, field)
            if not column_name or not str(column_name).strip():
                raise DataValidationError(f"Missing column mapping for '{field}'")
            normalized_mapping[field] = str(column_name)

        selected_columns = list(normalized_mapping.values())
        if len(set(selected_columns)) != len(selected_columns):
            raise DataValidationError("Each field must map to a different Excel column")

        return normalized_mapping

    @staticmethod
    def _load_dataframe(
        file_path: str,
        required_fields: List[str],
        sheet_name: str,
        column_mapping: Dict[str, str],
    ) -> pd.DataFrame:
        """Load and validate a DataFrame using the requested sheet and column mapping."""
        df = DataLoader._read_excel_sheet(file_path, sheet_name)
        normalized_mapping = DataLoader._normalize_column_mapping(column_mapping, required_fields)
        DataLoader.validate_excel_columns(df, list(normalized_mapping.values()))
        DataLoader.validate_numeric_data(df, list(normalized_mapping.values()))

        for column_name in normalized_mapping.values():
            df[column_name] = DataLoader._coerce_numeric_series(df[column_name], column_name)

        before = len(df)
        df = df.dropna(subset=list(normalized_mapping.values())).copy()
        after = len(df)

        if after == 0:
            raise DataValidationError(constants.ERROR_EMPTY_DATASET)

        if after < before:
            print(f"Filtered out {before - after} rows with NaN values.")

        return df.rename(columns={value: key for key, value in normalized_mapping.items()})

    @staticmethod
    def validate_excel_columns(df: pd.DataFrame, required_columns: list) -> None:
        """
        Validate that DataFrame has all required columns.
        
        Args:
            df: Pandas DataFrame
            required_columns: List of required column names
            
        Raises:
            DataValidationError: If required columns are missing
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(
                constants.ERROR_MISSING_COLUMNS.format(
                    ", ".join(missing_columns),
                    ", ".join(required_columns)
                )
            )

    @staticmethod
    def validate_numeric_data(df: pd.DataFrame, columns: list) -> None:
        """
        Validate that specified columns contain numeric data.
        
        Args:
            df: Pandas DataFrame
            columns: List of column names to validate
            
        Raises:
            DataValidationError: If columns contain non-numeric data
        """
        for col in columns:
            if col in df.columns:
                DataLoader._coerce_numeric_series(df[col], col)

    @staticmethod
    def _coerce_numeric_series(series: pd.Series, column_name: str) -> pd.Series:
        """Convert a column to numeric values while treating common placeholders as missing."""
        cleaned_series = series.copy()

        if pd.api.types.is_object_dtype(cleaned_series) or pd.api.types.is_string_dtype(cleaned_series):
            cleaned_series = cleaned_series.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            cleaned_series = cleaned_series.replace(list(DataLoader.MISSING_VALUE_MARKERS), np.nan)

        coerced_series = pd.to_numeric(cleaned_series, errors='coerce')
        invalid_mask = cleaned_series.notna() & coerced_series.isna()
        if invalid_mask.any():
            raise DataValidationError(
                constants.ERROR_INVALID_DATA_TYPE.format(column_name)
            )

        return coerced_series

    @staticmethod
    def validate_data_size(data: np.ndarray, min_points: int = constants.MIN_SURVEY_POINTS) -> None:
        """
        Validate that data array has acceptable number of points.
        
        Args:
            data: NumPy array of survey points
            min_points: Minimum required points
            
        Raises:
            DataValidationError: If point count is invalid
        """
        num_points = len(data)
        
        if num_points < min_points:
            raise DataValidationError(
                constants.ERROR_INSUFFICIENT_POINTS.format(num_points)
            )
        
        if num_points > constants.MAX_SURVEY_POINTS:
            raise DataValidationError(
                constants.ERROR_EXCESSIVE_POINTS.format(num_points)
            )

    @staticmethod
    def validate_coordinate_plausibility(centerline: np.ndarray) -> None:
        """
        Validate that coordinates are within plausible bounds.
        Issues warning but doesn't fail for slightly out-of-bounds coordinates.
        
        Args:
            centerline: Nx3 array of survey points
            
        Raises:
            DataValidationError: If coordinates are completely implausible
        """
        # Check for NaN or Inf
        if np.any(np.isnan(centerline)) or np.any(np.isinf(centerline)):
            raise DataValidationError("Survey data contains NaN or Inf values")
        
        # Plausibility checks (UTM-like coordinates)
        # UTM zones typically in range 100000-900000, elevations -100 to 10000
        easting = centerline[:, 0]
        northing = centerline[:, 1]
        elevation = centerline[:, 2]
        
        # Check for extremely implausible values
        if np.min(easting) < 0 or np.max(easting) > 10000000:
            raise DataValidationError(f"Easting coordinates out of plausible range: {np.min(easting)}, {np.max(easting)}")
        
        if np.min(northing) < 0 or np.max(northing) > 10000000:
            raise DataValidationError(f"Northing coordinates out of plausible range: {np.min(northing)}, {np.max(northing)}")
        
        if np.min(elevation) < -500 or np.max(elevation) > 10000:
            raise DataValidationError(f"Elevation values out of plausible range: {np.min(elevation)}, {np.max(elevation)}")

    @staticmethod
    def load_project_data(
        file_path: str,
        sheet_name: str = constants.EXCEL_SHEET_NAME,
        column_mapping: Dict[str, str] = None,
    ) -> np.ndarray:
        """
        Load project data (centerline without radii) from Excel file.
        
        Args:
            file_path: Path to .xlsx/.xlsm file with columns [Easting, Northing, Elev]
        
        Returns:
            Nx3 NumPy array of [Easting, Northing, Elevation] coordinates
        
        Raises:
            DataValidationError: If file or data validation fails
            Exception: pandas/Excel parsing errors
        """
        try:
            # Validate file
            DataLoader.validate_file_exists(file_path)
            
            df = DataLoader._load_dataframe(
                file_path,
                constants.REQUIRED_COLUMNS_PROJECT_DATA,
                sheet_name,
                column_mapping or {},
            )
            
            # Extract coordinates
            easting = df["Easting"].values.astype(float)
            northing = df["Northing"].values.astype(float)
            elevation = df["Elev"].values.astype(float)
            
            centerline = np.array([easting, northing, elevation]).T
            
            # Validate data size and plausibility
            DataLoader.validate_data_size(centerline)
            DataLoader.validate_coordinate_plausibility(centerline)
            
            print(f"✓ Successfully loaded {len(easting)} points from Excel file")
            return centerline
            
        except DataValidationError:
            raise
        except FileNotFoundError as e:
            raise DataValidationError(constants.ERROR_FILE_NOT_FOUND.format(file_path)) from e
        except Exception as e:
            raise DataValidationError(f"Excel parsing error: {str(e)}") from e

    @staticmethod
    def load_main_data(
        file_path: str,
        sheet_name: str = constants.EXCEL_SHEET_NAME,
        column_mapping: Dict[str, str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load main data (centerline with bore radii) from Excel file.
        
        Args:
            file_path: Path to .xlsx/.xlsm file with columns [Easting, Northing, Elev, Radius]
        
        Returns:
            Tuple of (Nx3 centerline array, N-array of bore radii in meters)
        
        Raises:
            DataValidationError: If file or data validation fails
            Exception: pandas/Excel parsing errors
        """
        try:
            # Validate file
            DataLoader.validate_file_exists(file_path)
            
            df = DataLoader._load_dataframe(
                file_path,
                constants.REQUIRED_COLUMNS_MAIN_DATA,
                sheet_name,
                column_mapping or {},
            )
            
            # Extract coordinates and radii
            easting = df["Easting"].values.astype(float)
            northing = df["Northing"].values.astype(float)
            elevation = df["Elev"].values.astype(float)
            radius = df["Radius"].values.astype(float)
            
            centerline = np.array([easting, northing, elevation]).T
            
            # Validate data size and plausibility
            DataLoader.validate_data_size(centerline)
            DataLoader.validate_coordinate_plausibility(centerline)
            
            # Validate radii
            if np.any(radius < 0):
                raise DataValidationError("Bore radii must be positive values")
            
            print(f"✓ Successfully loaded {len(easting)} points from Excel file")
            return centerline, radius
            
        except DataValidationError:
            raise
        except FileNotFoundError as e:
            raise DataValidationError(constants.ERROR_FILE_NOT_FOUND.format(file_path)) from e
        except Exception as e:
            raise DataValidationError(f"Excel parsing error: {str(e)}") from e
