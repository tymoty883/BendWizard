import pandas as pd
import numpy as np

class DataLoader:
    @staticmethod
    def load_centerline_data(file_path, sheet_name, required_columns):
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError:
            df = pd.read_excel(file_path, sheet_name=0)
            print(f"Warning: '{sheet_name}' not found, using first sheet instead.")
        df = df.dropna(subset=required_columns)
        print(f"Loaded {len(df)} points from {file_path}")
        coordinates = df[required_columns].values
        return np.array(coordinates)

    @staticmethod
    def load_main_data(file_path):
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        before = len(df)
        df = df.dropna(subset=["Easting", "Northing", "Elev", "Radius"])
        after = len(df)
        if after < before:
            print(f"Filtered out {before - after} rows with NaN values.")
        easting = df["Easting"].values
        northing = df["Northing"].values
        elevation = df["Elev"].values
        radius = df["Radius"].values
        return easting, northing, elevation, radius
