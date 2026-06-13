import numpy as np
import pandas as pd

class DataLoader:
    @staticmethod
    def load_project_data(file_path):
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        before = len(df)
        df = df.dropna(subset=["Easting", "Northing", "Elev"])
        after = len(df)
        
        if after < before:
            print(f"Filtered out {before - after} rows with NaN values.")
        
        easting = df["Easting"].values
        northing = df["Northing"].values
        elevation = df["Elev"].values
        
        print(f"Successfully read {len(easting)} points from Excel file after filtering")
        centerline = np.array([easting, northing, elevation]).T
        return centerline

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
        
        print(f"Successfully read {len(easting)} points from Excel file after filtering")
        centerline = np.array([easting, northing, elevation]).T
        return centerline, radius 