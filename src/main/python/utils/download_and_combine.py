import os
import openeo
import xarray as xr
from datetime import datetime
from dotenv import load_dotenv
from openeo_downloader import download_data

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CLIENT_ID = os.getenv("OPENEO_CLIENT_ID")
CLIENT_SECRET = os.getenv("OPENEO_CLIENT_SECRET")
SHAPEFILES_PATH = os.getenv("SHAPEFILES_PATH")
OPENEO_DOWNLOADS_PATH = os.getenv("OPENEO_DOWNLOADS_PATH")

# --- Main execution ---
if __name__ == "__main__":
    # 1. Connect and authenticate to OpenEO
    try:
        connection = openeo.connect("openeo.dataspace.copernicus.eu")
        connection.authenticate_oidc_client_credentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        print("Successfully connected and authenticated to OpenEO.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        exit()

    # 2. Define download parameters
    shapefile_name = "canteras.shp"  # CHANGE this to your desired shapefile
    shapefile_path = os.path.join(SHAPEFILES_PATH, shapefile_name)

    start_date_1 = datetime(2017, 10, 1)
    end_date_1 = datetime(2017, 10, 15)
    start_date_2 = datetime(2025, 10, 16)
    end_date_2 = datetime(2025, 10, 30)

    # 3. Download the two datacubes
    try:
        print(f"--- Starting download for period 1 ({start_date_1.date()} to {end_date_1.date()}) ---")
        file_path_1 = download_data(connection, shapefile_path, start_date_1, end_date_1, synchronous=True)
        print(f"Downloaded first file to: {file_path_1}")

        print(f"--- Starting download for period 2 ({start_date_2.date()} to {end_date_2.date()}) ---")
        file_path_2 = download_data(connection, shapefile_path, start_date_2, end_date_2, synchronous=True)
        print(f"Downloaded second file to: {file_path_2}")

    except Exception as e:
        print(f"Data download failed: {e}")
        exit()

    # 4. Combine the two NetCDF files
    try:
        print("\n--- Combining downloaded files ---")
        dataset1 = xr.open_dataset(file_path_1)
        dataset2 = xr.open_dataset(file_path_2)

        # Combine along the 't' dimension (time)
        combined_dataset = xr.concat([dataset1, dataset2], dim="t")
        
        # Sort by time to ensure chronological order
        combined_dataset = combined_dataset.sortby("t")

        print("Files combined successfully.")

    except Exception as e:
        print(f"Failed to combine NetCDF files: {e}")
        exit()
        
    # 5. Save the combined file
    try:
        combined_file_name = f"combined_{shapefile_name.replace('.shp', '')}_{start_date_1.date()}_{end_date_2.date()}.nc"
        combined_output_path = os.path.join(OPENEO_DOWNLOADS_PATH, combined_file_name)

        combined_dataset.to_netcdf(combined_output_path)
        print(f"Combined file saved to: {combined_output_path}")

    except Exception as e:
        print(f"Failed to save the combined file: {e}")
        exit()

    print("\n--- Process finished ---")
