import os
import re
import xarray as xr
import rasterio
import numpy as np
import xml.etree.ElementTree as ET
from rasterio.plot import reshape_as_raster

def extract_center_angle(root, tag_name):
    for elem in root.iter(tag_name):
        for child in elem:
            if child.tag.lower().endswith("center"):
                return float(child.text)
    return np.nan

# === File Paths ===
data_path = '/datassd/proyectos/terravision/terravision_EnMAP'

for mine in os.listdir(data_path):
    mine_name = mine.split('_')[-1]
    if mine_name == 'Zarza':
        mine_name = 'La_Zarza'
    enmap_path = os.path.join(data_path, mine, 'ENMAP.HSI.L2A')
    enmap_folder = os.listdir(enmap_path)[0]
    tif_path = os.path.join(enmap_path, enmap_folder, enmap_folder+'-SPECTRAL_IMAGE.TIF')
    xml_path = os.path.join(enmap_path, enmap_folder, enmap_folder+'-METADATA.XML')
    print("Reading TIF file: ", tif_path)
    print("Reading XML file: ", xml_path)

    # === Load the GeoTIFF ===
    with rasterio.open(tif_path) as src:
        data = src.read()  # (bands, rows, cols)
        transform = src.transform
        width = src.width
        height = src.height
        x_coords = np.arange(width) * transform.a + transform.c
        y_coords = np.arange(height) * transform.e + transform.f
        # Extract time coordinate from the file
        timestamps = re.findall(r'\d{8}T\d{6}Z', enmap_folder)
        time = timestamps[0]
        time_formatted = f"{time[:4]}-{time[4:6]}-{time[6:8]}T{time[9:11]}:{time[11:13]}:{time[13:15]}"
        t_coords = [np.datetime64(time_formatted)]

        # Create Dataset
        coords = {"t": t_coords, "x": x_coords, "y": y_coords}
        ds = xr.Dataset(coords=coords)

        # === Parse METADATA.XML for wavelengths and angles ===
        tree = ET.parse(xml_path)
        root = tree.getroot()

        wavelengths = []
        for elem in root.iter():
            if elem.tag.lower().endswith("wavelengthcenterofband"):
                wavelengths.append(float(elem.text))

        # === Add CRS variable ===
        ds["crs"] = xr.DataArray(np.array("1", dtype="S1"))
        ds["crs"].attrs["grid_mapping_name"] = 'transverse_mercator'
        ds["crs"].attrs["long_name"] = 'CRS definition'
        ds["crs"].attrs["spatial_ref"] = ":".join(src.crs.to_authority())
        ds["crs"].attrs["epsg_code"] = int(src.crs.to_epsg())
        # Hardcoded from WTK
        ds["crs"].attrs["wkt"] = src.crs.to_wkt()
        ds["crs"].attrs["scale_factor_at_central_meridian"] = 0.9996
        ds["crs"].attrs["longitude_of_central_meridian"] = 21.0
        ds["crs"].attrs["latitude_of_projection_origin"] = 0.0
        ds["crs"].attrs["false_easting"] = 500000.0
        ds["crs"].attrs["false_northing"] = 0.0
        ds["crs"].attrs["semi_major_axis"] = 6378137.0

        # Add each band as a separate variable
        for i in range(data.shape[0]):
            band_id = f"B{i+1:03}"
            ds[band_id] = (("t", "y", "x"), data[i][np.newaxis, :, :])
            ds[band_id].attrs["wavelength_nm"] = wavelengths[i]
            ds[band_id].attrs["grid_mapping"] = "crs"

        # Add maps
        ds["WVP"] = np.nan
        ds["AOT"] = np.nan
        ds["SCL"] = np.nan
        ds["WVP"].attrs["grid_mapping"] = "crs"
        ds["AOT"].attrs["grid_mapping"] = "crs"
        ds["SCL"].attrs["grid_mapping"] = "crs"

        # Add angles (also spatial, so attach grid_mapping)
        ds["sunAzimuthAngles"] = (("t", "y", "x"), np.full((1, height, width), extract_center_angle(root, "sunAzimuthAngle"), dtype=np.float32))
        ds["sunZenithAngles"] = np.nan
        ds["viewAzimuthAngles"] = (("t", "y", "x"), np.full((1, height, width), extract_center_angle(root, "viewingAzimuthAngle"), dtype=np.float32))
        ds["viewZenithAngles"] = (("t", "y", "x"), np.full((1, height, width), extract_center_angle(root, "viewingZenithAngle"), dtype=np.float32))

        for angle_var in ["sunAzimuthAngles", "viewAzimuthAngles", "viewZenithAngles"]:
            ds[angle_var].attrs["grid_mapping"] = "crs"

        # === Global Attributes ===
        ds.attrs["title"] = "EnMAP Level-2A Hyperspectral Datacube"
        ds.attrs["source"] = "DLR EnMAP Mission"
        ds.attrs["acquisition_time"] = time_formatted
        ds.attrs["wavelengths_nm"] = wavelengths
        ds.attrs["comment"] = "Bands filtered, UTM projected, CF-compliant output."

        # === Write to Disk ===
        print("Writing Datacube on disk...")
        out_path = os.path.join(enmap_path, enmap_folder, f"enmap_l2a_datacube_{mine_name}.nc")
        ds.to_netcdf(out_path)
        print("Done!")
        print(ds)
