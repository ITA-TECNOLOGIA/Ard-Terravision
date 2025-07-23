import os
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from pyproj import CRS as pyprojCRS
import h5py

def extract_prisma_data(path_img):
    with h5py.File(path_img, 'r') as h5f:
        # Extract VNIR and SWIR cubes (HS data)
        CW = np.concatenate([h5f.attrs['List_Cw_Vnir'][::-1], h5f.attrs['List_Cw_Swir'][::-1]])
        Flag = np.concatenate([h5f.attrs['CNM_VNIR_SELECT'][::-1], h5f.attrs['CNM_SWIR_SELECT'][::-1]])
        SWIR = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'])
        VNIR = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'])
        SWIR = np.swapaxes(SWIR, 1, 2)[:, :, ::-1]
        VNIR = np.swapaxes(VNIR, 1, 2)[:, :, ::-1]

        # Scale DN to reflectance
        SWIR = (h5f.attrs['L2ScaleSwirMin'] + SWIR * (h5f.attrs['L2ScaleSwirMax'] - h5f.attrs['L2ScaleSwirMin']) / 65535).astype(np.float32)
        VNIR = (h5f.attrs['L2ScaleVnirMin'] + VNIR * (h5f.attrs['L2ScaleVnirMax'] - h5f.attrs['L2ScaleVnirMin']) / 65535).astype(np.float32)

        hs_img = np.concatenate([VNIR, SWIR], axis=2)

        # Filtering bands (water absorption, defective)
        def search_band_index(cw, target):
            return np.argmin(np.abs(cw - target))

        water_bands_vnir = [920, 423.78476, 415.839, 406.9934]
        water_bands_swir = list(range(1350, 1480, 10)) + list(range(1800, 1960, 10)) + [1120, 2020] + \
            [2497.1155, 2490.2192, 2483.793, 2477.055, 2469.6272, 2463.0303, 2456.5857, 2449.1423,
             2442.403, 2435.5442, 2428.6677, 2421.2373, 2414.3567, 2407.6045, 2400.036, 2393.0388,
             2386.0618, 2378.771, 2371.5522, 2364.5945, 2357.2937, 2349.7915, 2342.8228]

        bands_to_remove = [search_band_index(CW, w) for w in water_bands_vnir + water_bands_swir]
        zero_flagged = list(np.where(Flag == 0)[0])
        zero_wavelength = list(np.where(CW == 0.0)[0])

        SWIR_err = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_PIXEL_L2_ERR_MATRIX'][()], dtype=np.uint16)[::-1]
        VNIR_err = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX'][()], dtype=np.uint16)[::-1]

        def count_defective_pixels(arr):
            zero_counts = np.sum(arr == 0, axis=(0, 2))
            total = arr.shape[0] * arr.shape[2]
            return list(np.where((zero_counts / total) * 100 < 90)[0])

        bands_to_remove += count_defective_pixels(SWIR_err)
        bands_to_remove += count_defective_pixels(VNIR_err)
        bands_to_remove += zero_flagged + zero_wavelength

        bands_to_keep = np.setdiff1d(np.arange(hs_img.shape[2]), bands_to_remove)
        hs_img_filtered = hs_img[:, :, bands_to_keep]
        CW_filtered = CW[bands_to_keep]

        # Extract geolocation and angles
        lat = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude'])
        lon = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude'])
        solar_zenith = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Geometric Fields/Solar_Zenith_Angle'])
        solar_azimuth = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Geometric Fields/Rel_Azimuth_Angle'])
        observing_angle = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Geometric Fields/Observing_Angle'])

        # Get acquisition time from filename
        filename = os.path.basename(path_img)
        try:
            timestamp = datetime.strptime(filename.split('_')[4], "%Y%m%d%H%M%S")
        except:
            timestamp = None

        return hs_img_filtered, CW_filtered, lat, lon, solar_zenith, solar_azimuth, observing_angle, timestamp

from pyproj import Transformer
import numpy as np
import utm

def infer_utm_epsg(lat, lon):
    lat0 = lat[lat.shape[0] // 2, lat.shape[1] // 2]
    lon0 = lon[lat.shape[0] // 2, lat.shape[1] // 2]
    zone_number, zone_letter = utm.from_latlon(lat0, lon0)[2:4]
    return f"EPSG:{32600 + zone_number}" if zone_letter >= 'N' else f"EPSG:{32700 + zone_number}"

def convert_latlon_to_utm_coords(lat, lon, epsg_out):
    h, w = lat.shape
    assert lat.shape == lon.shape, "Lat/lon shape mismatch."

    lat_line = lat[h // 2, :]
    lon_line = lon[h // 2, :]
    lat_col = lat[:, w // 2]
    lon_col = lon[:, w // 2]

    transformer = Transformer.from_crs("EPSG:4326", epsg_out, always_xy=True)
    x_utm, _ = transformer.transform(lon_line, lat_line)
    _, y_utm = transformer.transform(lon_col, lat_col)

    return np.array(x_utm), np.array(y_utm)



def save_to_netcdf(hs, wavelengths, x_coords, y_coords,
                   sun_zenith, sun_azimuth, view_zenith,
                   timestamp, output_path, epsg_code):



    h, w, bands = hs.shape

    with Dataset(output_path, 'w', format='NETCDF4') as ds:
        # Dimensions
        ds.createDimension('t', None)
        ds.createDimension('y', h)
        ds.createDimension('x', w)

        # Coordinates
        t_var = ds.createVariable('t', 'f8', ('t',))
        t_var.units = 'seconds since 1970-01-01 00:00:00'
        t_var.calendar = 'standard'
        t_var.standard_name = 'time'
        t_var[0] = (timestamp - datetime(1970, 1, 1)).total_seconds() if timestamp else 0

        x_var = ds.createVariable('x', 'f8', ('x',))
        x_var.units = 'm'
        x_var.long_name = 'Easting coordinate in UTM'
        x_var[:] = x_coords  # shape (w,)

        y_var = ds.createVariable('y', 'f8', ('y',))
        y_var.units = 'm'
        y_var.long_name = 'Northing coordinate in UTM'
        y_var[:] = y_coords  # shape (h,)
        
        # CRS / Projection info (grid_mapping variable)
        crs = ds.createVariable('crs', 'S1')
        crs.grid_mapping_name = 'transverse_mercator'
        crs.long_name = 'CRS definition'
        crs.spatial_ref = epsg_code
        crs.epsg_code = int(epsg_code.split(":")[1])

        # Dynamically extract ellipsoid from EPSG
        proj = pyprojCRS.from_epsg(crs.epsg_code)
        crs.semi_major_axis = proj.ellipsoid.semi_major_metre
        crs.inverse_flattening = proj.ellipsoid.inverse_flattening


        # Hyperspectral bands as separate variables (no band dimension)
        for i in range(bands):
            band_name = f'B{i+1:03d}'
            band_var = ds.createVariable(band_name, 'f4', ('t', 'y', 'x'), zlib=True, complevel=4)
            band_var[0, :, :] = hs[:, :, i]
            band_var.units = 'reflectance'
            band_var.long_name = f'Hyperspectral Reflectance - Band {band_name}'
            band_var.wavelength = float(wavelengths[i])  # optional
            band_var.grid_mapping = "crs"

                # Sentinel-like variables (faltantes en PRISMA, pero añadidos como placeholders)

        # WVP (Water Vapor Product) - scalar placeholder
        wvp_var = ds.createVariable('WVP', 'f4')
        wvp_var.long_name = 'Water Vapor Product'
        wvp_var.units = 'g/cm^2'  # o 'kg/m^2' si prefieres Sentinel style
        wvp_var.grid_mapping = "crs"
        wvp_var.assignValue(np.float32(np.nan))  # scalar NaN

        # AOT (Aerosol Optical Thickness) - scalar placeholder
        aot_var = ds.createVariable('AOT', 'f4')  # f8 = float64
        aot_var.long_name = 'Aerosol Optical Thickness'
        aot_var.units = 'unitless'
        aot_var.grid_mapping = "crs"
        aot_var.assignValue(np.float32(np.nan))  # scalar NaN

        # SCL (Scene Classification Layer) - scalar placeholder
        scl_var = ds.createVariable('SCL', 'f4')  # use float64 for consistent output
        scl_var.long_name = 'Scene Classification Layer'
        scl_var.grid_mapping = "crs"
        scl_var.assignValue(np.float32(np.nan))  # scalar NaN (even though it's categorical)


        sunAzimuth_var = ds.createVariable('sunAzimuthAngles', 'f4', ('t' ,'y', 'x'))
        sunAzimuth_var[0, :, :] = sun_azimuth
        sunAzimuth_var.units = 'degrees'

        sunZenith_var = ds.createVariable('sunZenithAngles', 'f4', ('t', 'y', 'x'))
        sunZenith_var[0, :, :] = sun_zenith
        sunZenith_var.units = 'degrees'

        # viewAzimuthMean (faltante en PRISMA)
        viewAzimuth_var = ds.createVariable('viewAzimuthMean', 'f4')  # f8 = float64
        viewAzimuth_var.long_name = 'Sensor View Azimuth Angle'
        viewAzimuth_var.units = 'degrees'
        viewAzimuth_var.assignValue(np.float32(np.nan))  # scalar NaN


        viewZenith_var = ds.createVariable('viewZenithMean', 'f4', ('t', 'y', 'x'))
        viewZenith_var[0, :, :] = view_zenith
        viewZenith_var.units = 'degrees'



        sunAzimuth_var.grid_mapping = "crs"
        sunZenith_var.grid_mapping = "crs"
        viewAzimuth_var.grid_mapping = "crs"
        viewZenith_var.grid_mapping = "crs"
        

        # Global attributes
        ds.Conventions = "CF-1.9"
        ds.title = 'PRISMA L2D Hyperspectral Datacube'
        ds.source = 'ASI PRISMA satellite - L2D product'
        ds.history = f'Created on {datetime.now().isoformat()}'
        ds.comment = 'Bands filtered, UTM projected, CF-compliant output.'




if __name__ == "__main__":
    input_file = '/datassd/proyectos/terravision/terravision_PRISMA/PRS_L2D_STD_20220712110958_20220712111002_0001.he5'
    output_nc = 'PRISMANETCDF/output_prisma_datacube.nc'

    hs, wavelengths, lat, lon, solar_zenith, solar_azimuth, observing_angle, timestamp = extract_prisma_data(input_file)
    
    epsg_code = infer_utm_epsg(lat, lon)
    x_coords, y_coords = convert_latlon_to_utm_coords(lat, lon, epsg_code)

    save_to_netcdf(
        hs, wavelengths, x_coords, y_coords,
        solar_zenith, solar_azimuth, observing_angle,
        timestamp, output_nc, epsg_code
    )
    print(f"Saved NetCDF to {output_nc}")

