import os
import openeo
import geopandas as gpd
from shapely.geometry import mapping
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - [TERRAVISION] - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger('openeo').setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

def get_download_workflow(connection, shape, start, end):
    """
    Defines the openEO workflow for downloading Sentinel-2 L2A data,
    including cloud masking.
    """
    # Generate initial S2 datacube
    s2_datacube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=shape,
        temporal_extent=[start, end]
    )
    
    # Create cloud mask from SCL layer
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=shape,
        temporal_extent=[start,end],
        bands=["SCL"]
    )
    
    cloud_mask = scl.process(
        "to_scl_dilation_mask",
        data=scl,
        kernel1_size=17, kernel2_size=77,
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3)

    # Apply the masking of the cloud and the provided shape
    return s2_datacube.mask(cloud_mask).mask_polygon(shape)


def download_data(connection, shape_file_path, start_date, end_date, synchronous=False):
    """
    Initializes and runs the openEO download job.
    If synchronous, it will download the file and return the path.
    If not, it starts a batch job and returns the job ID.
    """
    try:
        # Read the shapefile
        gdf = gpd.read_file(shape_file_path)
        shape = mapping(gdf.geometry[0])

        # Format dates to strings
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Define output path
        output_dir = os.getenv("OPENEO_DOWNLOADS_PATH")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_name_base = f"s2_l2a_{os.path.basename(shape_file_path).split('.')[0]}_{start_str}_{end_str}"
        
        # Get the datacube with the defined workflow
        datacube = get_download_workflow(connection, shape, start_str, end_str)

        if synchronous:
            file_name = f"{file_name_base}_sync.nc"
            output_path = os.path.join(output_dir, file_name)
            logger.info(f'Starting synchronous download to {output_path}')
            
            # Download the datacube
            datacube.download(output_path)

            logger.info(f"Synchronous download to {output_path} completed.")
            return output_path
        
        else:
            # Handle batch job download
            file_name = f"{file_name_base}_batch.nc"
            output_path = os.path.join(output_dir, file_name)
            
            logger.info(f'Starting batch job to download data to {output_path}')
            job = datacube.execute_batch(
                output_path,
                title=f"TerraVision - Download for {os.path.basename(shape_file_path)}"
            )
            logger.info(f"Batch job {job.job_id} for {file_name} started successfully.")
            return job.job_id

    except Exception as e:
        logger.error(f"An error occurred during the download process: {e}")
        raise e