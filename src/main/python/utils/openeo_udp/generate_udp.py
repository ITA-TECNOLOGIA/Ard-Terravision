# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# All rights reserved 
# --------------------------------------------------------------------------------
import json
import argparse
from pathlib import Path

import openeo
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict

def input_datacube(connection, spatial_extent, temporal_extent, bands):
    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=bands
    )
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["SCL"]
    )
    mask = scl.process(
        "to_scl_dilation_mask", 
        data=scl
    )
    
    masked_cube = s2_cube.mask(mask) 
    return masked_cube

def bsi_workflow(connection, spatial_extent, temporal_extent):
    bands = ["B02", "B04", "B08", "B11"]
    input_cube = input_datacube(connection, spatial_extent, temporal_extent, bands)
    b02 = input_cube.band("B02")
    b04 = input_cube.band("B04")
    b08 = input_cube.band("B08")
    b11 = input_cube.band("B11")
    bsi = ((b11 + b04) - (b02 + b08)) / ((b11 + b04) + (b02 + b08))
    return bsi

def amwi_workflow(connection, spatial_extent, temporal_extent):
    bands = ["B02", "B04"]
    input_cube = input_datacube(connection, spatial_extent, temporal_extent, bands)
    b02 = input_cube.band("B02")
    b04 = input_cube.band("B04")
    amwi = (b04 - b02) / (b04 + b02)
    return amwi

def nddi_workflow(connection, spatial_extent, temporal_extent):
    bands = ['B02', 'B12']
    input_cube = input_datacube(connection, spatial_extent, temporal_extent, bands)
    b02 = input_cube.band("B02")
    b12 = input_cube.band("B12")
    nddi = (b12 - b02) / (b12 + b02)
    return nddi

def generate(indices="bsi"):
    connection = openeo.connect("openeofed.dataspace.copernicus.eu")

    spatial_extent = Parameter.spatial_extent(
        name="spatial_extent", 
        description="Limits the data to process to the specified bounding box or polygons.\\n\\nFor raster data, the process loads the pixel into the data cube if the point at the pixel center intersects with the bounding box or any of the polygons (as defined in the Simple Features standard by the OGC).\\nFor vector data, the process loads the geometry into the data cube if the geometry is fully within the bounding box or any of the polygons (as defined in the Simple Features standard by the OGC). Empty geometries may only be in the data cube if no spatial extent has been provided.\\n\\nEmpty geometries are ignored.\\nSet this parameter to null to set no limit for the spatial extent."
        )
    
    temporal_extent = Parameter.temporal_interval(
        name="temporal_extent", 
        description="Temporal extent specified as two-element array with start and end date/date-time."
        )
    about = json.loads(Path("about.json").read_text())
    try:
        datacube = globals()[indices + "_workflow"](connection, spatial_extent, temporal_extent)
    except KeyError:
        raise ValueError(f"Invalid indices specified: {indices}")
    
    return build_process_dict(
        process_graph=datacube,
        process_id=indices.upper(),
        description=about[indices]["description"],
        summary=about[indices]["summary"],
        parameters=[
            spatial_extent,
            temporal_extent
        ]
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate openEO UDP for specified indices.")
    parser.add_argument("--indices", "-i", type=str, default="nddi",
                        help="Specify the indices to generate the corresponding workflow (default: nddi).")
    args = parser.parse_args()
    indices = args.indices
    with open(f"{indices}.json", "w") as f:
        json.dump(generate(indices=indices), f, indent=2)