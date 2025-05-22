# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import argparse
from logger import logger
from PipelineConfig import PipelineConfig

def main(config_path: str) -> None:
    logger.info("Starting pipeline with config: %s", config_path)
    
    pipeline = PipelineConfig.from_json(config_path)

    result = pipeline.run()
    logger.info("Pipeline completed successfully")
    print("Output:", result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Terravision Pipeline"
    )
    parser.add_argument(
        "-c", "--config",
        default="pipelines/satellite_example.json",
        help="Path to pipeline JSON configuration"
    )
    args = parser.parse_args()
    main(args.config)
