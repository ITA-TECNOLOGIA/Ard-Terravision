<p align="left">
  <img src="figures/Terravision_Logo_Official.png" alt="Terravision Logo" height="50">
  <img src="figures/ITA_Logo.png" alt="ITA Logo" height="50">
</p>

This repository contains scripts, utilities, and a modular pipeline to produce **Analysis Ready Data** (ARD) from satellite and airborne imagery. The main goal is to streamline the transition from raw data (**L1**) through various pre-processing and processing stages (**L2**, **L3**, **L4**) to reach advanced insights, such as semantic labeling, detection tasks, and final geospatial analyses.

The project is under active development at ITA, integrating multiple data sources and specialized algorithms in a unified, extensible framework.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Description](#description)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Testing](#testing)
6. [Pipeline Configuration (L1, L2, L3, L4)](#pipeline-configuration-l1-l2-l3-l4)

   * [Using JSON Config Files](#using-json-config-files)
   * [L1 (Input Data Layer)](#l1-input-data-layer)
   * [L2 (Pre-processing Layer)](#l2-pre-processing-layer)
   * [L3 (Analysis & Inference Layer)](#l3-analysis--inference-layer)
   * [L4 (Final Output Layer)](#l4-final-output-layer)
7. [How to Add Your Own Algorithms](#how-to-add-your-own-algorithms)
8. [Data Sources](#data-sources)

   * [Satellite Data](#satellite-data)
   * [Airborne Data](#airborne-data)
9. [Roadmap](#roadmap)
10. [Authors and Acknowledgment](#authors-and-acknowledgment)
11. [License](#license)
12. [Project Status](#project-status)

---

## Architecture Overview

Below is a high-level schematic of the pipeline:

```
                      ┌─────────────────┐
                      │    IMAGE DATA   │
                      │  + NUMERICAL    │
                      │      DATA       │
                      └─────────────────┘
                               |
                            ┌───────┐
                            │  L1   │     (Data Ingestion Layer)
                            └───────┘
                               |
                               ▼
                            ┌───────┐
                            │  L2   │     (Pre-processing and Corrections)
                            └───────┘
                               |
                               ▼
                            ┌───────┐
                            │  L3   │     (Feature Extraction / AI Inference)
                            └───────┘
                               |
                               ▼
                            ┌───────┐
                            │  L4   │     (Final Outputs / Post-processing)
                            └───────┘
```

* **L1** handles data ingestion from various sources (e.g. Sentinel 2, airborne imagery).
* **L2** covers tasks like atmospheric corrections, DEM integration, and orthorectification.
* **L3** includes advanced algorithms like semantic captioning, object detection, and label classification.
* **L4** collates outputs from L3 to produce final results or feed them into subsequent workflows.

---

## Description

This repository provides a collection of scripts and utilities designed to create **Analysis Ready Data (ARD)** primarily from satellite data (Sentinel 2) and airborne data. The tools are modular and can be easily extended by adding or customizing different *layers* (L1, L2, L3, and L4).

Key highlights:

* Modular *Layered* design for flexible data flow.
* Extensible base classes for each layer.
* Support for advanced deep-learning frameworks with GPU acceleration (CUDA).
* Example JSON configs for inspiration under `pipelines/`.

---

## Installation

1. **Create a Python environment (recommended conda)**:

   ```bash
   conda create -n terravision_ard python=3.10
   conda activate terravision_ard
   ```

2. **Create a `.env` file** at the project root with your desired device setting:

   ```dotenv
   DEVICE=cuda:1
   ```

   > ⚙️ The pipeline will load this environment variable (using `python-dotenv`) to configure GPU usage for all algorithms.

3. **Install dependencies**:

   ```bash
   pip install python-dotenv
   ```

4. **Install PyTorch (with CUDA 12.6 support)**:

   Since the environment has been tested on **salas.ita.es** with `CUDA 12.6`:

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

5. **Install the core dependencies**:

   ```bash
   pip install -r src/main/requirements.txt
   ```

6. **Install Grounded-SAM-2 and dependencies**:

   ```bash
   pip install -e src/main/python/L3/ObjectDetection/Grounded-SAM-2
   pip install src/main/python/L3/ObjectDetection/Grounded-SAM-2[demo]
   pip install --no-build-isolation -e src/main/python/L3/ObjectDetection/Grounded-SAM-2/grounding_dino
   ```

7. **Download and move pretrained checkpoints**:

   ```bash
   bash src/main/python/L3/ObjectDetection/Grounded-SAM-2/checkpoints/download_ckpts.sh
   mv sam2.1_hiera_*.pt checkpoints/
   ```

---

## Usage

1. **Main entry point**:

   ```bash
   python src/main/python/main.py --config pipelines/satellite_example.json
   ```

   The script will load `.env` variables (including `DEVICE`) and set the CUDA device accordingly.

2. **Pipeline configuration**:
   Place your JSON under `pipelines/`, e.g.:

   * `pipelines/satellite_example.json`
   * `pipelines/airborne_example.json`

---

## Testing

All code changes should be covered by unit tests. Tests live under `src/test/python` and are discovered automatically by Python's `unittest` framework. To run tests, use:

```bash
python -m unittest discover -v -s src/test/python
```

---

## Pipeline Configuration (L1, L2, L3, L4)

Instead of subclassing, pipelines are now defined via JSON. See `pipelines/satellite_example.json` and `pipelines/airborne_example.json` for samples.

### Using JSON Config Files

Each config is a JSON object with four keys:

* `l1_input`: object with `type` (e.g. "Satellite") and `params` for its constructor.
* `l2_algorithms`: array of `{ type: string, params: {...} }`.
* `l3_algorithms`: array of `{ type: string, params: {...} }`.
* `l4_algorithm`: object with `type` and `params`.

---

## How to Add Your Own Algorithms

1. Create a subfolder under the appropriate layer (e.g. `src/main/python/L2/MyAlg/`).
2. Add your `.py` implementing the layer’s abstract base.
3. No Git submodules—copy any third-party code in that folder.
4. Add your `type` and module path to `PipelineConfig.CLASS_REGISTRY`.

---

## Data Sources

### Satellite Data

Sentinel 2 data lives on Salas: `/datassd/proyectos/terravision/terravision_satellite/`.

### Airborne Data

DIMAP images: `/datassd/proyectos/terravision/terravision_airborne/`.

---

## Authors and Acknowledgment

* **Sergio Gracia** ([sgracia@ita.es](mailto:sgracia@ita.es))
* **Álvaro Navarro** ([anavarroa@ita.es](mailto:anavarroa@ita.es))
* **Rafael del Hoyo** ([rdelhoyo@ita.es](mailto:rdelhoyo@ita.es))
* **Carlos Marañes** ([cmaranes@ita.es](mailto:cmaranes@ita.es))

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file in the root directory for details.

---

## Project Status

Under active development.
