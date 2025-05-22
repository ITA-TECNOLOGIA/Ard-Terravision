# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import os
import numpy as np
import streamlit as st
from PipelineConfig import PipelineConfig

# Page config and title
st.set_page_config(page_title="ARD-TERRAVISION", layout="centered")

# Display logos side by side
logo_paths = [
    ("figures/Terravision_Logo_Official.png", 300),
    ("figures/ITA_Logo.png", 300),
]
cols = st.columns(len(logo_paths))
for col, (path, width) in zip(cols, logo_paths):
    col.image(path, width=width)

st.title("ARD-TERRAVISION")
st.write("Select and run a pipeline configuration interactively.")

# Session state flag
if 'run_pipeline' not in st.session_state:
    st.session_state.run_pipeline = False

# Helper functions
def list_pipelines(folder="pipelines", ext=".json"):
    if os.path.isdir(folder):
        return sorted(f for f in os.listdir(folder) if f.endswith(ext))
    return []

def normalize_image(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else arr

def display_stage(stage_name: str, results: list):
    st.header(stage_name)
    for alg_name, imgs in results:
        if imgs:
            for i, img in enumerate(imgs, 1):
                st.image(img, caption=f"{alg_name} output {i}", use_container_width=True)
            st.success(f"{alg_name} finished.")
        else:
            st.info(f"{alg_name} produced no image.")

# Sidebar: pipeline selection
pipeline_files = list_pipelines()
if not pipeline_files:
    st.error("No pipeline JSON files found in 'pipelines'.")
    st.stop()
selected = st.sidebar.selectbox("Choose a pipeline:", pipeline_files)
if st.sidebar.button("Run Pipeline"):
    st.session_state.run_pipeline = True

# Execution block
if st.session_state.run_pipeline:
    st.info(f"Loading '{selected}'...")
    cfg = PipelineConfig.from_json(os.path.join("pipelines", selected))
    st.success("Config loaded.")

    # L1: Input
    st.header("Stage L1: Input Loading")
    l1_data = cfg.run_l1()
    st.write(f"Loaded input: {l1_data}")
    try:
        img = getattr(l1_data, 'get_debug_image', lambda: None)()
        if img is not None:
            st.subheader("Input Debug Image")
            st.image(img, caption="Debug image", use_container_width=True)
        else:
            st.info("No debug image for input.")
    except Exception as e:
        st.error(f"Input debug failed: {e}")
    st.success("L1 complete.")

    # L2 and L3: Processing
    l2 = []
    for alg in cfg.l2_algorithms:
        name = alg.__class__.__name__
        st.write(f"Running L2: {name}...")
        res = alg.process_data(l1_data)
        imgs = []
        if res is not None:
            for out in (res if isinstance(res, (list, tuple)) else [res]):
                pil = getattr(out, 'debug_image', None)
                if pil is not None:
                    arr = np.array(pil)
                    imgs.append(normalize_image(arr))
        l2.append((name, imgs))
    display_stage("Stage L2: Processing Algorithms", l2)

    l3 = []
    for alg in cfg.l3_algorithms:
        name = alg.__class__.__name__
        st.write(f"Running L3: {name}...")
        res = alg.process_data(l1_data)
        imgs = []
        if res is not None:
            for out in (res if isinstance(res, (list, tuple)) else [res]):
                pil = getattr(out, 'debug_image', None)
                if pil is not None:
                    arr = np.array(pil)
                    imgs.append(normalize_image(arr))
        l3.append((name, imgs))
    display_stage("Stage L3: Generating Results", l3)

    # L4: Fusion
    st.header("Stage L4: Final Fusion")
    try:
        final = cfg.run_l4(l1_data)
        st.write(final)
        st.success("L4 fusion completed.")
    except Exception as e:
        st.error(f"L4 fusion failed: {e}")

    st.balloons()
    st.write("### Pipeline finished!")
    st.session_state.run_pipeline = False
