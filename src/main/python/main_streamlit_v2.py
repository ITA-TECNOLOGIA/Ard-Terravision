# --------------------------------------------------------------------------------
# ARD - TERRAVISION • Streamlit UI Revamp
# Version: 1.8 (Using authenticate_oidc_client_credentials)
# --------------------------------------------------------------------------------

import os
import json
import time
import numpy as np
import streamlit as st
import openeo
from PipelineConfig import PipelineConfig
from datetime import datetime
from utils.openeo_downloader import download_data
from dotenv import load_dotenv

# Load environment variables (ONLY for paths)
load_dotenv()
SHAPEFILES_PATH = os.getenv("SHAPEFILES_PATH")
OPENEO_DOWNLOADS_PATH = os.getenv("OPENEO_DOWNLOADS_PATH")

# -----------------------------
# Page config & lightweight CSS
# -----------------------------
st.set_page_config(
    page_title="ARD-TERRAVISION V2",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
/* tighter top padding and cleaner container look */
.main > div { padding-top: 2rem; }
.block-container { padding-top: 1rem; }
/* glassy cards */
div[data-testid="stHorizontalBlock"] > div, .stTabs [data-baseweb="tab-list"] {
    backdrop-filter: blur(6px);
}
.stAlert { border-radius: 14px; }
.stStatus { border-radius: 14px; }
img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def list_pipelines(folder: str = "pipelines", ext: str = ".json"):
    if os.path.isdir(folder):
        return sorted(f for f in os.listdir(folder) if f.endswith(ext))
    return []


def get_first_datacube_path(config_data):
    if "l1_input" in config_data:
        return config_data["l1_input"].get("params", {}).get("datacube_path", "Not found")
    if "l1_inputs" in config_data and config_data["l1_inputs"]:
        return config_data["l1_inputs"][0].get("params", {}).get("datacube_path", "Not found")
    return "Not found"


def override_datacube_path(config_data, new_path):
    if "l1_input" in config_data:
        config_data["l1_input"]["params"]["datacube_path"] = new_path
        return True
    if "l1_inputs" in config_data and config_data["l1_inputs"]:
        config_data["l1_inputs"][0]["params"]["datacube_path"] = new_path
        return True
    return False




def display_l3_results(l3_results: list):
    """Display L3 results from the new pipeline design."""
    if not l3_results:
        st.info("No L3 results generated.")
        return

    for idx, result in enumerate(l3_results):
        result_type = getattr(result, 'result_type', 'unknown')
        time_indices = getattr(result, 'time_indices', [])
        time_str = f" (t={time_indices})" if time_indices else ""

        with st.expander(f"🧩 L3 Result {idx+1}: {result_type}{time_str}", expanded=True):
            debug_img = getattr(result, 'debug_image', None)
            if debug_img:
                st.markdown("**Visual output**")
                st.image(debug_img, use_container_width=True, caption=f"L3 Result {idx+1}")
                st.success("Completed successfully.")
            else:
                st.warning("No visual output produced.")

            alg_results = getattr(result, 'algorithm_results', None)
            if alg_results is not None:
                display_algorithm_details([alg_results])

def display_algorithm_details(items):
    """
    Render extra details stored inside out.algorithm_results.
    Handles both L3_result objects and raw algorithm result objects.
    No nested expanders.
    """
    if not items:
        return

    for out_idx, item in enumerate(items, start=1):
        alg_results = getattr(item, "algorithm_results", item)
        if alg_results is None or (
            isinstance(alg_results, (list, tuple)) and len(alg_results) == 0
        ):
            continue
        if not isinstance(alg_results, (list, tuple)):
            alg_results = [alg_results]

        for fr_idx, fr in enumerate(alg_results, start=1):
            sam_scores = getattr(fr, "sam_scores", None)
            if sam_scores is None:
                continue
            st.markdown(f"### Details • Output {out_idx} • Frame {fr_idx}")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Input / Detections",
                "Florence raw text",
                "Florence parsed output",
                "SAM scores"
            ])

            with tab1:
                kwargs = getattr(fr, "kwargs", None)
                if kwargs is not None:
                    st.markdown("**Input kwargs**")
                    st.json(kwargs)

                detections = getattr(fr, "detections", None)
                sam_scores = getattr(fr, "sam_scores", None)

                if detections:
                    st.markdown("**Detections**")
                    det_rows = []
                    for i, det in enumerate(detections, start=1):
                        score = None
                        if sam_scores is not None and i - 1 < len(sam_scores):
                            score = sam_scores[i - 1]

                        det_rows.append({
                            "idx": i,
                            "class_id": getattr(det, "class_id", None),
                            "confidence": getattr(det, "confidence", None),
                            "sam_score": score,
                            "x": det.bbox.get("x") if getattr(det, "bbox", None) else None,
                            "y": det.bbox.get("y") if getattr(det, "bbox", None) else None,
                            "width": det.bbox.get("width") if getattr(det, "bbox", None) else None,
                            "height": det.bbox.get("height") if getattr(det, "bbox", None) else None,
                        })
                    st.dataframe(det_rows, use_container_width=True)
                else:
                    st.info("No detections available.")

            with tab2:
                florence_raw_text = getattr(fr, "florence_raw_text", None)
                if florence_raw_text:
                    st.markdown("**Raw text returned by Florence-2**")
                    st.code(florence_raw_text, language="text")
                else:
                    st.info("No Florence raw text available.")

            with tab3:
                florence_parsed_output = getattr(fr, "florence_parsed_output", None)
                if florence_parsed_output is not None:
                    st.markdown("**Parsed Florence-2 output**")
                    st.json(florence_parsed_output)
                else:
                    st.info("No Florence parsed output available.")

            with tab4:
                sam_scores = getattr(fr, "sam_scores", None)
                if sam_scores is not None:
                    st.write(sam_scores)
                else:
                    st.info("No SAM scores available.")

            st.markdown("---")
                    
# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    # --- [ Logo and Title Section ] ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.image("figures/Terravision_Logo_Official.png", use_container_width=True)
    with col_b:
        st.image("figures/ITA_Logo.png", use_container_width=True)
    st.markdown("<h2 style='text-align:center;'>ARD-TERRAVISION V2</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray; margin-bottom: 20px;'>Interactive pipeline runner</p>", unsafe_allow_html=True)
    st.divider()

    # --- [ Pipeline Configuration Section ] ---
    st.header("⚙️ Pipeline Configuration")
    pipeline_files = list_pipelines()
    if not pipeline_files:
        st.error("No pipeline JSON files found in 'pipelines'.")
        st.stop()
    selected = st.selectbox("Choose a pipeline", pipeline_files, index=0)

    if selected:
        try:
            pipeline_path = os.path.join("pipelines", selected)
            with open(pipeline_path, "r") as f:
                config_data = json.load(f)
            datacube_path = get_first_datacube_path(config_data)
            st.info(f"Datacube in this pipeline: `{datacube_path}`")
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            st.warning("Could not read datacube path from the selected pipeline.")

    openeo_files = ["Default"] + list_pipelines(folder=OPENEO_DOWNLOADS_PATH, ext=".nc")
    selected_openeo_input = st.selectbox("Choose an OpenEO input (optional)", openeo_files, index=0)
    autorun = st.toggle("Auto-run on selection", value=False)
    run_btn = st.button("▶️ Run Pipeline", use_container_width=True)
    st.divider()

    # --- [ Manual Upload Section ] ---
    st.header("📂 Manual Upload")
    uploaded_nc_file = st.file_uploader("Upload .nc Datacube", type=["nc"])
    uploaded_json_file = st.file_uploader("Upload .json Pipeline", type=["json"])
    st.caption("Note: Uploading a `.nc` file will override the datacube path from any uploaded `.json`.")
    if uploaded_json_file:
        try:
            uploaded_json_file.seek(0)
            config_data = json.load(uploaded_json_file)
            datacube_path = get_first_datacube_path(config_data)
            st.info(f"Datacube in JSON: `{datacube_path}`")
        except (json.JSONDecodeError, KeyError):
            st.warning("Could not read datacube path from the uploaded JSON.")
    run_uploaded_btn = st.button("▶️ Run Uploaded Pipeline", use_container_width=True)
    st.divider()

    # -----------------------------------------------------------------
    # OpenEO Login Section
    # -----------------------------------------------------------------
    st.header("🔐 OpenEO Login")
    
    # Initialize session state for connection
    if "openeo_connection" not in st.session_state:
        st.session_state.openeo_connection = None
    if "openeo_job_id" not in st.session_state:
        st.session_state.openeo_job_id = None
    if "downloaded_file_path" not in st.session_state:
        st.session_state.downloaded_file_path = None

    if st.session_state.openeo_connection:
        st.success("You are connected to OpenEO.")
        if st.button("Logout", use_container_width=True):
            st.session_state.openeo_connection = None
            st.rerun()
    else:
        st.info("Login with your OIDC Client Credentials from Copernicus Dashboard.")
        client_id = st.text_input("Client ID")
        client_secret = st.text_input("Client Secret", type="password")

        if st.button("Login", use_container_width=True):
            if client_id and client_secret:
                try:
                    with st.spinner("Authenticating..."):
                        
                        # 1. Connect to OpenEO
                        connection = openeo.connect("openeo.dataspace.copernicus.eu")
                        
                        # 2. Authenticate with OIDC
                        connection.authenticate_oidc_client_credentials(
                            client_id=client_id,
                            client_secret=client_secret
                        )

                        # 3. Test connection
                        connection.list_jobs(limit=1) 
                        
                        # 4. Save the successful connection
                        st.session_state.openeo_connection = connection
                        st.success("Login Successful!")
                        time.sleep(1) 
                        st.rerun()
                except Exception as e:
                    st.error(f"Login Failed: {e}")
                    st.session_state.openeo_connection = None
            else:
                st.warning("Please enter both Client ID and Client Secret.")
    st.divider() 

    # OpenEO Data Download Section
    st.header("🛰️ OpenEO Data Download")
    try:
        shapefiles = [f for f in os.listdir(SHAPEFILES_PATH) if f.endswith('.shp')]
    except FileNotFoundError:
        shapefiles = []
    if not shapefiles:
        st.warning(f"No shapefiles found in the directory: {SHAPEFILES_PATH}")
        st.stop()
    selected_shapefile = st.selectbox("Choose a Shapefile", shapefiles)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2025, 10, 12))
    with col2:
        end_date = st.date_input("End Date", datetime(2025, 10, 30))
    sync_download = st.checkbox("Download to my computer")
    download_btn = st.button("📥 Download Data", use_container_width=True)

    # Download Button Logic
    if download_btn:
        
        # 1. Check if user is logged in
        if not st.session_state.openeo_connection:
            st.error("You must be logged in to download data. Please login above.")
        
        # 2. Check if shapefile is selected
        elif not selected_shapefile:
            st.warning("Please select a shapefile to start the download.")
        
        # 3. If both checks pass, run the download
        else:
            try:
                # Get the connection we already saved
                connection = st.session_state.openeo_connection
                shapefile_path = os.path.join(SHAPEFILES_PATH, selected_shapefile)

                if sync_download:
                    with st.spinner("Downloading data synchronously... This might take a while."):
                        file_path = download_data(connection, shapefile_path, start_date, end_date, synchronous=True)
                        st.session_state.downloaded_file_path = file_path
                        st.success("File ready for download!")
                else:
                    with st.spinner("Initiating download... This may take a moment."):
                        job_id = download_data(connection, shapefile_path, start_date, end_date, synchronous=False)
                        st.session_state.openeo_job_id = job_id
                        st.success(f"Download job started with ID: {job_id}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                
    if st.session_state.downloaded_file_path:
        with open(st.session_state.downloaded_file_path, "rb") as fp:
            st.download_button(
                label="📥 Click to Download",
                data=fp,
                file_name=os.path.basename(st.session_state.downloaded_file_path),
                mime="application/x-netcdf",
                use_container_width=True
            )

# --- [ Session state for pipeline ] ---
if "run_pipeline" not in st.session_state:
    st.session_state.run_pipeline = False
if autorun:
    st.session_state.run_pipeline = True
elif run_btn:
    st.session_state.run_pipeline = True
elif run_uploaded_btn:
    st.session_state.run_pipeline = True

# -----------------------------
# Main content area
# -----------------------------
st.header("Pipeline Runner", anchor=False) 
st.write(
    "Use the tabs to follow each stage. All steps log their status, "
    "and images are shown in expandable cards."
)
tabs = st.tabs(["L1 • Input", "L2 • Processing", "L3 • Results", "L4 • Fusion", "Logs"])
log_lines = []
def log(msg: str):
    log_lines.append(msg)

# Pipeline Runner Logic
if st.session_state.run_pipeline:
    if uploaded_json_file is not None:
        uploaded_json_file.seek(0)
        config_data = json.load(uploaded_json_file)
        selected_pipeline_name = uploaded_json_file.name
        log(f"[INIT] Loaded uploaded config {selected_pipeline_name}")
    else:
        pipeline_path = os.path.join("pipelines", selected)
        with open(pipeline_path, "r") as f:
            config_data = json.load(f)
        selected_pipeline_name = selected
        log(f"[INIT] Loaded config {selected_pipeline_name}")

    with st.status(f"Loading configuration: **{selected_pipeline_name}**", expanded=True) as s:
        if uploaded_nc_file is not None:
            save_path = os.path.join(OPENEO_DOWNLOADS_PATH, uploaded_nc_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_nc_file.getbuffer())
            try:
                if override_datacube_path(config_data, save_path):
                    st.write(f"✔️ Overriding input with uploaded .nc file: **{uploaded_nc_file.name}**")
                    log(f"[INIT] Overriding L1 input with uploaded file {save_path}")
                else:
                    st.warning("Could not find L1 input path to override.")
                    log("[INIT] Failed to override L1 input path.")
            except KeyError:
                st.warning("Could not find 'l1_input.params.datacube_path' to override.")
                log("[INIT] Failed to override L1 input path.")
        elif selected_openeo_input != "Default":
            openeo_path = os.path.join(OPENEO_DOWNLOADS_PATH, selected_openeo_input)
            try:
                if override_datacube_path(config_data, openeo_path):
                    st.write(f"✔️ Overriding input with: **{selected_openeo_input}**")
                    log(f"[INIT] Overriding L1 input with {openeo_path}")
                else:
                    st.warning("Could not find L1 input path to override.")
                    log("[INIT] Failed to override L1 input path.")
            except KeyError:
                st.warning("Could not find 'l1_input.params.datacube_path' to override.")
                log("[INIT] Failed to override L1 input path.")

        cfg = PipelineConfig.from_dict(config_data)
        st.write("✔️ Config loaded.")
        time.sleep(0.2)
        s.update(label="Configuration ready", state="complete")

    with tabs[0]:
        with st.status("Stage L1: Input Loading", expanded=True) as s:
            try:
                l1_data = cfg.run_l1()
                st.write("**Loaded input:**")
                
                st.code(str(l1_data), language="text") 
                
                log("[L1] Input loaded")
                try:
                    items = l1_data if isinstance(l1_data, list) else [l1_data]
                    for i, item in enumerate(items):
                        img = getattr(item, 'get_debug_image', lambda: None)()
                        if img is not None:
                            caption = f"L1 Debug Image #{i+1}" if len(items) > 1 else "Input Debug Image"
                            st.image(img, caption=caption, use_container_width=True)
                        elif len(items) == 1:
                            st.info("No debug image available for input.")
                except Exception as e:
                    st.error(f"Input debug failed: {e}")
                    log(f"[L1] Debug image error: {e}")
                s.update(label="L1 complete", state="complete")
            except Exception as e:
                st.error(f"L1 failed: {e}")
                log(f"[L1] Failed: {e}")
                s.update(label="L1 failed", state="error")

    

    # Stage L2: Processing Algorithms
    with tabs[1]:
        with st.status("Stage L2: Processing Algorithms", expanded=True) as s:
            try:
                l2_output = cfg.run_l2(l1_data)
                if l2_output:
                    st.write("**L2 Output:**")
                    st.code(f"Processed bands: {list(l2_output.processed_band_info.keys())}", language="text")
                    if l2_output.debug_image:
                        st.image(l2_output.debug_image, caption="L2 Debug Image", use_container_width=True)
                    st.success("L2 processing completed.")
                    log("[L2] Processing complete")
                else:
                    st.info("No L2 algorithms configured.")
                    l2_output = None
                s.update(label="L2 complete", state="complete")
            except Exception as e:
                st.error(f"L2 processing failed: {e}")
                log(f"[L2] Failed: {e}")
                s.update(label="L2 failed", state="error")
                l2_output = None

    # Stage L3: Generating Results
    with tabs[2]:
        with st.status("Stage L3: Generating Results", expanded=True) as s:
            l2_datacube = l2_output.datacube if l2_output else None
            try:
                l3_results = cfg.run_l3(l1_data, l2_output)
                if l3_results:
                    st.write(f"**L3 Results:** {len(l3_results)} result(s) generated")
                    st.success("L3 processing completed.")
                    log(f"[L3] Generated {len(l3_results)} result(s)")
                else:
                    st.info("No L3 algorithms configured.")
                    l3_results = []
                s.update(label="L3 complete", state="complete")
            except Exception as e:
                st.error(f"L3 processing failed: {e}")
                log(f"[L3] Failed: {e}")
                s.update(label="L3 failed", state="error")
                l3_results = []

        # Display results OUTSIDE the status block to avoid nested expanders
        if l3_results:
            display_l3_results(l3_results)

    with tabs[3]:
        with st.status("Stage L4: Final Fusion", expanded=True) as s:
            try:
                l3_results = l3_results if 'l3_results' in dir() else []
                final = cfg.run_l4(l1_data, l3_results)
                if final is None:
                    st.info("L4 fusion not configured (l4_algorithm is null)")
                    log("[L4] Skipped - no algorithm configured")
                    s.update(label="L4 skipped", state="complete")
                else:
                    st.success("L4 fusion completed.")
                    st.write("**Final Output:**")
                    st.code(str(final), language="text")
                    log("[L4] Fusion completed")
                    s.update(label="L4 complete", state="complete")
            except Exception as e:
                st.error(f"L4 fusion failed: {e}")
                log(f"[L4] Failed: {e}")
                s.update(label="L4 failed", state="error")

    with tabs[4]:
        st.subheader("Run Log", anchor=False)
        if log_lines:
            st.code("\n".join(log_lines), language="text")
        else:
            st.info("No logs captured.")

    st.balloons()
    st.success("Pipeline finished!")
    st.session_state.run_pipeline = False
else:
    st.info("Select a pipeline and click **Run Pipeline** (or enable **Auto-run**).")