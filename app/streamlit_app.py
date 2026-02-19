import os
import sys
import tempfile
import uuid
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st

# cmd to initialize: uv run streamlit app/streamlit_app.py


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.ATES.PRA.PRA_Buhler_OBIA import OBIAPRADelineation


def _get_dem_prefix(dem_path):
    """Extract DEM prefix from path (part before first underscore)"""
    if not dem_path:
        return None
    dem_filename = Path(dem_path).stem
    return dem_filename.split('_')[0] if '_' in dem_filename else dem_filename


def _get_cached_files_for_dem(dem_prefix):
    """Get list of all cached tif files for a given DEM prefix"""
    cache_dir = REPO_ROOT / "cache" / "ATES" / "PRA"
    if not cache_dir.exists():
        return []
    
    tif_files = []
    for file in cache_dir.glob(f"{dem_prefix}__*.tif"):
        stat = file.stat()
        tif_files.append({
            'name': file.name,
            'path': str(file),
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime)
        })
    
    # Sort by modification time, newest first
    tif_files.sort(key=lambda x: x['modified'], reverse=True)
    return tif_files


def _get_latest_tif(dem_prefix):
    """Get the most recently modified tif file for a DEM"""
    files = _get_cached_files_for_dem(dem_prefix)
    return files[0] if files else None


def _save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    temp_dir = Path(tempfile.gettempdir()) / "autoates_streamlit"
    temp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix or ".tif"
    filename = f"{Path(uploaded_file.name).stem}_{uuid.uuid4().hex}{suffix}"
    file_path = temp_dir / filename

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


def _resolve_path(text_value, uploaded_file):
    uploaded_path = _save_uploaded_file(uploaded_file)
    if uploaded_path:
        return uploaded_path

    text_value = (text_value or "").strip()
    if not text_value:
        return None

    return text_value


def _validate_path(path_value, label):
    if not path_value:
        st.error(f"{label} is required.")
        return False

    if not Path(path_value).exists():
        st.error(f"{label} does not exist: {path_value}")
        return False

    return True


def _run_custom_pra(
    obia,
    slope_min,
    slope_max,
    ruggedness_limit,
    plan_curvature_limit,
    min_area_m2,
    cluster_tolerance_cells,
):
    """Wrapper function to call the OBIA delineate_pra_custom method"""
    return obia.delineate_pra_custom(
        slope_min=slope_min,
        slope_max=slope_max,
        ruggedness_limit=ruggedness_limit,
        plan_curvature_limit=plan_curvature_limit,
        min_area_m2=min_area_m2,
        cluster_tolerance_cells=cluster_tolerance_cells,
    )


st.set_page_config(page_title="AutoATES PRA", layout="wide")

# Create main layout with two columns
left_col, right_col = st.columns([2, 1])

# ============================================================================
# LEFT COLUMN - Main Interface
# ============================================================================
with left_col:
    st.title("AutoATES PRA Generator")
    st.write(
        "Select your DEM and forest cover rasters, then choose a PRA scenario to run."
    )

    with st.expander("Input rasters", expanded=True):
        dem_path_text = st.text_input("DEM GeoTIFF path")
        dem_upload = st.file_uploader("Or select DEM .tif", type=["tif", "tiff"])

        forest_path_text = st.text_input("Forest cover GeoTIFF path (optional)")
        forest_upload = st.file_uploader(
            "Or select forest cover .tif", type=["tif", "tiff"]
        )

    st.caption(r"Outputs will be saved to: {REPO_ROOT}\cache\ATES\PRA".format(REPO_ROOT=REPO_ROOT))

    col1, col2, col3 = st.columns(3)

    if "show_custom" not in st.session_state:
        st.session_state.show_custom = False

    with col1:
        frequent_clicked = st.button("Generate Frequent PRA")

    with col2:
        extreme_clicked = st.button("Generate Extreme PRA")

    with col3:
        custom_clicked = st.button("Generate Custom PRA")

    if custom_clicked:
        st.session_state.show_custom = True

    if frequent_clicked or extreme_clicked:
        dem_path = _resolve_path(dem_path_text, dem_upload)
        forest_path = _resolve_path(forest_path_text, forest_upload)

        if _validate_path(dem_path, "DEM path"):
            if forest_path and not Path(forest_path).exists():
                st.error(f"Forest cover path does not exist: {forest_path}")
            else:
                with st.spinner("Running PRA delineation..."):
                    obia = OBIAPRADelineation(
                        dem_path,
                        forest_path=forest_path or None,
                    )

                    if frequent_clicked:
                        output = obia.delineate_pra_frequent()
                        st.success("Frequent PRA generated.")
                        output_file = "PRA_frequent_scenario.tif"
                    else:
                        output = obia.delineate_pra_extreme()
                        st.success("Extreme PRA generated.")
                        output_file = "PRA_extreme_scenario.tif"

                    st.write("Output saved to:")
                    st.write(os.path.join(obia.output_dir, output_file))
                    
                    # Trigger rerun to update file list
                    st.rerun()

    if st.session_state.show_custom:
        st.subheader("Custom PRA parameters")
        slope_min = st.number_input("Minimum slope (degrees)", value=30.0, step=1.0)
        slope_max = st.number_input("Maximum slope (degrees)", value=60.0, step=1.0)
        ruggedness_limit = st.number_input(
            "Ruggedness limit", value=0.06, step=0.01, format="%.3f"
        )
        plan_curvature_limit = st.number_input(
            "Plan curvature limit", value=6.0, step=0.5, format="%.2f"
        )
        min_area_m2 = st.number_input(
            "Small areas limit (m2)", value=500, step=10
        )
        cluster_tolerance_cells = st.number_input(
            "Cluster tolerance (cells)", value=1, step=1
        )

        go_clicked = st.button("Go")

        if go_clicked:
            dem_path = _resolve_path(dem_path_text, dem_upload)
            forest_path = _resolve_path(forest_path_text, forest_upload)

            if _validate_path(dem_path, "DEM path"):
                if forest_path and not Path(forest_path).exists():
                    st.error(f"Forest cover path does not exist: {forest_path}")
                else:
                    with st.spinner("Running custom PRA delineation..."):
                        obia = OBIAPRADelineation(
                            dem_path,
                            forest_path=forest_path or None,
                        )
                        output_path = _run_custom_pra(
                            obia,
                            slope_min,
                            slope_max,
                            ruggedness_limit,
                            plan_curvature_limit,
                            min_area_m2,
                            int(cluster_tolerance_cells),
                        )

                    st.success("Custom PRA generated.")
                    st.write("Output saved to:")
                    st.write(output_path)
                    
                    # Trigger rerun to update file list
                    st.rerun()

# ============================================================================
# RIGHT COLUMN - File Monitoring
# ============================================================================
with right_col:
    st.subheader("Cache Monitor")
    
    # Get current DEM path and extract prefix
    dem_path = _resolve_path(dem_path_text, dem_upload)
    dem_prefix = _get_dem_prefix(dem_path)
    
    if dem_prefix:
        st.caption(f"Monitoring files for DEM: **{dem_prefix}**")
        
        # Top section: Latest TIF preview
        st.markdown("### Latest Output")
        latest_file = _get_latest_tif(dem_prefix)
        
        if latest_file:
            with st.container(border=True):
                st.text(latest_file['name'])
                st.caption(f"Modified: {latest_file['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.caption(f"Size: {latest_file['size'] / 1024:.1f} KB")
                
                # Try to display the raster as an image
                try:
                    import rasterio
                    from rasterio.plot import show
                    import matplotlib.pyplot as plt
                    
                    with rasterio.open(latest_file['path']) as src:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        show(src, ax=ax, cmap='terrain')
                        ax.set_title(latest_file['name'], fontsize=10)
                        st.pyplot(fig)
                        plt.close()
                except Exception as e:
                    st.info("Preview not available")
        else:
            st.info("No files created yet")
        
        st.markdown("---")
        
        # Bottom section: List of all cached files
        st.markdown("### All Cached Files")
        cached_files = _get_cached_files_for_dem(dem_prefix)
        
        if cached_files:
            with st.container(height=400, border=True):
                for idx, file_info in enumerate(cached_files):
                    with st.expander(f"📄 {file_info['name']}", expanded=False):
                        st.text(f"Path: {file_info['path']}")
                        st.text(f"Size: {file_info['size'] / 1024:.1f} KB")
                        st.text(f"Modified: {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.caption(f"Total: {len(cached_files)} files")
        else:
            st.info("No cached files found for this DEM")
    else:
        st.info("Enter a DEM path to monitor cached files")

