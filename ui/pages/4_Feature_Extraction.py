import streamlit as st
from dataclasses import asdict
import yaml
import os
import subprocess
from io import BytesIO

from configs.config_models import VisionFeatureExtractionConfig
from scripts.models.vision.base import ModelName

DEFAULT_CONFIG = VisionFeatureExtractionConfig(
    wsi_dir="", coordinates_dir="", features_dir="", wsi_meta_data_path=""
)

# --- Helper Functions for UI Sections ---


def render_file_uploader_section(current_params: dict) -> dict:
    """Render the config file uploader section."""
    uploaded_file = st.file_uploader(
        "Config File",
        type=["yaml", "yml"],
        help="Upload the config file for the feature extraction.",
    )
    if uploaded_file is not None:
        current_params = yaml.safe_load(uploaded_file)
    return current_params


def render_paths_section(current_params: dict) -> dict:
    """Renders the input fields for path configurations and returns their values."""
    st.markdown("Specify the directories and file paths for your data.")

    current_params["wsi_dir"] = st.text_input(
        "Whole Slide Image Directory",
        value=current_params.get("wsi_dir", DEFAULT_CONFIG.wsi_dir),
        help="Path to the directory containing WSI files.",
    )
    current_params["coordinates_dir"] = st.text_input(
        "Coordinates Directory",
        value=current_params.get("coordinates_dir", DEFAULT_CONFIG.coordinates_dir),
        help="Path to the directory where patch coordinates are stored.",
    )
    current_params["features_dir"] = st.text_input(
        "Features Directory",
        value=current_params.get("features_dir", DEFAULT_CONFIG.features_dir),
        help="Path to the directory where extracted features will be saved.",
    )
    current_params["wsi_meta_data_path"] = st.text_input(
        "WSI Metadata Path",
        value=current_params.get(
            "wsi_meta_data_path", DEFAULT_CONFIG.wsi_meta_data_path
        ),
        help="Path to the WSI metadata file (e.g., CSV or JSON).",
    )
    return current_params


def render_processing_settings_section(current_params: dict) -> dict:
    """Renders the input fields for processing settings and returns their values."""

    device_options = ["cpu", "cuda", "mps"]
    current_device_index = (
        device_options.index(current_params.get("device", DEFAULT_CONFIG.device))
        if current_params.get("device", DEFAULT_CONFIG.device) in device_options
        else 0
    )
    current_params["device"] = st.selectbox(
        "Device",
        options=device_options,
        index=current_device_index,
        help="Device to use for computation (e.g., 'cpu', 'cuda', 'mps').",
    )
    current_params["patch_batch_size"] = st.number_input(
        "Patch Batch Size",
        min_value=1,
        value=current_params.get("patch_batch_size", DEFAULT_CONFIG.patch_batch_size),
        step=1,
        help="Number of patches to process in each batch.",
    )
    current_params["num_workers"] = st.number_input(
        "Number of Workers",
        min_value=0,
        value=current_params.get("num_workers", DEFAULT_CONFIG.num_workers),
        step=1,
        help="Number of subprocesses to use for data loading. 0 means main process only.",
    )
    log_level_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    current_log_level_index = (
        log_level_options.index(current_params.get("log_level", DEFAULT_CONFIG.log_level))
        if current_params.get("log_level", DEFAULT_CONFIG.log_level) in log_level_options
        else 0
    )
    current_params["log_level"] = st.selectbox(
        "Log Level",
        options=log_level_options,
        index=current_log_level_index,
        help="Log level for the feature extraction script.",
    )
    current_params["disable_progress_bar"] = st.checkbox(
        "Disable Progress Bar",
        value=current_params.get("disable_progress_bar", DEFAULT_CONFIG.disable_progress_bar),
        help="Disable the progress bar for the feature extraction script.",
    )
    return current_params


def render_model_config_section(current_params: dict) -> dict:
    """Renders the input fields for model configuration and returns their values."""
    st.subheader("🧠 Model Configuration")

    model_name_options = [
        ModelName.RESNET50.value,
        ModelName.UNI.value,
        ModelName.CONCH.value,
        ModelName.PLIP.value,
    ]
    current_model_index = (
        model_name_options.index(
            current_params.get("model_name", DEFAULT_CONFIG.model_name)
        )
        if current_params.get("model_name", DEFAULT_CONFIG.model_name)
        in model_name_options
        else 0
    )
    current_params["model_name"] = st.selectbox(
        "Model Name",
        options=model_name_options,
        index=current_model_index,
        help="Name of the pre-trained model to use for feature extraction.",
    )
    current_params["hf_token"] = st.text_input(
        "HF Token",
        value=current_params.get("hf_token", DEFAULT_CONFIG.hf_token),
        help="Hugging Face token for the model. (It's optional for ResNet50)",
    )
    return current_params


def render_config_preview(config_data: dict):
    """Displays the current configuration in JSON format."""
    st.subheader("Current Configuration Preview")
    try:
        # Create a VisionFeatureExtractionConfig object on the fly for display
        preview_config = VisionFeatureExtractionConfig(**config_data)
        st.json(asdict(preview_config))
    except Exception as e:
        st.warning(
            f"Could not generate full config preview due to incomplete data: {e}"
        )
        st.json(config_data)  # Show raw data if full config can't be formed


def render_config_download_button(config_data: dict):

    yaml_str = yaml.dump(config_data)
    yaml_bytes = yaml_str.encode("utf-8")

    st.download_button(
        label="📥 Download Config",
        data=BytesIO(yaml_bytes),
        file_name="generated_config.yml",
        mime="text/yaml",
    )


def run_button(config_data: dict):
    """Renders buttons for running the script."""

    # Save the config to a file
    output_config_path = "configs/streamlit_generated_config.yml"
    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    config_to_save = VisionFeatureExtractionConfig(**config_data)
    config_to_save.to_yaml(output_config_path)

    # Create the command to run the script
    command = [
        "python",
        "extract_vision_features_local.py",
        "--config",
        output_config_path,
    ]
    
    # Create columns for start and stop buttons
    col1, col2 = st.columns(2)
    
    with st.expander("Logs"):
        log_placeholder = st.empty()
        
    # Initialize session state for process management
    if 'process' not in st.session_state:
        st.session_state.process = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    with col1:
        if st.button("🚀 Start Extraction", disabled=st.session_state.is_running):
            st.session_state.is_running = True
            st.session_state.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
    
    with col2:
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("⏹️ Stop Extraction", disabled=not st.session_state.is_running):
                if st.session_state.process:
                    st.session_state.process.terminate()  # Graceful termination
                    st.session_state.is_running = False
                    st.success("Process stopped")
        with col2b:
            if st.button("🗑️ Clear Logs"):
                st.session_state.logs = []
                st.success("Logs cleared!")
    
    # Display logs if process is running
    if st.session_state.is_running and st.session_state.process:
        with st.spinner("Extracting features..."):
            try:
                with log_placeholder.container():
                    st.write("--- Real-time Logs ---")
                    for line in iter(st.session_state.process.stdout.readline, ''):
                        log_line = line.strip()
                        st.session_state.logs.append(log_line)  # Store in session state
                        st.code(log_line) # Display each line as received

                return_code = st.session_state.process.wait()
                st.session_state.is_running = False
                
                if return_code == 0:
                    st.success("Feature extraction completed successfully!")
                    st.balloons()
                else:
                    st.error(f"Feature extraction failed with error code {return_code}, see logs for details")
            except Exception as e:
                st.error(f"Error during feature extraction: {e}")
                st.exception(e)
                st.session_state.is_running = False
    
    # Display stored logs if not running (after completion or termination)
    elif st.session_state.logs:
        with log_placeholder.container():
            st.write("--- Process Logs ---")
            for log_line in st.session_state.logs:
                st.code(log_line)
        

# --- Main App Function ---
def main():
    st.set_page_config(page_title="Feature Extraction", layout="wide")
    st.title("🔬 Feature Extraction")
    st.markdown("Set up your parameters for the feature extraction script.")

    config_params = {}
    config_params = render_file_uploader_section(config_params)
    with st.sidebar:
        config_params = render_model_config_section(config_params)
        config_params = render_processing_settings_section(config_params)

    with st.expander("📂 Paths"):
        config_params = render_paths_section(config_params)

    with st.expander("Current Configuration Preview"):
        render_config_preview(config_params)
        render_config_download_button(config_params)

    run_button(config_params)


if __name__ == "__main__":
    main()
