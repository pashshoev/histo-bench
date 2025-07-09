import streamlit as st
import subprocess


def get_processing_parameters():
    """
    Handles all Streamlit UI components for collecting processing parameters.
    Returns a dictionary of collected parameters.
    """
    st.markdown("---")
    st.header("Core Directories & Files")

    source_dir = st.text_input(
        "Source Directory (`--source`)",
        value="data/TCGA-LGG/", # Example default
        help="Path to the folder containing raw WSI image files."
    )

    save_dir = st.text_input(
        "Save Directory (`--save_dir`)",
        value="data/processed_wsi/", # Example default
        help="Directory to save processed data (patches, masks, stitches)."
    )

    preset_file = st.text_input(
        "Preset File (`--preset`)",
        value="",
        placeholder="e.g., tcga.csv (optional)",
        help="Predefined profile of default segmentation and filter parameters (.csv)."
    )

    process_list_file = st.text_input(
        "Process List File (`--process_list`)",
        value="",
        placeholder="e.g., image_list.csv (optional)",
        help="Name of a CSV file (within save_dir) listing specific images to process."
    )

    st.sidebar.header("Processing Parameters")

    step_size = st.sidebar.number_input(
        "Step Size (`--step_size`)",
        min_value=1,
        value=512,
        step=1,
        help="Step size for sliding window during patching."
    )

    patch_size = st.sidebar.number_input(
        "Patch Size (`--patch_size`)",
        min_value=64,
        value=512,
        step=64,
        help="Size of the square patches to extract."
    )

    patch_level = st.sidebar.number_input(
        "Patch Level (`--patch_level`)",
        min_value=0,
        value=0,
        step=1,
        help="Downsample level at which to extract patches (0 is original resolution)."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Processing Steps")

    seg_flag = st.sidebar.checkbox(
        "Enable Segmentation (`--seg`)",
        value=True,
        help="Perform tissue segmentation to identify regions of interest."
    )

    patch_flag = st.sidebar.checkbox(
        "Enable Patch Extraction (`--patch`)",
        value=True,
        help="Extract patches from the WSI based on segmentation (if enabled) and patching parameters."
    )

    stitch_flag = st.sidebar.checkbox(
        "Enable Stitching (`--stitch`)",
        value=True,
        help="Stitch patches back together for visualization purposes."
    )

    auto_skip_enabled = st.sidebar.checkbox(
        "Enable Auto Skip (`--auto_skip`)",
        value=False,
        help="If checked, the script will automatically skip already processed files."
    )
    disable_progress_bar = st.sidebar.checkbox(
        "Disable Progress Bar (`--disable_progress_bar`)",
        value=True,
        help="Disable the progress bar."
    )
    log_level = st.sidebar.selectbox(
        "Log Level (`--log_level`)",
        options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        index=1,
        help="Log level for the script."
    )


    st.markdown("---")

    return {
        "source_dir": source_dir,
        "save_dir": save_dir,
        "preset_file": preset_file,
        "process_list_file": process_list_file,
        "step_size": step_size,
        "patch_size": patch_size,
        "patch_level": patch_level,
        "seg_flag": seg_flag,
        "patch_flag": patch_flag,
        "stitch_flag": stitch_flag,
        "auto_skip_enabled": auto_skip_enabled,
        "log_level": log_level,
        "disable_progress_bar": disable_progress_bar,
    }


def run_processing_command(params):
    """
    Constructs and runs the subprocess command based on the collected parameters.
    Displays real-time logs and success/error messages.
    """
    if not params["source_dir"] or not params["save_dir"]:
        st.error("Please provide both Source Directory and Save Directory.")
        return

    command_parts = [
        "python", "-u",
        "CLAM/create_patches_fp.py",
        "--source", params["source_dir"],
        "--save_dir", params["save_dir"],
        "--patch_size", str(params["patch_size"]),
    ]

    if params["preset_file"]:
        command_parts.extend(["--preset", params["preset_file"]])
    if params["process_list_file"]:
        command_parts.extend(["--process_list", params["process_list_file"]])

    command_parts.extend(["--step_size", str(params["step_size"])])
    command_parts.extend(["--patch_level", str(params["patch_level"])])

    if params["seg_flag"]:
        command_parts.append("--seg")
    if params["patch_flag"]:
        command_parts.append("--patch")
    if params["stitch_flag"]:
        command_parts.append("--stitch")
    if params["auto_skip_enabled"]:
        command_parts.append("--auto_skip")
    if params["disable_progress_bar"]:
        command_parts.append("--disable_progress_bar")
    
    command_parts.extend(["--log_level", params["log_level"].upper()])

    st.code(f"Running command: {' '.join(command_parts)}")

    with st.expander("Logs"):
        log_placeholder = st.empty()
    
    try:
        with st.spinner("Running command... This might take a while."):
            process = subprocess.Popen(
                        command_parts,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1
                    )

            with log_placeholder.container():
                st.write("--- Real-time Logs ---")
                for line in iter(process.stdout.readline, ''):
                    st.code(line.strip()) # Display each line as received

            return_code = process.wait()

            if return_code == 0:
                st.success(f"Segmentation & Patching completed successfully! (Exit Code: {return_code})")
                st.balloons()
            else:
                st.error(f"Segmentation & Patching failed with Exit Code: {return_code}, see logs for details")
    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        st.exception(e) # Display full traceback for debugging


def app():
    st.set_page_config(
        page_title="Extract tissue patch coordinate",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.title("Extract tissue patch coordinates")

    # Call the function to get parameters from the UI
    params = get_processing_parameters()

    if st.button("Exract"):
        run_processing_command(params)


if __name__ == "__main__":
    app()