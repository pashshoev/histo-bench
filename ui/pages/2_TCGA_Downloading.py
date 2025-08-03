import streamlit as st
import subprocess
import os
from scripts.data_preparation.download_from_manifest import download_and_move


st.set_page_config(
    page_title="Download from Manifest",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="auto",
)


# File picker for manifest
manifest_file = st.file_uploader("Select manifest file", type=['txt', 'csv', 'tsv'])

# If file is uploaded, save it temporarily and get the path
manifest_path = None
if manifest_file is not None:
    # Save uploaded file to temp location
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    manifest_path = os.path.join(temp_dir, manifest_file.name)
    with open(manifest_path, "wb") as f:
        f.write(manifest_file.getbuffer())
    st.success(f"Manifest file uploaded: {manifest_file.name}")

idx_column1, idx_column2 = st.columns(2)
start_index = idx_column1.number_input("Start index (-s)", min_value=0, step=1)
end_index = idx_column2.number_input("End index (-e)", min_value=0, step=1)

# Directory inputs
download_dir = os.path.join("data", "tmp")
log_file = os.path.join(download_dir, "failed_downloads.log")

move_dir = st.text_input("Destination directory", value="data/TCGA-LGG/")


# Run button
if st.button("Download"):
    if not all([manifest_path, download_dir, move_dir, log_file]):
        st.warning("Please fill in all fields and upload a manifest file.")
    else:
        
        try:
            with st.spinner("🔄 Downloading files..."):
                failed_downloads = download_and_move(manifest_file_path=manifest_path,
                                start_index=start_index, 
                                end_index=end_index, 
                                download_dir=download_dir, 
                                destination_dir=move_dir)
                st.success("Script completed successfully.")
                st.balloons()

                if failed_downloads:
                    st.write(f"Failed downloads: {len(failed_downloads)}")
                    with open(log_file, "w") as f:
                        for filename in failed_downloads:
                            f.write(f"{filename}\n")
                    st.download_button("Failed downloads", log_file, file_name=log_file)
        except Exception as e:
            st.error("Script failed.")
            st.text(e)
