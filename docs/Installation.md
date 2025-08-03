### Installation & Setup

To get the application running, follow these steps:
### Installation & Setup

To get the application running, follow these steps:

* **1. Clone the Repository:**
    ```bash
    git clone [https://github.com/pashshoev/histo-bench](https://github.com/pashshoev/histo-bench)
    cd histo-bench
    ```
    Once cloned, you have two primary ways to run the application, each with its own advantages and considerations, especially regarding your operating system and hardware:

    * **Option A: Run as Docker Containers** (See steps 2 - 4)
        * **Recommended for:**
            * **Reproducibility and consistent environments:** Docker guarantees your setup works identically across different machines.
  
        * **Note for macOS (M1/M2) users:** While Docker is great for development and isolation, **GPU acceleration (MPS) for deep learning models is NOT available directly within Linux containers** on your Mac. 

    * **Option B: Run in a Local Virtual Environment** (See step 5)
        * **Recommended for:**
            * **Faster iterative development on any OS:** Avoids Docker's virtualization overhead.
            * **Leveraging native GPU acceleration on your local machine:**
                * On **macOS (M1/M2/etc.)**: Allows PyTorch to utilize **MPS**
                * On **Linux with NVIDIA GPUs**: Allows PyTorch to utilize **CUDA**
        * **Consideration:** Environment consistency and reproducibility are harder to guarantee across different local systems due to variations in installed libraries and system configurations.

* **2. Prepare Docker Image:**
    Choose one of the following options to get the Docker image ready:

    * **2.a. Build Locally (Recommended for Development):**
        ```bash
        make build
        ```
        *(This command builds the `linux/amd64` Docker image from the source, which is necessary for the GDC Data Transfer Tool to function correctly.)*

    * **2.b. Pull from Docker Hub (For Quick Start):**
        Alternatively, you can pull the pre-built image directly from Docker Hub.
        ```bash
        docker pull pashshoev/histo-bench:latest
        ```

* **3. Run Streamlit Application (Docker):**
    * **3.a. Configure Data Mount Path (Crucial for Data Persistence!):**
        This application performs data-intensive operations like downloading gigapixel slides, extracting tissue coordinates, and generating features. To ensure these large output files are **saved and accessible on your host machine** (even after the Docker container stops) and not lost, a **volume mount** is used.

        You must specify the correct path.

        * **Option 1: Permanent Change (Edit Makefile)**
            For a persistent change that applies every time you run `make run`, directly edit the `Makefile` and update the `HOST_DATA_PATH` variable:
            ```makefile
            HOST_DATA_PATH := /path/to/your/actual/data/folder
            ```

        * **Option 2: Temporary Change (Command Line Parameter)**
            For a one-time run or quick test without modifying the `Makefile`, you can pass the path as a parameter directly in your terminal:
            ```bash
            make run HOST_DATA_PATH=/path/to/your/actual/data/folder
            ```

    * **3.b. Start the Application:**
        Once the data path is configured, run the application:
        ```bash
        make run
        ```
        Once started, you can access the Streamlit application in your web browser at `http://localhost:8501`.

* **4. Stop/Restart Docker Container:**
    ```bash
    # To stop the running container
    make stop

    # To stop and then restart the container
    make restart
    ```

* **5. Local Setup (Optional - for local development without Docker):**
    - Make sure that `gdc-client` is installed. 
    - Run following commands:
    - ```bash
        # Install Python dependencies
        make install_deps

        # Run the Streamlit app locally
        make run_local
        ```

---

### Supported Operating Systems

This Docker setup is designed to run on the following systems:

* **Linux (x64) & Windows (x64):** Runs natively and optimally.
* **macOS (Intel x64):** Runs natively.
* **macOS (Apple Silicon / M1/M2/ARM64):** Runs successfully via Rosetta 2 emulation, allowing the `linux/amd64` Docker image to function. Performance might be slightly reduced compared to a native build.
* **Linux (ARM64):** The container will run via emulation, but expect noticeably slower performance due to the cross-architecture translation.

---

### Troubleshooting

* **`gdc-client` not working or errors related to binaries:**
    * This is often due to an architecture mismatch. The current setup explicitly builds and runs for `linux/amd64`.
    * Ensure your `make build` and `make run` commands (or direct `docker run` if not using make) both include `--platform linux/amd64`.

* **`Port 8501 already in use`:**
    * Another application or container is already using port `8501` on your host machine.
    * Use `make stop` to ensure any previous container instances are removed. If the issue persists, find and stop the conflicting process manually. (e.g., `lsof -i :8501` on Linux/macOS to identify, then `kill <PID>`).

* **Container starts but then immediately exits/crashes:**
    * Check the container logs for errors:
        ```bash
        docker logs histo-app-instance
        ```
    * This usually points to issues within your application code or missing runtime dependencies.

---

### Supported Operating Systems

This Docker setup is designed to run on the following systems:

* **Linux (x64) & Windows (x64):** Runs natively and optimally.
* **macOS (Intel x64):** Runs natively.
* **macOS (Apple Silicon / M1/M2/ARM64):** Runs successfully via Rosetta 2 emulation, allowing the `linux/amd64` Docker image to function. Performance might be slightly reduced compared to a native build.
* **Linux (ARM64):** The container will run via emulation, but expect noticeably slower performance due to the cross-architecture translation.

---

### Troubleshooting

* **`gdc-client` not working or errors related to binaries:**
    * This is often due to an architecture mismatch. The current setup explicitly builds and runs for `linux/amd64`.
    * Ensure your `make build` and `make run` commands (or direct `docker run` if not using make) both include `--platform linux/amd64`.

* **`Port 8501 already in use`:**
    * Another application or container is already using port `8501` on your host machine.
    * Use `make stop` to ensure any previous container instances are removed. If the issue persists, find and stop the conflicting process manually. (e.g., `lsof -i :8501` on Linux/macOS to identify, then `kill <PID>`).

* **Container starts but then immediately exits/crashes:**
    * Check the container logs for errors:
        ```bash
        docker logs histo-app-instance
        ```
    * This usually points to issues within your application code or missing runtime dependencies.