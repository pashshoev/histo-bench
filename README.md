# histo-bench

Benchmarking Generalist Vision-Language Models vs. Domain-Specific Foundation Models in Digital Pathology

## Description

This project aims to evaluate and compare the performance of general-purpose multimodal models (such as CLIP, SAM, and GPT-4) with specialized models designed for histopathological image analysis. We focus on H&E-stained slides for classification and segmentation tasks, analyzing performance, scalability, and trade-offs.

## Goals

- Compare generalist and specialist models on histopathological datasets
- Benchmark classification and segmentation tasks
- Evaluate performance, data efficiency, and cost trade-offs
- Provide reproducible results and insights for the research community

## Project Structure

## Installation
1. `git clone https://github.com/pashshoev/histo-bench.git` to clone repo
2. `make install_deps` to install dependencies
3. `make setup_clam` to setup CLAM repo for patch coordinates extraction

## Usage
1. Download patches from manifest:
   - `python scripts/data_preparation/download_from_manifest.py
     -m MANIFEST_FILE \
     -s START_INDEX \
-e END_INDEX \
-d WHERE_TO_DOWNLOAD \
-D WHERE_TO_MOVE \
-u FILE_TO_LOG_FAILED_DOWNLOAD_IDS`

2. Extract coordinates using CLAM
3. Extract features using...