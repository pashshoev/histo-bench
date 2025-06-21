split_manifest:
	python scripts/data_preparation/split_gdc_manifest.py -m data/TCGA-LGG/manifests/gdc_manifest.2025-05-31.101400.txt -o data/TCGA-LGG/manifests

install_deps:
	pip install --upgrade pip
	pip install -r requirements.txt

setup_clam:
	git clone https://github.com/mahmoodlab/CLAM.git
	pip install -r requirements_clam.txt
	cp -r CLAM/presets ./

install_conch:
	pip install git+https://github.com/Mahmoodlab/CONCH.git
