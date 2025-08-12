python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "COMET_API_KEY=" > .env

mkdir -p data/TCGA-Lung/features/ResNet50
gcloud storage cp -r gs://histo-bench/TCGA-Lung/features/ResNet50/* data/TCGA-Lung/features/ResNet50/
gcloud storage cp -r gs://histo-bench/TCGA-LUSC/features/ResNet50/* data/TCGA-Lung/features/ResNet50/