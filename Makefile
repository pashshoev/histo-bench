.PHONY: install_deps run_local build run stop restart push

# Path on your local machine
HOST_DATA_PATH:=
# Path inside the container
CONTAINER_DATA_PATH:=/app/data

DOCKER_IMAGE_NAME:=histo-bench-app
DOCKER_PLATFORM:=linux/amd64
DOCKERFILE_NAME:=Dockerfile


install_deps:
	pip install --upgrade pip
	pip install -r requirements.txt

run_local:
	PYTHONPATH=. streamlit run ui/1_Home.py

build:
	docker build --platform $(DOCKER_PLATFORM) -t $(DOCKER_IMAGE_NAME) -f $(DOCKERFILE_NAME) .

run:
	docker run --platform $(DOCKER_PLATFORM) -d --name histo-app-instance -p 8501:8501 -v $(HOST_DATA_PATH):$(CONTAINER_DATA_PATH) $(DOCKER_IMAGE_NAME)

stop:
	docker stop histo-app-instance || true
	docker rm histo-app-instance || true

restart: stop run

push:
	docker tag $(DOCKER_IMAGE_NAME) pashshoev/histo-bench:latest
	docker push pashshoev/histo-bench:latest