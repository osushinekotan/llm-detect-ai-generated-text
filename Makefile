# GCP utils
.PHONY: chown_git install_docker gcloud_auth setup_gcs_mount install_gpu_driver

PROJECT_ID ?=
BUCKET_NAME ?=
MOUNT_DIR ?= "./resources"

chown_git:
	sudo chown -R $(shell whoami) .git/

# install docker and compose in ubuntu 
install_docker:
	sudo apt-get update
	sudo apt-get install -y ca-certificates curl gnupg
	sudo install -m 0755 -d /etc/apt/keyrings
	curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
	sudo chmod a+r /etc/apt/keyrings/docker.gpg
	echo "deb [arch=$$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $$(. /etc/os-release && echo $$VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	sudo apt-get update
	sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
	sudo docker run hello-world

gcloud_auth:
	gcloud auth login

setup_gcs_mount: 
	gcloud config set project $(PROJECT_ID)
	if ! gsutil ls gs://$(BUCKET_NAME); then \
		gcloud storage buckets create gs://$(BUCKET_NAME) --location=us-central1; \
	fi
	export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
	echo "deb http://packages.cloud.google.com/apt $$GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list && \
	curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - && \
	sudo apt-get update && \
	sudo apt-get install -y fuse gcsfuse && \
	sudo gcsfuse -o allow_other --file-mode=777 --dir-mode=777 $(BUCKET_NAME) $(MOUNT_DIR) && \
	echo "Mounted $(BUCKET_NAME) at $(MOUNT_DIR)."

# install gpu driver in gce instance
install_gpu_driver:
	sudo /opt/deeplearning/install-driver.sh

