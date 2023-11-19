# GCP utils
.PHONY: chown_git install_docker gcloud_auth install_gcsfuse install_gpu_driver mount_bucket

PROJECT_ID ?=
BUCKET_NAME ?=
MOUNT_DIR ?= "./resources"

chown_git:
	sudo chown -R $(shell whoami) .git/

# install docker and compose in ubuntu 
install_docker:
	sudo apt-get update
	sudo apt-get install ca-certificates curl gnupg
	sudo install -m 0755 -d /etc/apt/keyrings
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
	sudo chmod a+r /etc/apt/keyrings/docker.gpg

	echo \
	"deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
	"$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
	sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	sudo apt-get update
	sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
	sudo docker run hello-world

gcloud_auth:
	gcloud auth login

install_gcsfuse: 
	gcloud config set project ${PROJECT_ID}
	gcloud storage buckets create gs://${BUCKET_NAME} --location=asia-east1
	export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
	echo "deb http://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
	curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
	sudo apt-get update
	sudo apt-get install -y fuse gcsfuse

mount_bucket:
	sudo gcsfuse -o allow_other -file-mode=777 -dir-mode=777 ${BUCKET_NAME} ${MOUNT_DIR}
	
# install gpu driver in gce instance
install_gpu_driver:
	sudo /opt/deeplearning/install-driver.sh

