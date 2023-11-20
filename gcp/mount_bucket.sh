gcloud config set project ${PROJECT_ID}
gcloud storage buckets create gs://${BUCKET_NAME} --location=${LOCATION}
sudo gcsfuse -o allow_other -file-mode=777 -dir-mode=777 ${BUCKET_NAME} ${MOUNT_DIR}