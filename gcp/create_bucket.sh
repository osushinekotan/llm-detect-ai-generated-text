gcloud config set project ${PROJECT_ID}
gcloud storage buckets create gs://${BUCKET_NAME} --location=${LOCATION}