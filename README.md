
```
export PROJECT_ID=""
export BUCKET_NAME="llm-detect-ai-generated-text"
PROJECT_ID=${PROJECT_ID} BUCKET_NAME=${BUCKET_NAME} make install_gcsfuse
BUCKET_NAME=${BUCKET_NAME} make mount_bucket
```