docker run -d -p 5000:5000 \
    -v ../dev/mlflow/data:/mlflow/mlruns/ \
    --user $UID:$GID \
    --name mlflow-server  \
    burakince/mlflow \
    mlflow server --backend-store-uri /mlflow/mlruns/ --default-artifact-root /mlflow/mlruns/ --host=0.0.0.0 --port=5000