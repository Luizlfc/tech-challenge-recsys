# Standalone MLflow tracking server image, reused by docker-compose (local) and
# Azure Container Apps (cloud deploy bonus) so both environments run the exact
# same image built from the exact same source.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.16.2

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", "--port", "5000"]
