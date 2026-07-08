# Deploy Bonus (Azure) - Container acessivel via URL publica

Este guia builda a imagem do servidor MLflow **na nuvem** via `az acr build` (nao
precisa de Docker local) e publica um Container App com URL publica, satisfazendo o
criterio de bonus "Container acessivel via URL publica" (5%).

Rode estes comandos no seu terminal, onde o `az` ja esta logado (`az login`).

## Custos
- **Azure Container Registry (Basic)**: ~US$ 0.167/dia (~US$ 5/mes) enquanto existir.
- **Azure Container Apps (Consumption plan)**: tem uma cota gratuita mensal generosa
  (180.000 vCPU-segundos + 360.000 GiB-segundos); para 1 replica pequena rodando
  durante a avaliacao, o custo tende a ficar dentro do free tier ou proximo de zero.
- **Recomendado**: rode o `az group delete` no final (ultimo comando deste guia) assim
  que a avaliacao terminar, para nao deixar recursos cobrando indefinidamente.

## 1. Variaveis (ajuste se quiser)

```bash
export RESOURCE_GROUP="tech-challenge-recsys-rg"
export LOCATION="brazilsouth"
export ACR_NAME="recsysacr$RANDOM"      # precisa ser globalmente unico (so letras/numeros)
export ENV_NAME="recsys-env"
export MLFLOW_APP="recsys-mlflow"
```

## 2. Criar resource group + Azure Container Registry

```bash
az group create --name $RESOURCE_GROUP --location $LOCATION

az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME \
  --sku Basic --admin-enabled true
```

## 3. Buildar a imagem do MLflow direto na nuvem (sem Docker local)

```bash
az acr build --registry $ACR_NAME --image mlflow-server:v1 \
  -f deploy/mlflow.Dockerfile .
```

## 4. Criar o Container Apps environment e publicar o MLflow

```bash
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az containerapp env create --name $ENV_NAME \
  --resource-group $RESOURCE_GROUP --location $LOCATION

ACR_SERVER="$ACR_NAME.azurecr.io"
ACR_USER=$(az acr credential show -n $ACR_NAME --query username -o tsv)
ACR_PASS=$(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv)

az containerapp create --name $MLFLOW_APP \
  --resource-group $RESOURCE_GROUP --environment $ENV_NAME \
  --image "$ACR_SERVER/mlflow-server:v1" \
  --target-port 5000 --ingress external \
  --registry-server "$ACR_SERVER" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
  --cpu 0.5 --memory 1.0Gi --min-replicas 1 --max-replicas 1
```

## 5. Pegar a URL publica

```bash
az containerapp show --name $MLFLOW_APP --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv
```

Abra `https://<fqdn-retornada>` no navegador - essa e a URL publica do container para
colocar na entrega. A UI do MLflow deve carregar (vazia, ate rodar o passo opcional
abaixo).

> Nota: o backend SQLite fica em armazenamento efemero do Container App (sem volume
> persistente configurado, para manter o guia simples). Isso significa que um restart
> do container apaga o historico. Para a avaliacao, isso nao e um problema - a URL fica
> no ar e acessivel; para uso continuo real, seria necessario montar Azure Files como
> volume persistente.

## 6. (Opcional) Popular o MLflow publico rodando o pipeline uma vez

Builda a imagem principal do projeto e roda como um Container Apps Job (executa uma vez
e termina), apontando para a URL publica do MLflow:

```bash
az acr build --registry $ACR_NAME --image recsys-train:v1 -f Dockerfile .

MLFLOW_URL=$(az containerapp show --name $MLFLOW_APP --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv)

az containerapp job create --name recsys-train-job \
  --resource-group $RESOURCE_GROUP --environment $ENV_NAME \
  --trigger-type Manual --replica-timeout 1800 --replica-retry-limit 1 \
  --image "$ACR_SERVER/recsys-train:v1" \
  --registry-server "$ACR_SERVER" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
  --cpu 1.0 --memory 2.0Gi \
  --env-vars "MLFLOW_TRACKING_URI=https://$MLFLOW_URL" "MLFLOW_EXPERIMENT_NAME=recsys-movielens" "MLFLOW_REGISTRY_MODEL_NAME=movielens-recommender"

az containerapp job start --name recsys-train-job --resource-group $RESOURCE_GROUP
```

Acompanhe os logs em Azure Portal (Container Apps Jobs > recsys-train-job > Execution
history > Logs) ate concluir, depois recarregue a URL publica do MLflow para ver os
5 runs e o modelo `movielens-recommender` registrado/promovido.

## 7. Limpeza (rodar depois da avaliacao)

```bash
az group delete --name $RESOURCE_GROUP --yes --no-wait
```

Isso remove o resource group inteiro (registry, container apps, job) para parar de
gerar custo.
