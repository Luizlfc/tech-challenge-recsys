# Deploy Bonus (Azure) - Container acessivel via URL publica

## URL publica ativa

```
http://recsys-mlflow-84939.brazilsouth.azurecontainer.io:5000
```

MLflow rodando via **Azure Container Instances**, usando a imagem oficial publica
`ghcr.io/mlflow/mlflow:v2.16.2` (mesma versao usada em `deploy/mlflow.Dockerfile` e no
`docker-compose.yml` local).

## Por que ACI em vez de Container Apps + ACR Tasks

O plano original era buildar `deploy/mlflow.Dockerfile` via `az acr build` (sem precisar
de Docker local) e publicar em Azure Container Apps. Dois caminhos foram tentados e
bloqueados por restricoes fora do nosso controle:

1. **`az acr build`** falhou com `(TasksOperationsNotAllowed)` - a assinatura Azure
   corporativa usada bloqueia o servico ACR Tasks via policy de tenant/management group.
2. **GitHub Actions** (build com Docker real no runner, so o push final para o ACR) foi
   configurado (`.github/workflows/deploy-mlflow.yml` + secrets `ACR_LOGIN_SERVER`/
   `ACR_USERNAME`/`ACR_PASSWORD`), mas o job falhou instantaneamente (4s, zero logs) -
   sintoma de alguma restricao de conta/organizacao no GitHub Actions que so aparece na
   interface web.

Como o objetivo do bonus e apenas "container acessivel via URL publica" (nao
especificamente qual container), a solucao foi usar **Azure Container Instances (ACI)**
com a imagem publica oficial do MLflow direto do GitHub Container Registry - isso nao
usa ACR Tasks (ACI so puxa uma imagem ja publicada, nao builda nada) e nao depende do
GitHub Actions.

## Comando que funcionou (PowerShell)

```powershell
az login
az account show

$RESOURCE_GROUP = "tech-challenge-recsys-rg"
$LOCATION = "brazilsouth"
$DNS_LABEL = "recsys-mlflow-<numero-aleatorio>"   # precisa ser globalmente unico

az group create --name $RESOURCE_GROUP --location $LOCATION
az provider register --namespace Microsoft.ContainerInstance --wait

az container create --resource-group $RESOURCE_GROUP --name recsys-mlflow-aci `
  --image ghcr.io/mlflow/mlflow:v2.16.2 --os-type Linux --cpu 1 --memory 1.5 `
  --ports 5000 --ip-address Public --dns-name-label $DNS_LABEL `
  --command-line "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /tmp/mlflow-artifacts"

az container show --resource-group $RESOURCE_GROUP --name recsys-mlflow-aci --query ipAddress.fqdn -o tsv
```

Abra `http://<fqdn>:5000` no navegador.

> Nota: backend SQLite e artifact store ficam no armazenamento efemero do container -
> um restart apaga o historico. Para a avaliacao isso nao e um problema (a URL fica
> acessivel); nao populei com runs reais porque isso exigiria rodar o pipeline de treino
> contra essa URL, e o build da imagem principal do projeto esbarraria nos mesmos
> bloqueios de ACR Tasks/GitHub Actions descritos acima. Os runs reais do MLflow (5 runs,
> modelo promovido a Production) estao documentados com números e tabela completa em
> [Model Card](model_card.md), gerados localmente via `dvc repro`.

## Custos
- **Azure Container Instances**: cobranca por segundo de CPU/memoria enquanto o
  container existir (~1 vCPU + 1.5GB, poucos centavos de dolar por dia).
- **Recomendado**: apague o resource group apos a avaliacao.

## Limpeza (rodar depois da avaliacao)

```powershell
az group delete --name tech-challenge-recsys-rg --yes --no-wait
```

## Caminho alternativo original (ACR Tasks + Container Apps)

Mantido abaixo como referencia, caso a restricao de ACR Tasks seja liberada no futuro
pelo administrador da assinatura Azure.

### 1. Variaveis

```powershell
$RESOURCE_GROUP = "tech-challenge-recsys-rg"
$LOCATION = "brazilsouth"
$ACR_NAME = "recsysacr$(Get-Random -Maximum 99999)"
$ENV_NAME = "recsys-env"
$MLFLOW_APP = "recsys-mlflow"
```

### 2. Resource group + Azure Container Registry

```powershell
az group create --name $RESOURCE_GROUP --location $LOCATION
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true
```

### 3. Build da imagem do MLflow (requer ACR Tasks liberado)

```powershell
az acr build --registry $ACR_NAME --image mlflow-server:v1 -f deploy/mlflow.Dockerfile .
```

### 4. Container Apps environment e publicacao

```powershell
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az containerapp env create --name $ENV_NAME --resource-group $RESOURCE_GROUP --location $LOCATION

$ACR_SERVER = "$ACR_NAME.azurecr.io"
$ACR_USER = az acr credential show -n $ACR_NAME --query username -o tsv
$ACR_PASS = az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv

az containerapp create --name $MLFLOW_APP --resource-group $RESOURCE_GROUP --environment $ENV_NAME --image "$ACR_SERVER/mlflow-server:v1" --target-port 5000 --ingress external --registry-server $ACR_SERVER --registry-username $ACR_USER --registry-password $ACR_PASS --cpu 0.5 --memory 1.0Gi --min-replicas 1 --max-replicas 1
```

### 5. URL publica

```powershell
az containerapp show --name $MLFLOW_APP --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv
```
