# Tech Challenge Fase 2 - Sistema de Recomendacao

## Descricao
Pipeline completo de Machine Learning para recomendacao de produtos (top-K) baseada em
comportamento de navegacao/interacao dos usuarios. O modelo central e uma rede neural
estilo NeuralCF (embeddings + MLP) treinada com PyTorch, comparada com baselines
Scikit-Learn, com pipeline reprodutivel via DVC, experimentos rastreados no MLflow
(incluindo Model Registry com promocao Staging -> Production) e tudo containerizado em
Docker.

## Problema de Negocio
Uma empresa de e-commerce precisa recomendar produtos aos usuarios com base no seu
historico de navegacao/interacao. Como dataset de e-commerce publico com o volume de
interacoes exigido nao estava disponivel sem credenciais adicionais, o pipeline usa o
**MovieLens `ml-latest-small`** como proxy: avaliacoes de filmes com `rating >= 4.0` sao
tratadas como interacoes positivas (o equivalente a um "engajamento forte" com um
produto), preservando a mesma estrutura user-item-interacao de um catalogo de
e-commerce. Ver [Model Card](docs/model_card.md) para as limitacoes dessa escolha.

## Dataset
**MovieLens `ml-latest-small`** (GroupLens Research) - ~100.836 avaliacoes, 610
usuarios, 9.742 filmes. Baixado automaticamente por `scripts/download_data.py` (ou pela
stage `preprocess` do DVC) a partir de
`https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`.

## Design Patterns
- **Strategy** (`src/recsys/models/base.py`): `RecommenderStrategy` e a interface comum
  implementada por `PopularityRecommender`, `ItemKNNRecommender`, `SVDRecommender` e
  `MLPRecommenderStrategy` - o pipeline de treino/avaliacao nunca conhece o tipo
  concreto do modelo.
- **Factory** (`src/recsys/models/factory.py`): `ModelFactory.create(name, **kwargs)`
  instancia a strategy correta a partir do nome declarado em `params.yaml`.
- **Template Method** (`src/recsys/training/trainer.py`): `Trainer.run()` fixa a
  sequencia `fit -> evaluate`, compartilhada por todos os modelos.

## Estrutura do Projeto
```
tech-challenge-recsys/
|-- data/{raw,processed,features}/     # versionado via DVC
|-- models/{mlp,baselines}/            # versionado via DVC
|-- metrics/                           # metricas do DVC (json/csv, git-tracked)
|-- scripts/
|   |-- validate_env.py                # valida ambiente (deps, .env, binarios)
|   |-- download_data.py               # baixa o MovieLens
|   |-- promote_model.py               # gate manual Staging -> Production
|-- src/recsys/
|   |-- config.py                      # Pydantic Settings (.env)
|   |-- pipeline/{preprocess,feature_eng,train,evaluate}.py   # as 4 stages do DVC
|   |-- data/                          # loaders, dataset PyTorch, negative sampling
|   |-- models/                        # RecommenderStrategy, Factory, baselines, MLP
|   |-- training/trainer.py            # Template Method
|   |-- evaluation/{metrics,evaluator}.py
|   |-- mlflow_utils/{tracking,registry}.py
|   |-- utils/{seeding,io,logger}.py
|-- tests/                             # pytest
|-- dvc.yaml / params.yaml             # pipeline DVC (4 stages) + hiperparametros
|-- Dockerfile / docker-compose.yml    # containerizacao (train + mlflow server)
|-- docs/model_card.md
```

## Instalacao (Poetry)

```bash
git clone <este-repositorio>
cd tech-challenge-recsys

cp .env.example .env

poetry install
poetry run python scripts/validate_env.py
```

> Alternativa sem Poetry (usada para validar este projeto): `python -m venv .venv`,
> ative o ambiente e instale as dependencias de `pyproject.toml` com `pip`.

## Como Rodar o Pipeline (DVC)

```bash
poetry run dvc init          # ja versionado neste repo (.dvc/)
poetry run python scripts/download_data.py
poetry run dvc repro         # roda as 4 stages: preprocess -> feature_eng -> train -> evaluate
```

Cada `dvc repro` re-executa apenas as stages cujas dependencias/params mudaram
(`dvc.lock` registra os hashes). As metricas finais ficam em `metrics/eval_metrics.json`
e `metrics/eval_table.csv`.

## Como Rodar via Docker

```bash
cp .env.example .env
docker compose up --build
```

Isso sobe:
- **`mlflow`**: servidor MLflow (backend SQLite + artifact store local em volume),
  acessivel em http://localhost:5000.
- **`train`**: builda a imagem multi-stage do `Dockerfile` e roda
  `python scripts/download_data.py && dvc repro` dentro do container, com
  `data/`, `models/` e `metrics/` montados como volumes (saem no host).

## MLflow: Tracking e Model Registry

Acesse http://localhost:5000 (ou `http://localhost:5000` local, se rodando fora do
Docker com `mlflow server`) para ver:
- Um run por modelo treinado (`popularity`, `item_knn`, `svd`, `mlp`) na experiment
  `recsys-movielens`, com params e metricas de validacao.
- Um run `evaluate` com a tabela comparativa completa (`eval_table.csv`) como artefato.
- O modelo vencedor (por `ndcg@10`) registrado em **Model Registry** sob o nome
  `movielens-recommender`, promovido automaticamente `Staging -> Production` se
  `ndcg@10 >= promotion_threshold` (configuravel em `params.yaml`).

Para reaplicar o gate de promocao manualmente (ex.: apos re-treinar):
```bash
poetry run python scripts/promote_model.py
```

## Como Testar

```bash
poetry run pytest tests/ -v
poetry run ruff check src/ tests/ scripts/
```

## Resultados (test set, protocolo leave-one-out)

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MRR@10 |
|--------|--------------|-----------|---------|--------|
| Popularity | 0.0043 | 0.0433 | 0.0229 | 0.0166 |
| Item-KNN | 0.0062 | 0.0616 | 0.0288 | 0.0191 |
| **SVD (Production)** | **0.0083** | **0.0832** | **0.0414** | **0.0290** |
| MLP (NeuralCF) | 0.0045 | 0.0449 | 0.0223 | 0.0154 |

Tabela completa (k = 5, 10, 20) em `metrics/eval_table.csv`. Detalhes, limitacoes e
vieses em [docs/model_card.md](docs/model_card.md).

## Tech Stack
- **Python 3.11+**
- **PyTorch** - rede neural NeuralCF-style
- **Scikit-Learn** - baselines (popularity, item-KNN, SVD) e utilitarios
- **DVC** - versionamento de dados e pipeline reprodutivel
- **MLflow** - tracking de experimentos e Model Registry
- **Poetry** - gerenciamento de dependencias
- **Docker** - containerizacao multi-stage
- **Pytest / Ruff** - testes e linting

## Documentacao
- [Model Card](docs/model_card.md)
- [Roteiro do Vídeo STAR](docs/video_script.md)

## Autor
Luiz Carvalho

## Licenca
MIT (codigo). O dataset MovieLens segue a licenca do GroupLens - uso nao comercial/educacional.
