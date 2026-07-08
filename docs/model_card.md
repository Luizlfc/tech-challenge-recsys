# Model Card - Sistema de Recomendacao (MovieLens)

## Finalidade do Modelo
Recomendar top-K itens (filmes, como proxy de "produtos") a um usuario com base no seu
historico de interacoes positivas, simulando um sistema de recomendacao de e-commerce
baseado em comportamento de navegacao/interacao.

## Dataset
MovieLens `ml-latest-small` (GroupLens Research) - ~100.836 avaliacoes explicitas de
610 usuarios sobre 9.742 filmes. As avaliacoes sao transformadas em **interacoes
implicitas**: uma avaliacao com `rating >= 4.0` e tratada como uma interacao positiva
("o usuario engajou fortemente com este item"), simulando um clique/compra/adicionar-ao-
carrinho em um catalogo de produtos. Apos filtrar usuarios/itens com menos de 5
interacoes positivas, restam 601 usuarios e 1.956 itens.

Uso do dataset restrito a fins nao comerciais/educacionais, conforme licenca do GroupLens.

## Arquitetura e Baselines
- **MLP (NeuralCF-style)**: embeddings de usuario/item concatenados, passados por uma
  torre `Linear -> ReLU -> Dropout` (`[128, 64, 32]`), com saida de 1 logit
  (`BCEWithLogitsLoss`). Treinado com negative sampling (4 negativos por positivo,
  pre-computado no `feature_eng`) e early stopping por val loss.
- **Popularity**: baseline nao-personalizado, ranking pela contagem global de interacoes.
- **Item-KNN**: similaridade de cosseno item-item (`sklearn.neighbors.NearestNeighbors`)
  sobre a matriz esparsa usuario-item.
- **SVD (matrix factorization)**: `sklearn.decomposition.TruncatedSVD` sobre a mesma
  matriz esparsa, score via produto interno de fatores latentes.

## Metricas de Performance (test set, protocolo leave-one-out)

| Modelo | Precision@10 | Recall@10 (=HitRate@10) | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| Popularity | 0.0043 | 0.0433 | 0.0229 | 0.0166 |
| Item-KNN | 0.0062 | 0.0616 | 0.0288 | 0.0191 |
| **SVD (vencedor, Production)** | **0.0083** | **0.0832** | **0.0414** | **0.0290** |
| MLP (NeuralCF) | 0.0045 | 0.0449 | 0.0223 | 0.0154 |

Tabela completa (todos os `k` em `[5, 10, 20]`) em `metrics/eval_table.csv`, gerada pelo
`dvc repro` e versionada como artefato no MLflow.

O modelo **SVD** venceu a comparacao pelo `promotion_metric` (`ndcg@10`) configurado em
`params.yaml` e foi automaticamente registrado no MLflow Model Registry, promovido a
`Staging` e, por superar o `promotion_threshold` (0.03), promovido a `Production`.

## Limitacoes
- Dataset pequeno (~100k interacoes, ~600 usuarios) e de dominio de filmes, nao de um
  catalogo de e-commerce real - serve como prova de conceito da arquitetura/pipeline,
  nao como um modelo pronto para producao de e-commerce.
- Neste cenario leave-one-out (1 item relevante por usuario), `Recall@K` e
  matematicamente identico a `HitRate@K` - nao sao duas fontes de sinal independentes,
  apenas duas formulas convergindo no mesmo caso extremo.
- O MLP (NeuralCF) nao superou o baseline SVD neste dataset pequeno - resultado
  esperado e documentado na literatura: redes neurais de recomendacao tendem a precisar
  de mais dados/interacoes para superar fatoracao de matriz bem ajustada. O pipeline
  compara os modelos de forma justa e transparente, sem favorecer o MLP artificialmente.

## Riscos e Vieses
- **Vies de popularidade**: itens muito avaliados dominam recomendacoes nao-
  personalizadas (baseline Popularity) e podem enviesar negative sampling se a
  amostragem for trocada para popularity-weighted no futuro.
- **Cold-start**: usuarios/itens novos (sem interacoes no treino) nao tem
  embedding/vizinhos aprendidos - nenhum dos 4 modelos trata cold-start explicitamente.
- **Escolha do limiar de positivo** (`rating >= 4.0`): uma escolha de design que trata
  "gostou muito" como interacao; um limiar mais baixo mudaria a distribuicao de
  positivos/negativos e os resultados.

## Uso Recomendado
Prova de conceito / material de estudo para pipeline de recomendacao ponta-a-ponta
(PyTorch + baselines + DVC + MLflow Registry). Base para experimentacao com datasets
de e-commerce reais (RetailRocket, Instacart) trocando apenas a stage `preprocess`.

## Uso Nao Recomendado
Uso direto em producao de e-commerce sem retrain em dados de comportamento real de
navegacao/compra, e sem tratamento de cold-start.
