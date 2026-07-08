# Roteiro do Vídeo STAR (~5 minutos)

Sugestão de tela para cada bloco entre colchetes. Ajuste o texto com sua própria voz -
isto é um guia, não um texto para decorar.

## Situation (~45s)
[Tela: README.md do repositório]

"Este projeto e o Tech Challenge Fase 2, cujo objetivo e construir um sistema de
recomendacao de produtos para e-commerce baseado no comportamento de navegacao dos
usuarios. Como nao tinha acesso imediato a um dataset de e-commerce publico com volume
suficiente de interacoes sem credenciais extras, usei o MovieLens `ml-latest-small` como
proxy: cerca de 100 mil avaliacoes de filmes, tratando avaliacoes altas (nota >= 4) como
'interacoes positivas' — o equivalente a um clique ou compra em um catalogo de produtos.
Isso preserva a mesma estrutura usuario-item-interacao de um sistema de recomendacao
real, so trocando o dominio."

## Task (~45s)
[Tela: estrutura de pastas do projeto no VS Code / terminal `tree`]

"O desafio tecnico era construir um pipeline completo e profissional: um modelo de rede
neural em PyTorch para recomendacao, comparado com baselines de Scikit-Learn usando pelo
menos 4 metricas; tudo isso versionado com DVC em um pipeline de pelo menos 3 stages;
experimentos rastreados no MLflow, com o melhor modelo promovido no Model Registry; e
tudo containerizado em Docker com um Dockerfile multi-stage. Alem disso, o codigo
precisava seguir boas praticas: SOLID, type hints, pelo menos um design pattern, e
gerenciamento de dependencias moderno com Poetry."

## Action (~2min30s)
[Tela: src/recsys/models/base.py, factory.py — depois dvc.yaml — depois MLflow UI]

"Para a arquitetura, usei tres design patterns. Primeiro, **Strategy**: todo modelo
implementa a interface `RecommenderStrategy`, com metodos `fit`, `score_items` e um
`recommend_top_k` ja implementado na propria interface. Isso significa que o pipeline de
treino e avaliacao nunca precisa saber se esta lidando com o MLP ou com um baseline.
Segundo, **Factory**: o `ModelFactory` cria a estrategia certa a partir de um nome
configurado no `params.yaml` — popularity, item-KNN, SVD ou MLP — sem o codigo de treino
precisar importar cada classe concreta. Terceiro, **Template Method**: a classe
`Trainer` fixa a sequencia fit-avalia, compartilhada por todos os modelos.

O modelo central e uma rede neural estilo NeuralCF: embeddings de usuario e item
concatenados e passados por uma torre MLP com dropout, treinada com negative sampling
- ja que so temos interacoes positivas no dataset - e early stopping baseado na loss de
validacao. Comparei esse MLP com tres baselines do Scikit-Learn: um baseline de
popularidade, um item-KNN por similaridade de cosseno, e uma fatoracao de matriz via
TruncatedSVD.

Para versionamento, montei um pipeline DVC de 4 stages: preprocess, que limpa e binariza
as interacoes; feature_eng, que gera os indices de usuario/item, faz o split
leave-one-out por timestamp e o negative sampling; train, que treina os 4 modelos; e
evaluate, que calcula as metricas e decide o vencedor. Rodar `dvc repro` reproduz o
pipeline inteiro do zero, e o `dvc.lock` garante que qualquer mudanca em codigo ou
parametro dispare so as stages necessarias.

Cada modelo treinado vira um run no MLflow, com parametros e metricas de validacao. Na
stage de avaliacao, comparo os 4 modelos no conjunto de teste usando 5 metricas de
ranking que implementei do zero: Precision@K, Recall@K, NDCG@K, HitRate@K e MRR@K. O
modelo vencedor - neste caso o SVD - e registrado automaticamente no MLflow Model
Registry, promovido para Staging, e depois para Production se ultrapassar um limiar de
qualidade configuravel.

Por fim, todo o pipeline roda em um Dockerfile multi-stage: um estagio 'builder' que
instala as dependencias com Poetry, e um estagio 'runtime' enxuto, sem toolchain de
build nem dependencias de desenvolvimento, rodando como usuario nao-root. O
docker-compose sobe um servidor MLflow e o servico de treino, que baixa o dataset e
executa `dvc repro` dentro do container."

## Result (~1min)
[Tela: metrics/eval_table.csv ou MLflow UI com a tabela de comparacao]

"O resultado final: o SVD venceu a comparacao com NDCG@10 de 0.041, superando o item-KNN
(0.029), o baseline de popularidade (0.023) e, de forma honesta, tambem o MLP (0.022).
Essa e uma licao real e documentada no Model Card: em um dataset pequeno como este, com
so 600 usuarios, redes neurais de recomendacao tendem a precisar de mais dados para
superar uma fatoracao de matriz bem ajustada — o pipeline nao favorece o MLP
artificialmente, ele so venceria se realmente performasse melhor.

O principal trade-off do projeto foi escolher o MovieLens como proxy de e-commerce: isso
permitiu focar na engenharia do pipeline (DVC, MLflow, Docker, design patterns) sem
depender de credenciais de dataset externas, mas significa que o modelo nao esta pronto
para producao real sem retrain em dados de navegacao de e-commerce de verdade. A
principal licao tecnica foi a importancia de um pipeline reprodutivel de ponta a ponta:
consegui rodar `dvc repro` do zero varias vezes durante o desenvolvimento e obter
exatamente os mesmos resultados, o que da confianca para evoluir o modelo sem quebrar
nada."

---

**Checklist antes de gravar:**
- [ ] Testar `docker compose up --build` uma vez para ter a MLflow UI aberta durante a gravacao
- [ ] Deixar `metrics/eval_table.csv` ou a MLflow UI já abertos em uma aba
- [ ] Cronometrar: ~45s + 45s + 150s + 60s = 5min
