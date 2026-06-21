# CIFAR-10 Classification — Atividade Prática 1

Comparação de cinco estratégias de classificação de imagens no dataset CIFAR-10 (10 classes, 60 000 imagens 32×32).

## Estrutura do repositório

```
cifar10-classification/
├── src/
│   └── train.py          # Script principal (todas as estratégias)
├── results/
│   └── resultados.csv    # Métricas exportadas automaticamente após cada execução
├── figures/              # Matrizes de confusão e curvas de validação (geradas em runtime)
├── docs/
│   └── relatorio.pdf     # Relatório final com análise dos resultados
├── requirements.txt
└── README.md
```

## Estratégias implementadas

| # | Estratégia | Backbone | Melhor acurácia |
|---|-----------|----------|-----------------|
| 1 | CNN treinada do zero | Arquitetura própria | 86,19% |
| 2a | Extração de características | MobileNetV2 + MLP | 80,81% |
| 2b | Extração de características | ResNet50 + MLP | 86,84% |
| 3 | Fine-tuning | ResNet50 | 89,91% |
| 4 | Fine-tuning + data augmentation | MobileNetV2 | **90,82%** |
| 5 | Fine-tuning (bônus) | EfficientNetV2B0 | 89,05% |

> **Melhor resultado geral:** Estratégia 4 com GAP + SGD + augmentation básica → **90,82%**

## Questões em aberto respondidas

**2a — MobileNetV2 vs ResNet50 como extrator:**
ResNet50 supera MobileNetV2 em 6,03 p.p. (86,84% vs 80,81%) como extrator de características, refletindo sua maior capacidade representacional. A penalidade é o maior custo computacional na etapa de extração.

**4a — Flatten vs GlobalMaxPooling2D:**
A substituição de `Flatten` por `GlobalMaxPooling2D` não trouxe ganho relevante (90,13% → 90,11%), indicando que, com augmentation, a estratégia de pooling tem impacto marginal.

**4b — Adam vs SGD:**
SGD com momentum (lr=1e-4, momentum=0,9) superou Adam (lr=1e-5) na fase de fine-tuning: 90,82% vs 90,11%. O SGD tende a generalizar melhor quando o modelo já está próximo de um bom mínimo.

**4c — Augmentation básica vs agressiva:**
A augmentation agressiva (rotação ±30°, zoom 20%, shear, variação de brilho) reduziu a acurácia para 89,83% frente aos 90,82% da básica. Transformações excessivas podem distorcer características relevantes em imagens de baixa resolução (96×96).

## Requisitos

```
tensorflow>=2.10
tensorflow-hub
opencv-python
scikit-learn
matplotlib
seaborn
```

Instale com:

```bash
pip install -r requirements.txt
```

Para aceleração via DirectML (AMD/Intel):

```bash
pip install tensorflow-directml-plugin
set DML_VISIBLE_DEVICES=0   # Windows
```

## Execução

```bash
cd src
python train.py
```

Os resultados são salvos automaticamente em `results/resultados.csv` e as figuras em `figures/`.

## Saídas geradas

- `results/resultados.csv` — acurácia, precisão, recall e F1 macro de cada experimento
- `figures/*.png` — matrizes de confusão de todas as estratégias
- `figures/S3_curva_validacao.png` — curva de validação da Estratégia 3 (Fase 1 + Fase 2)
