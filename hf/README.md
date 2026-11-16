---
license: other
license_name: veri-776-research-only
license_link: https://github.com/JDAI-CV/VeRidataset
library_name: pytorch
pipeline_tag: image-feature-extraction
tags:
  - vehicle-reidentification
  - re-identification
  - metric-learning
  - resnet50
datasets:
  - VeRi-776
---

# Vehicle Re-ID — ResNet50 + BNNeck (VeRi-776)

Modelo de **reidentificação de veículos**: recebe a foto de um veículo e devolve um vetor de 2048 números
(embedding). Fotos do mesmo veículo, mesmo vindas de câmeras diferentes, geram vetores próximos. Para
reencontrar um veículo, compara-se o embedding da foto (query) com os de uma galeria e ordena-se por distância.

Reimplementado do zero, em PyTorch, seguindo o *strong baseline* de
[Luo et al., 2019](https://arxiv.org/abs/1903.07071). Código, testes e histórico de commits:
[github.com/sarahcalixto/vehicle-reid](https://github.com/sarahcalixto/vehicle-reid).

## Resultados (VeRi-776, test set)

| mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|
| X% | X% | X% | X% |

Protocolo do VeRi: 1.678 queries contra 11.579 imagens de galeria, 200 veículos que o modelo nunca viu
no treino. Imagens do mesmo veículo **na mesma câmera** da query são descartadas da galeria.
Inferência com feature após a BNNeck, média com a imagem espelhada (flip) e distância euclidiana sobre
features normalizadas (equivalente a cosseno).

## Arquitetura

```
imagem 256x256 -> ResNet50 (ImageNet, last stride = 1) -> GAP -> feature [2048]
                                                                   |-> triplet loss (batch-hard)
                                                                   v
                                                        BatchNorm1d (BNNeck)
                                                                   |-> feature de inferência
                                                                   v
                                                        Linear (576 IDs) -> CE com label smoothing
```

- **Last stride = 1:** o último estágio da ResNet não reduz a resolução (mapa 16x16 em vez de 8x8).
- **BNNeck:** a triplet usa a feature antes da BN; a cross-entropy usa a de depois. Na inferência, a de depois.
- **Losses:** triplet batch-hard (margem 0,3) + cross-entropy com label smoothing (eps = 0,1).

## Treino

| | |
|---|---|
| Dados | VeRi-776 train: 37.778 imagens, 576 veículos, 20 câmeras |
| Batch | P x K = 16 veículos x 4 imagens |
| Augmentation | resize 256x256, flip horizontal, pad 10 + random crop, random erasing (p = 0,5) |
| Otimizador | Adam, lr 3,5e-4, weight decay 5e-4 |
| Agenda | warmup linear de 10 épocas (3,5e-5 -> 3,5e-4), x0,1 nas épocas 30 e 50, 60 épocas |
| Precisão | mixed precision (autocast float16 + GradScaler) |
| Hardware | 1x RTX 4050 notebook (6 GB), ~X h |

Diferença em relação ao artigo: 60 épocas em vez de 120 (degraus 30/50 em vez de 40/70), por limite de tempo.
Center loss não foi usada.

## Como usar

```python
import torch
from reid.models.baseline import Baseline          # do repositório no GitHub
from reid.data.transforms import build_transforms
from PIL import Image

model = Baseline(num_classes=576, pretrained=False)
model.load_state_dict(torch.load("vehicle_reid_baseline.pt", map_location="cpu"))
model.eval()

transform = build_transforms(train=False)
image = transform(Image.open("carro.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    embedding = torch.nn.functional.normalize(model(image), dim=1)   # [1, 2048]
```

## Limitações

- Treinado e avaliado **só no VeRi-776**: 20 câmeras de vigilância de uma única região, de dia. O desempenho
  em outras cidades, câmeras, resoluções, à noite ou com chuva não foi medido e tende a cair.
- Sensível a mudanças grandes de ângulo (frente x traseira) e de iluminação.
- Veículos do mesmo modelo e cor são a principal fonte de erro: o modelo não lê placas.

## Uso e licença

- O dataset VeRi-776 é liberado **apenas para pesquisa não comercial**, e estes pesos herdam essa restrição.
- Reidentificação de veículos pode ser usada para rastrear pessoas. Este modelo é um projeto de estudo e
  não deve ser usado para vigilância de indivíduos nem em decisões que afetem pessoas sem revisão humana.

## Citações

```bibtex
@inproceedings{luo2019bag,
  title     = {Bag of Tricks and A Strong Baseline for Deep Person Re-identification},
  author    = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  booktitle = {CVPR Workshops},
  year      = {2019}
}

@inproceedings{liu2016deep,
  title     = {A Deep Learning-Based Approach to Progressive Vehicle Re-identification for Urban Surveillance},
  author    = {Liu, Xinchen and Liu, Wu and Mei, Tao and Ma, Huadong},
  booktitle = {ECCV},
  year      = {2016}
}

@inproceedings{liu2016large,
  title     = {Large-scale vehicle re-identification in urban surveillance videos},
  author    = {Liu, Xinchen and Liu, Wu and Ma, Huadong and Fu, Huiyuan},
  booktitle = {ICME},
  year      = {2016}
}

@article{liu2018provid,
  title   = {PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance},
  author  = {Liu, Xinchen and Liu, Wu and Mei, Tao and Ma, Huadong},
  journal = {IEEE Transactions on Multimedia},
  volume  = {20},
  number  = {3},
  pages   = {645--658},
  year    = {2018}
}
```
