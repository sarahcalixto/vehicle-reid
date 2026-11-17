import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss (Wen et al., 2016; artigo do Luo, seção 3.6).

    Cada veículo tem um "centro" aprendido no espaço das features. A loss é a
    distância ao quadrado de cada feature até o centro do seu veículo:
    puxa as fotos do mesmo veículo para perto umas das outras.

    A triplet só olha distâncias RELATIVAS (positivo mais perto que negativo + margem);
    a center loss aperta cada grupo em termos ABSOLUTOS, o que ajuda o mAP.
    """

    def __init__(self, num_classes, feat_dim=2048):
        super().__init__()
        # os centros são parâmetros: o otimizador atualiza eles junto com a rede
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feats, labels):
        # feats: [B, D] (antes da BNNeck, a mesma da triplet), labels: [B]
        centers_batch = self.centers[labels]                       # [B, D]: centro de cada imagem
        return ((feats.float() - centers_batch) ** 2).sum(dim=1).mean()
