import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(x):
    """Distância euclidiana entre todas as linhas de x: [B, D] -> [B, B], sem laços.

    Usa ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    """
    sq = (x ** 2).sum(dim=1, keepdim=True)    # [B, 1]: ||a||^2 de cada linha
    dist2 = sq + sq.t() - 2 * (x @ x.t())     # [B, B]
    # clamp: erro numérico pode dar valor levemente negativo, e sqrt(0) tem gradiente infinito
    return dist2.clamp(min=1e-12).sqrt()


class TripletLoss(nn.Module):
    """Triplet batch-hard (Hermans et al., 2017).

    Para cada âncora do batch: o positivo MAIS DISTANTE e o negativo MAIS PRÓXIMO,
    ou seja, os casos mais difíceis. Loss = relu(d_ap - d_an + margem).
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats, labels):
        # float32: no AMP as features chegam em float16, e distâncias perdem precisão
        dist = pairwise_distance(feats.float())                   # [B, B]
        same_id = labels.unsqueeze(0) == labels.unsqueeze(1)      # [B, B] True = mesmo veículo

        # positivo mais distante: zera quem não é do mesmo veículo e pega o máximo da linha
        d_ap = (dist * same_id).max(dim=1).values                 # [B]
        # negativo mais próximo: põe infinito em quem é do mesmo veículo e pega o mínimo
        d_an = dist.masked_fill(same_id, float("inf")).min(dim=1).values  # [B]

        return F.relu(d_ap - d_an + self.margin).mean()
