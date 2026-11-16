import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """Cross-entropy com label smoothing (artigo, seção 3.3).

    Em vez do alvo "100% veículo y", usa 1 - eps + eps/N para y e eps/N para os outros.
    Evita que o modelo fique confiante demais nos 576 IDs do treino (overfitting),
    já que no teste os veículos são outros.
    """

    def __init__(self, num_classes, eps=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        # logits: [B, N], targets: [B] com rótulos 0..N-1
        log_probs = F.log_softmax(logits.float(), dim=1)                  # [B, N]

        smooth_targets = torch.full_like(log_probs, self.eps / self.num_classes)
        # na coluna do rótulo certo, coloca o valor maior
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.eps + self.eps / self.num_classes)

        return (-smooth_targets * log_probs).sum(dim=1).mean()
