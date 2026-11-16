import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Baseline(nn.Module):
    """ResNet50 + last stride 1 + BNNeck (artigo, seções 3.4 e 3.5).

    Treino: devolve (logits, feat_antes_bn): a triplet usa a feature, a CE usa os logits.
    Avaliação: devolve feat_depois_bn, que é a usada na busca.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)

        # last stride = 1: o layer4 deixa de reduzir a resolução pela metade.
        # Mapa final 16x16 em vez de 8x8 (entrada 256x256): mais detalhe espacial,
        # custo de mais memória e computação.
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        # tudo menos avgpool e fc (os 2 últimos filhos)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        # BNNeck: a triplet trabalha com distância euclidiana (antes da BN) e a CE
        # com a direção do vetor (depois da BN). Separar evita que uma atrapalhe a outra.
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)  # artigo: sem deslocamento, a feature fica centrada

        self.classifier = nn.Linear(2048, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        feat_map = self.backbone(x)                  # [B, 2048, 16, 16]
        feat = self.gap(feat_map).flatten(1)         # [B, 2048]
        feat_bn = self.bnneck(feat)                  # [B, 2048]

        if self.training:
            logits = self.classifier(feat_bn)        # [B, num_classes]
            return logits, feat
        return feat_bn
