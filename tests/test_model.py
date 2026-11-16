import torch

from reid.models.baseline import Baseline


def test_formatos_no_treino_e_na_avaliacao():
    model = Baseline(num_classes=10, pretrained=False)
    images = torch.randn(4, 3, 256, 256)

    model.train()
    logits, feat = model(images)
    assert logits.shape == (4, 10)
    assert feat.shape == (4, 2048)

    model.eval()
    with torch.no_grad():
        feat_bn = model(images)
    assert feat_bn.shape == (4, 2048)
