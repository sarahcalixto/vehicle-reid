import math

import torch
import torch.nn.functional as F

from reid.losses.ce_ls import CrossEntropyLabelSmooth
from reid.losses.triplet import TripletLoss, pairwise_distance


def test_pairwise_distance_bate_com_cdist():
    x = torch.randn(8, 16)
    assert torch.allclose(pairwise_distance(x), torch.cdist(x, x), atol=1e-4)


def test_triplet_batch_hard_calculado_a_mao():
    # veículo 0: (0,0) e (0,1)   veículo 1: (3,0) e (3,2)
    feats = torch.tensor([[0., 0.], [0., 1.], [3., 0.], [3., 2.]])
    labels = torch.tensor([0, 0, 1, 1])
    # com margem 1.5, só as âncoras 2 e 3 têm loss positiva:
    #   âncora 2: d_ap = 2, d_an = 3        -> 2 - 3 + 1.5 = 0.5
    #   âncora 3: d_ap = 2, d_an = sqrt(10) -> 3.5 - sqrt(10)
    expected = (0.5 + 3.5 - math.sqrt(10)) / 4
    loss = TripletLoss(margin=1.5)(feats, labels)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)


def test_label_smoothing_com_eps_zero_e_cross_entropy_comum():
    logits = torch.randn(4, 10)
    targets = torch.tensor([0, 3, 9, 3])
    loss = CrossEntropyLabelSmooth(num_classes=10, eps=0.0)(logits, targets)
    assert torch.allclose(loss, F.cross_entropy(logits, targets), atol=1e-5)
