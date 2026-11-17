import numpy as np
import torch
import torch.nn.functional as F

from reid.eval.rerank import re_ranking


def test_lambda_1_mantem_a_ordem_original():
    # com lambda=1 a Jaccard some e sobra só a distância original: a ordem não muda
    q = F.normalize(torch.randn(5, 8), dim=1)
    g = F.normalize(torch.randn(30, 8), dim=1)
    final = re_ranking(q, g, k1=4, k2=2, lambda_value=1.0)
    assert final.shape == (5, 30)
    original = torch.cdist(q, g).numpy()
    assert (np.argsort(final, axis=1) == np.argsort(original, axis=1)).all()


def test_vizinho_reciproco_vence():
    # dois grupos bem separados: depois do re-ranking, a gallery mais próxima
    # de cada query continua sendo do grupo dela
    torch.manual_seed(0)
    centro_a, centro_b = torch.randn(8), torch.randn(8)
    grupo_a = centro_a + 0.05 * torch.randn(10, 8)
    grupo_b = centro_b + 0.05 * torch.randn(10, 8)
    q = F.normalize(torch.stack([grupo_a[0], grupo_b[0]]), dim=1)
    g = F.normalize(torch.cat([grupo_a[1:], grupo_b[1:]]), dim=1)   # 9 do A, depois 9 do B
    final = re_ranking(q, g, k1=5, k2=3, lambda_value=0.3)
    assert final[0].argmin() < 9
    assert final[1].argmin() >= 9
