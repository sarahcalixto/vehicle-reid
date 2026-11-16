import numpy as np

from reid.eval.metrics import evaluate_rank


def test_exemplo_de_uma_query_feito_a_mao():
    # query: veículo 1, câmera 1. Gallery já em ordem de distância.
    distmat = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    g_ids = [1, 2, 1, 3, 1]
    g_cams = [1, 2, 2, 2, 3]
    # a 1ª da gallery é mesmo veículo e mesma câmera -> descartada.
    # Sobra: [veíc. 2, veíc. 1, veíc. 3, veíc. 1] -> acertos nas posições 2 e 4.
    # AP = (1/2 + 2/4) / 2 = 0.5; Rank-1 = 0 (errou a 1ª); Rank-2 = 1.
    mAP, cmc = evaluate_rank(distmat, [1], g_ids, [1], g_cams, max_rank=5)
    assert np.isclose(mAP, 0.5)
    assert cmc[0] == 0
    assert cmc[1] == 1
