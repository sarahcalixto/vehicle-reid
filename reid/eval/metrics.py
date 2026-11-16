import numpy as np


def evaluate_rank(distmat, q_ids, g_ids, q_cams, g_cams, max_rank=10):
    """mAP e curva CMC no protocolo do VeRi.

    distmat: [Q, G] distância de cada query para cada imagem da gallery
    q_ids, q_cams: [Q]   g_ids, g_cams: [G]
    Devolve (mAP, cmc), onde cmc[0] é o Rank-1, cmc[4] o Rank-5 etc.
    """
    distmat = np.asarray(distmat)
    q_ids, g_ids = np.asarray(q_ids), np.asarray(g_ids)
    q_cams, g_cams = np.asarray(q_cams), np.asarray(g_cams)

    all_ap = []
    all_cmc = []

    for i in range(len(q_ids)):
        order = np.argsort(distmat[i])               # gallery da mais próxima à mais distante

        # descarta mesmo veículo NA MESMA câmera: seriam quase a mesma foto (mesma
        # trajetória), e o objetivo é reencontrar o veículo em OUTRA câmera.
        same_id = g_ids[order] == q_ids[i]
        same_cam = g_cams[order] == q_cams[i]
        keep = ~(same_id & same_cam)

        matches = same_id[keep]                      # True onde a lista acertou o veículo
        if not matches.any():
            continue                                 # query sem resposta possível: não entra na média

        # CMC: 1 a partir da posição do primeiro acerto
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # AP: média da precisão em cada posição onde houve acerto
        hits_so_far = matches.cumsum()
        positions = np.arange(1, len(matches) + 1)
        precision_at_hits = hits_so_far[matches] / positions[matches]
        all_ap.append(precision_at_hits.mean())

    mAP = float(np.mean(all_ap))
    cmc = np.mean(np.stack(all_cmc), axis=0)
    return mAP, cmc
