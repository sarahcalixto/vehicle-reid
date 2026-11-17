import numpy as np
import torch


def re_ranking(q_feats, g_feats, k1=20, k2=6, lambda_value=0.3):
    """Re-ranking com k-reciprocal encoding (Zhong et al., 2017).

    Ideia: se B está entre os k mais próximos de A E A está entre os k mais próximos
    de B (vizinhos recíprocos), a chance de serem o mesmo veículo é bem maior do que
    só "B está perto de A". Cada imagem vira um vetor de pesos sobre seus vizinhos
    recíprocos, e a nova distância (Jaccard) mede quantos vizinhos duas imagens
    compartilham. A distância final mistura a Jaccard com a original.

    q_feats: [Q, D], g_feats: [G, D], já normalizadas.
    Devolve a distmat nova [Q, G] (numpy), no mesmo formato que o evaluate_rank espera.
    Parâmetros padrão do artigo: k1=20, k2=6, lambda=0.3.
    """
    num_q = len(q_feats)
    feats = torch.cat([q_feats, g_feats])                  # [N, D], N = Q + G
    num_all = len(feats)

    # distância euclidiana ao quadrado entre TODAS as imagens (query e gallery juntas)
    dist = torch.cdist(feats, feats).pow_(2).numpy()       # [N, N]
    dist /= dist.max(axis=1, keepdims=True)                # cada linha entre 0 e 1

    # k1+1 vizinhos mais próximos de cada imagem (o primeiro é ela mesma).
    # Em blocos de linhas: o argsort da matriz inteira de uma vez gastaria ~1,4 GB.
    initial_rank = np.concatenate([np.argsort(dist[i:i + 1000], axis=1)[:, :k1 + 1]
                                   for i in range(0, num_all, 1000)])   # [N, k1+1]
    half_k1 = int(np.around(k1 / 2))

    # V[i, j] = peso do vizinho j no "vetor de vizinhos" da imagem i (zero se não é vizinho)
    V = np.zeros((num_all, num_all), dtype=np.float16)
    for i in range(num_all):
        k_reciprocal = reciprocal_neighbors(initial_rank, i, k1)

        # expansão: junta os vizinhos recíprocos de cada vizinho, se eles concordam
        # bastante (mais de 2/3 em comum). Recupera fotos do mesmo veículo em ângulos
        # diferentes, que não estavam entre os k1 mais próximos.
        expansion = k_reciprocal
        for candidate in k_reciprocal:
            candidate_reciprocal = reciprocal_neighbors(initial_rank, candidate, half_k1)
            common = np.intersect1d(candidate_reciprocal, k_reciprocal)
            if len(common) > 2 / 3 * len(candidate_reciprocal):
                expansion = np.append(expansion, candidate_reciprocal)
        expansion = np.unique(expansion)

        # vizinho mais perto pesa mais (exp(-distância)), e os pesos somam 1
        weight = np.exp(-dist[i, expansion])
        V[i, expansion] = weight / weight.sum()

    dist = dist[:num_q]                                     # só as linhas das queries interessam daqui pra frente

    # query expansion local: o vetor de cada imagem vira a média dos seus k2 vizinhos
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(num_all):
            V_qe[i] = V[initial_rank[i, :k2]].mean(axis=0)
        V = V_qe

    # índice invertido: para cada imagem j, quem tem j como vizinho (evita varrer V inteira)
    inv_index = [np.where(V[:, j] != 0)[0] for j in range(num_all)]

    # Jaccard: 1 - (vizinhos em comum) / (união), usando os pesos de V
    jaccard = np.zeros((num_q, num_all), dtype=np.float16)
    for i in range(num_q):
        common = np.zeros(num_all, dtype=np.float32)
        for j in np.where(V[i] != 0)[0]:
            others = inv_index[j]
            common[others] += np.minimum(V[i, j], V[others, j])
        jaccard[i] = 1 - common / (2 - common)

    final = jaccard * (1 - lambda_value) + dist * lambda_value
    return final[:, num_q:]                                 # [Q, G]: só query contra gallery


def reciprocal_neighbors(initial_rank, i, k):
    """Vizinhos j de i (entre os k+1 mais próximos) que também têm i entre os seus k+1."""
    forward = initial_rank[i, :k + 1]                       # vizinhos de i
    backward = initial_rank[forward, :k + 1]                # vizinhos de cada vizinho
    is_reciprocal = (backward == i).any(axis=1)
    return forward[is_reciprocal]
