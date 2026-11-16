from collections import Counter

from reid.data.sampler import PKSampler


def test_cada_batch_tem_p_ids_com_k_imagens():
    # 20 veículos; o veículo 0 tem só 2 imagens (força a reposição)
    labels = [0, 0] + [vehicle_id for vehicle_id in range(1, 20) for _ in range(10)]
    sampler = PKSampler(labels, p=16, k=4)
    indices = list(sampler)

    assert len(indices) == len(sampler)
    for start in range(0, len(indices), 64):
        batch = indices[start:start + 64]
        counts = Counter(labels[idx] for idx in batch)
        assert len(batch) == 64
        assert len(counts) == 16
        assert all(count == 4 for count in counts.values())
