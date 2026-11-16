import random

from torch.utils.data import Sampler


class PKSampler(Sampler):
    """Monta batches com P veículos e K imagens de cada (P*K índices por batch).

    A triplet batch-hard precisa disso: sem pelo menos 2 imagens do mesmo veículo
    no batch não existe par positivo.
    """

    def __init__(self, labels, p=16, k=4):
        self.p = p
        self.k = k

        # para cada veículo, a lista de posições (índices) das suas imagens no dataset
        self.index_by_id = {}
        for idx, vehicle_id in enumerate(labels):
            if vehicle_id not in self.index_by_id:
                self.index_by_id[vehicle_id] = []
            self.index_by_id[vehicle_id].append(idx)

        self.ids = list(self.index_by_id.keys())
        # uma "época" vê mais ou menos o número de imagens do dataset
        self.num_batches = len(labels) // (p * k)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_ids = random.sample(self.ids, self.p)
            for vehicle_id in batch_ids:
                indices = self.index_by_id[vehicle_id]
                if len(indices) >= self.k:
                    chosen = random.sample(indices, self.k)
                else:
                    # veículo com menos de K imagens: repete imagens (com reposição)
                    chosen = random.choices(indices, k=self.k)
                for idx in chosen:
                    yield idx

    def __len__(self):
        return self.num_batches * self.p * self.k
