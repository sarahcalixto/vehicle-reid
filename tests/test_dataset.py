from reid.data.dataset import parse_filename

def test_parse_filename():
    assert parse_filename("0025_c016_00045735_0.jpg") == (25, 16)
    assert parse_filename("0613_c008_00062940_0.jpg") == (613, 8)
    assert parse_filename("0165_c003_00075690_0.jpg") == (165, 3)

    

import os
import pytest
from reid.data.dataset import VeRiDataset

TRAIN_DIR = "data/VeRi/image_train"
precisa_do_veri = pytest.mark.skipif(not os.path.isdir(TRAIN_DIR), reason="VeRi não baixado")


@precisa_do_veri
def test_dataset_treino_remapeia_ids_sem_buracos():
    dataset = VeRiDataset(TRAIN_DIR, relabel=True)
    ids = set(vehicle_id for _, vehicle_id, _ in dataset.samples)
    assert len(dataset) == 37778
    assert ids == set(range(576))
    assert dataset.num_ids == 576


@precisa_do_veri
def test_dataset_getitem_devolve_imagem_id_camera():
    dataset = VeRiDataset(TRAIN_DIR)
    item = dataset[0]
    assert len(item) == 3
    image, vehicle_id, camera_id = item
    assert image.mode == "RGB"
    assert (vehicle_id, camera_id) == (1, 1)  # sem relabel, mantém o ID original
