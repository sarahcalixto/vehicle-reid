from torch.utils.data import Dataset
from PIL import Image
import os

def parse_filename(filename):

    parts = filename.split('_')
    vehicle_id = int(parts[0])
    camera_id = int(parts[1][1:])

    return (vehicle_id, camera_id)

class VeRiDataset(Dataset):

    def __init__(self, pasta, transform=None, relabel=False):

        files = []
        pasta_sorted = sorted(os.listdir(pasta))

        for file in pasta_sorted:
            vehicle_id, camera_id = parse_filename(file)
            img_path = os.path.join(pasta, file)
            img_info = (img_path, vehicle_id, camera_id)
            files.append(img_info)

        # treino: o classificador precisa de rótulos 0..N-1 sem buracos.
        # query/gallery mantêm os IDs originais (só comparamos uma com a outra).
        if relabel:
            ids_unicos = sorted(set(vehicle_id for _, vehicle_id, _ in files))
            id_para_rotulo = {}
            for rotulo, vehicle_id in enumerate(ids_unicos):
                id_para_rotulo[vehicle_id] = rotulo

            relabeled = []
            for img_path, vehicle_id, camera_id in files:
                relabeled.append((img_path, id_para_rotulo[vehicle_id], camera_id))
            files = relabeled

        self.samples = files
        self.transform = transform
        self.num_ids = len(set(vehicle_id for _, vehicle_id, _ in files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, vehicle_id, camera_id = self.samples[idx]
        # convert("RGB") garante 3 canais mesmo se a imagem vier em cinza ou com alfa
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, vehicle_id, camera_id
