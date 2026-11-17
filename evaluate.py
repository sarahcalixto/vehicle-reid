import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reid.data.dataset import VeRiDataset
from reid.data.transforms import build_transforms
from reid.eval.metrics import evaluate_rank
from reid.eval.rerank import re_ranking
from reid.models.baseline import Baseline


@torch.no_grad()
def extract_features(model, pasta, flip, device):
    dataset = VeRiDataset(pasta, transform=build_transforms(train=False))
    loader = DataLoader(dataset, batch_size=128, num_workers=6)

    all_feats, all_ids, all_cams = [], [], []
    for images, vehicle_ids, camera_ids in loader:
        images = images.to(device)
        feats = model(images)
        if flip:
            # média com a imagem espelhada: feature mais estável, custa 2x o tempo
            feats = (feats + model(torch.flip(images, dims=[3]))) / 2
        all_feats.append(feats.cpu())
        all_ids.append(vehicle_ids)
        all_cams.append(camera_ids)

    return torch.cat(all_feats), torch.cat(all_ids), torch.cat(all_cams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/last.pt")
    parser.add_argument("--data", default="data/VeRi")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="mostra também o resultado com re-ranking")
    parser.add_argument("--device", default="cuda", help="cpu se a GPU estiver ocupada com um treino")
    args = parser.parse_args()
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = Baseline(ckpt["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # BN usa as médias acumuladas no treino; o modelo devolve feat_depois_bn

    q_feats, q_ids, q_cams = extract_features(model, os.path.join(args.data, "image_query"), args.flip, device)
    g_feats, g_ids, g_cams = extract_features(model, os.path.join(args.data, "image_test"), args.flip, device)

    # normalizar e usar distância euclidiana equivale a ordenar por similaridade de cosseno
    q_feats = F.normalize(q_feats, dim=1)
    g_feats = F.normalize(g_feats, dim=1)
    distmats = {"original": torch.cdist(q_feats, g_feats).numpy()}   # [Q, G]
    if args.rerank:
        distmats["re-ranking"] = re_ranking(q_feats, g_feats)

    for name, distmat in distmats.items():
        mAP, cmc = evaluate_rank(distmat, q_ids.numpy(), g_ids.numpy(),
                                 q_cams.numpy(), g_cams.numpy())
        print(f"época {ckpt['epoch'] + 1}  {name}  mAP {mAP:.1%}  Rank-1 {cmc[0]:.1%}  "
              f"Rank-5 {cmc[4]:.1%}  Rank-10 {cmc[9]:.1%}")


if __name__ == "__main__":
    main()
