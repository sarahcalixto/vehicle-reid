import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from reid.data.dataset import VeRiDataset
from reid.data.sampler import PKSampler
from reid.data.transforms import build_transforms
from reid.losses.ce_ls import CrossEntropyLabelSmooth
from reid.losses.triplet import TripletLoss
from reid.models.baseline import Baseline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/VeRi")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps", type=int, nargs="+", default=[40, 70],
                        help="épocas em que a lr é multiplicada por 0.1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--p", type=int, default=16, help="veículos por batch")
    parser.add_argument("--k", type=int, default=4, help="imagens por veículo")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--subset", type=int, default=0,
                        help="usa só os N primeiros veículos (treinos rápidos / overfit)")
    parser.add_argument("--max-iters", type=int, default=0,
                        help="para depois de N iterações (teste de memória)")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def lr_factor(epoch, warmup, steps):
    """Multiplicador da lr na época `epoch` (artigo, seção 3.1).

    Warmup: sobe linear de 0.1x até 1x nas primeiras épocas (3.5e-5 -> 3.5e-4),
    porque no começo os pesos novos (BNNeck, classificador) são aleatórios e uma lr
    alta estragaria os pesos pré-treinados. Depois cai 10x em cada degrau.
    """
    if epoch < warmup:
        return 0.1 + 0.9 * epoch / warmup
    factor = 1.0
    for step in steps:
        if epoch >= step:
            factor *= 0.1
    return factor


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    dataset = VeRiDataset(os.path.join(args.data, "image_train"),
                          transform=build_transforms(train=True), relabel=True)
    num_classes = dataset.num_ids
    if args.subset:
        # rótulos já são 0..N-1 em ordem, então "< subset" continua sem buracos
        dataset.samples = [s for s in dataset.samples if s[1] < args.subset]
        num_classes = args.subset

    labels = [vehicle_id for _, vehicle_id, _ in dataset.samples]
    sampler = PKSampler(labels, p=args.p, k=args.k)
    loader = DataLoader(dataset, batch_size=args.p * args.k, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = Baseline(num_classes).to(device)
    ce_loss = CrossEntropyLabelSmooth(num_classes)
    triplet_loss = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_factor(epoch, args.warmup, args.steps))
    # mixed precision: float16 onde é seguro (metade da memória, tensor cores).
    # O GradScaler multiplica a loss antes do backward para gradientes pequenos
    # não virarem zero em float16 (underflow).
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"retomando da época {start_epoch}")

    print(f"{len(dataset)} imagens, {num_classes} veículos, {len(loader)} iterações/época")
    iters = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        sum_loss, sum_acc, n = 0.0, 0.0, 0

        for images, vehicle_ids, _ in loader:
            images = images.to(device, non_blocking=True)
            vehicle_ids = vehicle_ids.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16):
                logits, feats = model(images)
            loss = ce_loss(logits, vehicle_ids) + triplet_loss(feats, vehicle_ids)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            sum_loss += loss.item()
            sum_acc += (logits.argmax(dim=1) == vehicle_ids).float().mean().item()
            n += 1
            iters += 1
            if args.max_iters and iters >= args.max_iters:
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"{iters} iterações, loss {loss.item():.3f}, pico de memória {peak:.2f} GB")
                return

        scheduler.step()
        print(f"época {epoch + 1}/{args.epochs}  loss {sum_loss / n:.4f}  acc {sum_acc / n:.3f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s", flush=True)

        # checkpoint a cada época, para retomar com --resume se travar
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "num_classes": num_classes}, last_path)


if __name__ == "__main__":
    main()
