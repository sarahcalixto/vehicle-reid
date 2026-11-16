import math
import random

from torchvision import transforms as T

# média e desvio do ImageNet: a ResNet50 pré-treinada espera imagens normalizadas assim
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomErasing:
    """Apaga um retângulo aleatório da imagem (artigo, seção 3.2).

    Simula oclusão (poste, outro carro na frente): o modelo não pode depender
    de uma única região do veículo.
    """

    def __init__(self, p=0.5, area=(0.02, 0.4), ratio=(0.3, 3.33)):
        self.p = p
        self.area = area
        self.ratio = ratio

    def __call__(self, img):
        # img: tensor [3, H, W] já normalizado
        if random.random() > self.p:
            return img

        _, h, w = img.shape
        # tenta algumas vezes achar um retângulo que caiba dentro da imagem
        for _ in range(100):
            target_area = random.uniform(*self.area) * h * w
            aspect = random.uniform(*self.ratio)
            erase_h = int(round(math.sqrt(target_area * aspect)))
            erase_w = int(round(math.sqrt(target_area / aspect)))

            if erase_h < h and erase_w < w:
                top = random.randint(0, h - erase_h)
                left = random.randint(0, w - erase_w)
                img = img.clone()
                # depois do Normalize, 0 é exatamente o pixel médio do ImageNet
                img[:, top:top + erase_h, left:left + erase_w] = 0
                return img

        return img


def build_transforms(train, size=(256, 256)):
    if train:
        return T.Compose([
            T.Resize(size),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            RandomErasing(p=0.5),
        ])

    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
