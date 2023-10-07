from typing import Tuple

from torchvision import transforms


def default_image_transforms(
    image_size: Tuple[int, int], mean: Tuple[float, float, float], std: Tuple[float, float, float]
):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
