import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from cats_vs_dogs.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    DATA_URL,
    DEVICE,
    IMAGE_MEAN,
    IMAGE_SIZE_H,
    IMAGE_SIZE_W,
    IMAGE_STD,
    NUM_WORKERS,
    RESULTS_DIR,
)
from cats_vs_dogs.data_utils import load_and_unzip
from cats_vs_dogs.models import load_resnet18
from cats_vs_dogs.transforms import default_image_transforms


def main() -> None:
    if not (
        (DATA_DIR / "train_11k").exists()
        and (DATA_DIR / "test_labeled").exists()
        and (DATA_DIR / "val").exists()
    ):
        load_and_unzip(DATA_URL, DATA_DIR)
    else:
        print(f"Found data in {DATA_DIR}")

    model = load_resnet18(num_classes=2)

    image_transforms = default_image_transforms((IMAGE_SIZE_H, IMAGE_SIZE_W), IMAGE_MEAN, IMAGE_STD)

    test_dataset = ImageFolder(DATA_DIR / "test_labeled", transform=image_transforms)

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS
    )

    print(f"Using device: {DEVICE}")
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "model.pt", map_location=DEVICE))
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    accuracy = sum([1 if p == t else 0 for p, t in zip(predictions, targets)]) / len(predictions)
    print(f"Accuracy: {accuracy:.4f}")

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)
        print(f"Created results directory: {RESULTS_DIR}")
    else:
        print(f"Found results directory: {RESULTS_DIR}")

    results_filepath = RESULTS_DIR / "test_predictions.csv"
    df = pd.DataFrame({"input": test_dataset.imgs, "label": predictions})
    df.to_csv(results_filepath, index=False)
    print(f"Saved results to {results_filepath}")


if __name__ == "__main__":
    main()
