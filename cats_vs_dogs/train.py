import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from cats_vs_dogs.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    DATA_URL,
    DEVICE,
    EPOCH_NUM,
    IMAGE_MEAN,
    IMAGE_SIZE_H,
    IMAGE_SIZE_W,
    IMAGE_STD,
    LR,
    NUM_WORKERS,
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

    train_dataset = ImageFolder(DATA_DIR / "train_11k", transform=image_transforms)
    val_dataset = ImageFolder(DATA_DIR / "val", transform=image_transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    if not CHECKPOINT_DIR.exists():
        CHECKPOINT_DIR.mkdir(parents=True)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")
    else:
        print(f"Found checkpoint directory: {CHECKPOINT_DIR}")

    print(f"Using device: {DEVICE}")
    model = model.to(DEVICE)

    print("Starting training...")
    for epoch in range(EPOCH_NUM):
        model.train()
        train_losses = []
        for data, target in tqdm(train_dataloader, desc=f"TRAIN epoch {epoch + 1}"):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(val_dataloader, desc=f"VAL epoch {epoch + 1}"):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f"Epoch: {epoch + 1}, mean loss: {sum(train_losses) / len(train_losses)}")
        print(f"Validation accuracy: {correct / total}\n")

    print(f"Saving model to {CHECKPOINT_DIR / 'model.pt'}...")
    torch.save(model.state_dict(), CHECKPOINT_DIR / "model.pt")


if __name__ == "__main__":
    main()
