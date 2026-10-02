import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class SyntheticImageDataset(Dataset):
    """
    Generates synthetic 3-channel RGB image tensors (C, H, W).
    Applies standard torchvision augmentation pipelines.
    """

    def __init__(self, num_samples: int = 120, transform=None):
        torch.manual_seed(42)
        # Random RGB images of size 3x32x32 in float range [0, 1]
        self.images = torch.rand(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, 4, (num_samples,))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def build_augmentation_pipelines():
    # Production train-time augmentations
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(size=32, padding=4, padding_mode="reflect"),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation/Inference deterministic pipeline
    val_transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def inspect_pipeline():
    train_tf, val_tf = build_augmentation_pipelines()

    train_dataset = SyntheticImageDataset(num_samples=100, transform=train_tf)
    val_dataset = SyntheticImageDataset(num_samples=40, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print("--- Vision Transform & Augmentation Pipeline ---")
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size:   {len(val_dataset)}")

    for batch_idx, (batch_imgs, batch_lbls) in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1} Shapes:")
        print(f"  Images: {tuple(batch_imgs.shape)} | Mean: {batch_imgs.mean():.4f} | Std: {batch_imgs.std():.4f}")
        print(f"  Labels: {tuple(batch_lbls.shape)}")
        break


if __name__ == "__main__":
    inspect_pipeline()