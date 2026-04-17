import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_data_loaders(batch_size=32, image_size=224, train_fraction=1.0, seed=42):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.Flowers102(
        root="data",
        split="train",
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.Flowers102(
        root="data",
        split="val",
        download=True,
        transform=eval_transform
    )

    test_dataset = datasets.Flowers102(
        root="data",
        split="test",
        download=True,
        transform=eval_transform
    )

    # Use only part of training set if needed
    if train_fraction < 1.0:
        total_size = len(train_dataset)
        subset_size = int(total_size * train_fraction)

        random.seed(seed)
        indices = list(range(total_size))
        random.shuffle(indices)

        selected_indices = indices[:subset_size]
        train_dataset = Subset(train_dataset, selected_indices)

        print(f"Using {subset_size}/{total_size} training samples "
              f"({train_fraction * 100:.0f}%)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Flowers102 class names
    class_names = datasets.Flowers102(
        root="data",
        split="train",
        download=True
    ).classes

    return train_loader, val_loader, test_loader, class_names