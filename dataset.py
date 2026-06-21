from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# pipeline to transform MNIST images to tensors and normalize them
def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_dataloaders(batch_size=64, data_dir="./data"):
    transform = get_transforms()

    # load train and test datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    print(f"Loading Training samples : {len(train_dataset):,}")
    print(f"Loading Test samples     : {len(test_dataset):,}")

    # create data loaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
