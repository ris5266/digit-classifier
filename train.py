import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import DigitClassifier

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backprop and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def train(num_epochs=10, batch_size=64, lr=1e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = DigitClassifier(dropout_rate=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    # adam with a bit of weight decay (l2) to fight overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # check how the model does on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total * 100
        print(f"Epoch {epoch}/{num_epochs}  loss {train_loss:.4f}  train {train_acc:.2f}%  val {val_acc:.2f}%")

        # keep only the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Done. Best val accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    train(num_epochs=10)
