import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloaders
from model import DigitClassifier


def evaluate(model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloaders(batch_size=256)

    # load weights
    model = DigitClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # run test set
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    plot_confusion_matrix(all_labels, all_preds)


def plot_confusion_matrix(labels, preds):
    # count how often each digit got predicted
    matrix = np.zeros((10, 10), dtype=int)
    for true, pred in zip(labels, preds):
        matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion matrix")

    for i in range(10):
        for j in range(10):
            color = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    evaluate()
