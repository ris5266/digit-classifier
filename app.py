import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from model import DigitClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_model.pth")


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def dataset_ready():
    # true if the mnist files are already on disk
    try:
        datasets.MNIST(DATA_DIR, train=True, download=False)
        datasets.MNIST(DATA_DIR, train=False, download=False)
        return True
    except Exception:
        return False


def download_dataset():
    try:
        train_ds = datasets.MNIST(DATA_DIR, train=True, download=True)
        test_ds = datasets.MNIST(DATA_DIR, train=False, download=True)
        return (
            f"Dataset ready!\n\n"
            f"Training samples : {len(train_ds):,}\n"
            f"Test samples : {len(test_ds):,}\n"
        )
    except Exception as e:
        return f"Download failed: {e}"


# this is a generator so the log streams into the ui live, epoch by epoch
def train_model(num_epochs, batch_size, lr, weight_decay):
    if not dataset_ready():
        yield "Please download the dataset first (step 1)."
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()

    train_ds = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = DigitClassifier(dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    log = [f"Device : {device}", "Training started...\n"]
    yield "\n".join(log)

    for epoch in range(1, int(num_epochs) + 1):
        # train for one epoch
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        train_loss = total_loss / len(train_loader)

        # validate on the test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100

        line = f"Epoch {epoch:2d}/{int(num_epochs)}  loss {train_loss:.4f}  train {train_acc:.1f}%  val {val_acc:.1f}%"

        # save the model whenever it gets better
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            line += "  saved"

        log.append(line)
        yield "\n".join(log)

    log.append(f"\nDone!")
    yield "\n".join(log)


def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        return "No trained model found. Train it first (step 2).", None
    if not dataset_ready():
        return "Dataset not found. Download it first (step 1).", None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = DigitClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall = (all_preds == all_labels).mean() * 100

    # accuracy for each digit separately
    lines = [f"Overall accuracy: {overall:.2f}%\n", "Per-digit accuracy:"]
    for d in range(10):
        mask = all_labels == d
        acc = (all_preds[mask] == all_labels[mask]).mean() * 100
        bar = "█" * int(acc / 5)
        lines.append(f"  {d}: {acc:5.2f}%  {bar}")
    report = "\n".join(lines)

    # build the confusion matrix
    matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_labels, all_preds):
        matrix[t][p] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    for i in range(10):
        for j in range(10):
            color = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7, color=color)
    plt.tight_layout()

    return report, fig


def predict_drawing(drawing):
    if not os.path.exists(MODEL_PATH):
        return "No trained model found. Train it first (step 2).", None

    if drawing is None:
        return "Draw a digit on the canvas.", None

    img = drawing.get("composite") if isinstance(drawing, dict) else drawing
    if img is None:
        return "Draw a digit on the canvas.", None

    # canvas is black on white, so invert it to match mnist
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)

    transform = get_transform()
    tensor = transform(img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)), dim=1).squeeze().cpu().tolist()

    pred = int(np.argmax(probs))
    confidence = probs[pred] * 100
    label = f"Predicted: **{pred}**  ({confidence:.1f}% confident)"

    # bar chart of the confidence per digit
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = []
    for i in range(10):
        if i == pred:
            colors.append("#6C63FF")
        else:
            colors.append("#D3D1C7")
    ax.barh(range(10), [p * 100 for p in probs], color=colors, edgecolor="none")
    ax.set_yticks(range(10))
    ax.set_xlabel("Confidence (%)")
    ax.set_title(f"Prediction: {pred}  ({confidence:.1f}%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    plt.tight_layout()

    return label, fig


# build the gradio interface, everything on one page top to bottom
with gr.Blocks(title="Digit Classifier", theme=gr.themes.Soft()) as demo:

    # step 1 - download
    gr.Markdown("## Step 1: Download the dataset")
    dl_btn = gr.Button("Download MNIST", variant="primary")
    dl_status = gr.Textbox(label="Status", lines=4, interactive=False)
    dl_btn.click(fn=download_dataset, outputs=dl_status)

    # step 2 - train
    gr.Markdown("## Step 2: Train the model")
    with gr.Row():
        epochs = gr.Slider(1, 20, value=10, step=1, label="Epochs")
        batch_size = gr.Slider(32, 256, value=64, step=32, label="Batch size")
    with gr.Row():
        lr = gr.Number(value=1e-3, label="Learning rate (Adam)")
        weight_decay = gr.Number(value=1e-4, label="Weight decay (L2)")
    train_btn = gr.Button("Start training", variant="primary")
    log_box = gr.Textbox(label="Training log", lines=14, interactive=False)
    train_btn.click(
        fn=train_model,
        inputs=[epochs, batch_size, lr, weight_decay],
        outputs=log_box
    )

    # step 3 - evaluate (two ways)
    gr.Markdown("## Step 3: Evaluate the model")

    gr.Markdown("### Run it on a test set")
    gr.Markdown("Checks the model on the 10,000 test images it never saw during training.")
    eval_btn = gr.Button("Evaluate on test set", variant="primary")
    with gr.Row():
        eval_report = gr.Textbox(label="Accuracy report", lines=14, interactive=False)
        conf_matrix = gr.Plot(label="Confusion matrix")
    eval_btn.click(fn=evaluate_model, outputs=[eval_report, conf_matrix])

    gr.Markdown("### Draw your own digit")
    gr.Markdown("Draw any digit (0–9) on the canvas. It predicts when you release the mouse.")
    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(label="Draw here", type="pil", canvas_size=(280, 280))
        with gr.Column(scale=1):
            pred_label = gr.Markdown("Draw a digit to see the prediction.")
            pred_chart = gr.Plot(label="Confidence per digit")
    canvas.change(fn=predict_drawing, inputs=canvas, outputs=[pred_label, pred_chart])


if __name__ == "__main__":
    demo.launch(share=False)
