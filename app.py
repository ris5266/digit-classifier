import os
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps, ImageFilter, ImageChops

import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from model import DigitClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def list_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)), reverse=True)
    return files


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def dataset_ready():
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


def train_model(num_epochs, batch_size, lr, weight_decay):
    if not dataset_ready():
        yield "Please download the dataset first (step 1).", gr.update()
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

    # every run gets saved into the models folder with its own timestamp
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    best_acc = 0.0
    best_path = None
    log = [f"Device : {device}", "Training started...\n"]
    yield "\n".join(log), gr.update()

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

        # keep the best epoch of this run, named with epoch / accuracy / date
        if val_acc > best_acc:
            best_acc = val_acc
            new_path = os.path.join(MODELS_DIR, f"model_ep{epoch}_acc{val_acc:.2f}_{run_time}.pth")
            torch.save(model.state_dict(), new_path)
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
            best_path = new_path
            line += "  saved"

        log.append(line)
        yield "\n".join(log), gr.update()

    log.append(f"\nDone! Saved as {os.path.basename(best_path)}" if best_path else "\nDone!")
    # refresh the model picker and select the one we just trained
    selected = os.path.basename(best_path) if best_path else None
    yield "\n".join(log), gr.update(choices=list_models(), value=selected)


def evaluate_model(model_name):
    if not model_name:
        return "Select a model first (or train one in step 2).", None
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        return "That model file no longer exists. Refresh the list.", None
    if not dataset_ready():
        return "Dataset not found. Download it first (step 1).", None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = DigitClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
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


# mnist digits are cropped, scaled to 20px and centered in a 28x28 box,
# so we have to do the same to the drawing or the model gets confused
def canvas_to_mnist(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    arr = np.array(img)

    # find the pixels that were actually drawn
    coords = np.argwhere(arr > 30)
    if coords.size == 0:
        return None

    # crop to the bounding box of the digit
    (y0, x0), (y1, x1) = coords.min(0), coords.max(0)
    img = img.crop((x0, y0, x1 + 1, y1 + 1))

    # scale the longest side to 20px and keep the aspect ratio
    w, h = img.size
    if w > h:
        new_w, new_h = 20, max(1, round(h * 20 / w))
    else:
        new_w, new_h = max(1, round(w * 20 / h)), 20
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # thicken the strokes
    img = img.filter(ImageFilter.MaxFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(0.5))

    # paste it in middle of a 28x28 black canvas
    out = Image.new("L", (28, 28), 0)
    out.paste(img, ((28 - new_w) // 2, (28 - new_h) // 2))

    # shift so the center of mass sits in the middle
    arr = np.array(out, dtype=np.float32)
    ys, xs = np.nonzero(arr)
    if len(xs) > 0:
        cx = (xs * arr[ys, xs]).sum() / arr[ys, xs].sum()
        cy = (ys * arr[ys, xs]).sum() / arr[ys, xs].sum()
        out = ImageChops.offset(out, int(round(14 - cx)), int(round(14 - cy)))

    return out


def predict_drawing(drawing, model_name):
    if not model_name:
        return "Select a model first (step 3).", None
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        return "That model file no longer exists. Refresh the list.", None

    if drawing is None:
        return "", None

    img = drawing.get("composite") if isinstance(drawing, dict) else drawing
    if img is None:
        return "", None

    img = canvas_to_mnist(img)
    if img is None:
        return "", None

    transform = get_transform()
    tensor = transform(img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
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


# build gradio ui
with gr.Blocks(title="Digit Classifier") as demo:

    # download
    gr.Markdown("## Step 1: Download the dataset")
    dl_btn = gr.Button("Download MNIST", variant="primary")
    dl_status = gr.Textbox(label="Status", lines=4, interactive=False)

    # train
    gr.Markdown("## Step 2: Train the model")
    with gr.Row():
        epochs = gr.Slider(1, 20, value=10, step=1, label="Epochs")
        batch_size = gr.Slider(32, 256, value=64, step=32, label="Batch size")
    with gr.Row():
        lr = gr.Number(value=1e-3, label="Learning rate (Adam)")
        weight_decay = gr.Number(value=1e-4, label="Weight decay (L2)")
    train_btn = gr.Button("Start training", variant="primary")
    log_box = gr.Textbox(label="Training log", lines=14, interactive=False)

    # evaluate
    gr.Markdown("## Step 3: Evaluate the model")
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=list_models(), label="Trained model", scale=4)
        refresh_btn = gr.Button("Refresh list", scale=1)

    gr.Markdown("### Run it on a test set")
    gr.Markdown("Checks the model on the 10,000 test images it never saw during training.")
    eval_btn = gr.Button("Evaluate on test set", variant="primary")
    with gr.Row():
        eval_report = gr.Textbox(label="Accuracy report", lines=14, interactive=False)
        conf_matrix = gr.Plot(label="Confusion matrix")

    gr.Markdown("### Draw your own digit")
    gr.Markdown("Draw any digit (0–9) on the canvas for the model to predict.")
    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                label="Draw here",
                type="pil",
                canvas_size=(280, 280),
                brush=gr.Brush(default_size=10, colors=["#000000"], default_color="#000000", color_mode="fixed"),
                eraser=False,
                transforms=(),
                layers=False,
            )
        with gr.Column(scale=1):
            pred_label = gr.Markdown("")
            pred_chart = gr.Plot(label="Confidence per digit")

    # wiring (done here so training can refresh the model dropdown defined above)
    dl_btn.click(fn=download_dataset, outputs=dl_status)
    train_btn.click(
        fn=train_model,
        inputs=[epochs, batch_size, lr, weight_decay],
        outputs=[log_box, model_dropdown]
    )
    refresh_btn.click(fn=lambda: gr.update(choices=list_models()), outputs=model_dropdown)
    eval_btn.click(fn=evaluate_model, inputs=model_dropdown, outputs=[eval_report, conf_matrix])
    canvas.change(fn=predict_drawing, inputs=[canvas, model_dropdown], outputs=[pred_label, pred_chart])


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
