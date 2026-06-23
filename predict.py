import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from model import DigitClassifier


def preprocess_image(image_path):

    img = Image.open(image_path).convert("L")   # grayscale

    pixels = list(img.getdata())
    avg_brightness = sum(pixels) / len(pixels)
    if avg_brightness > 127:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform(img).unsqueeze(0)


def predict(image_path, model_path="best_model.pth", show_plot=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).squeeze()

    predicted_digit = probs.argmax().item()
    confidence = probs[predicted_digit].item() * 100

    print(f"\nPredicted digit : {predicted_digit}")
    print(f"Confidence      : {confidence:.1f}%\n")
    print("All probabilities:")
    for digit, prob in enumerate(probs.tolist()):
        bar = "#" * int(prob * 40)
        marker = " <-- predicted" if digit == predicted_digit else ""
        print(f"  {digit}: {prob*100:5.1f}%  {bar}{marker}")

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

        raw_img = Image.open(image_path).convert("L").resize((28, 28))
        ax1.imshow(raw_img, cmap="gray")
        ax1.set_title(f"Input image")
        ax1.axis("off")

        colors = ["#7F77DD" if i == predicted_digit else "#D3D1C7"
                  for i in range(10)]
        ax2.barh(range(10), [p * 100 for p in probs.tolist()],
                 color=colors, edgecolor="none")
        ax2.set_yticks(range(10))
        ax2.set_xlabel("Confidence (%)")
        ax2.set_title(f"Prediction: {predicted_digit}  ({confidence:.1f}%)")
        ax2.set_xlim(0, 100)
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig("prediction_result.png", dpi=150)
        print("\nPlot saved to prediction_result.png")
        plt.show()

    return predicted_digit, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a handwritten digit")
    parser.add_argument("--image", required=True, help="Path to the image file (PNG or JPG)")
    parser.add_argument("--model", default="best_model.pth", help="Path to saved model weights")
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib visualisation")
    args = parser.parse_args()

    predict(args.image, model_path=args.model, show_plot=not args.no_plot)
