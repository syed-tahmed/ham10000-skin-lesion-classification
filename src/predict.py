"""
Single-image inference for the HAM10000 skin lesion classifier.

Usage:
    python src/predict.py --image path/to/lesion.jpg --model models/exp4_resnet18_sched.pt

Outputs the predicted class with per-class probabilities, and flags the
image for review if melanoma probability exceeds the triage threshold.
"""

import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_NAMES = {
    'akiec': 'Actinic keratoses / intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions',
}

IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Chosen from the threshold sweep in section 8.3 - favours recall over
# precision, appropriate for triage rather than diagnosis.
MEL_THRESHOLD = 0.20


def load_model(checkpoint_path, device):
    """Rebuild the ResNet18 architecture and load trained weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model.to(device)


def preprocess(image_path):
    """Apply the same deterministic transform used for validation and test."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, tensor, device):
    with torch.no_grad():
        probs = torch.softmax(model(tensor.to(device)), dim=1)
    return probs.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the lesion image")
    parser.add_argument("--model", required=True, help="Path to the .pt checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)
    probs = predict(model, preprocess(args.image), device)

    top = probs.argmax()
    print(f"\nPrediction: {CLASSES[top]} - {CLASS_NAMES[CLASSES[top]]}")
    print(f"Confidence: {probs[top]:.3f}\n")

    print("All classes:")
    for i in probs.argsort()[::-1]:
        bar = "#" * int(probs[i] * 40)
        print(f"  {CLASSES[i]:<6} {probs[i]:.3f}  {bar}")

    mel_prob = probs[CLASSES.index('mel')]
    print()
    if mel_prob >= MEL_THRESHOLD:
        print(f"FLAGGED FOR REVIEW - melanoma probability {mel_prob:.3f} "
              f"exceeds threshold {MEL_THRESHOLD}")
    else:
        print(f"Not flagged - melanoma probability {mel_prob:.3f} "
              f"below threshold {MEL_THRESHOLD}")

    print("\nThis is an academic project, not a medical device. "
          "It must not be used for diagnosis.")


if __name__ == "__main__":
    main()
