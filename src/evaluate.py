import os
import time
import json
import argparse

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from dataset import get_data_loaders
from model import get_model


def save_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    fig, ax = plt.subplots(figsize=(18, 18))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an image classification model.")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2"],
                        help="Model architecture to evaluate.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Training fraction used when training this model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, _, test_loader, class_names = get_data_loaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_fraction=1.0
    )

    num_classes = len(class_names)
    model = get_model(args.model, num_classes).to(device)

    fraction_tag = str(args.train_fraction).replace(".", "p")

    model_path = f"outputs/{args.model}_img{args.image_size}_frac{fraction_tag}_best.pth"
    results_path = f"results/{args.model}_img{args.image_size}_frac{fraction_tag}_eval_metrics.json"

    cm_raw_path = f"results/{args.model}_img{args.image_size}_frac{fraction_tag}_confusion_matrix.png"
    cm_norm_path = f"results/{args.model}_img{args.image_size}_frac{fraction_tag}_confusion_matrix_normalized.png"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    total_images = 0

    inference_start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_images += labels.size(0)

    total_inference_time = time.time() - inference_start_time
    avg_inference_time_per_image = total_inference_time / total_images

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )

    os.makedirs("results", exist_ok=True)

    # Save confusion matrices
    save_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        save_path=cm_raw_path,
        normalize=None,
        title=f"{args.model} | img={args.image_size} | frac={args.train_fraction} | Raw Confusion Matrix"
    )

    save_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        save_path=cm_norm_path,
        normalize="true",
        title=f"{args.model} | img={args.image_size} | frac={args.train_fraction} | Normalized Confusion Matrix"
    )

    summary = {
        "model": args.model,
        "image_size": args.image_size,
        "train_fraction": args.train_fraction,
        "batch_size": args.batch_size,
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_inference_time_seconds": total_inference_time,
        "average_inference_time_per_image_seconds": avg_inference_time_per_image,
        "confusion_matrix_raw_path": cm_raw_path,
        "confusion_matrix_normalized_path": cm_norm_path,
        "classification_report": report_dict
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Total inference time: {total_inference_time:.4f}s")
    print(f"Average inference time per image: {avg_inference_time_per_image:.8f}s")
    print(f"Raw confusion matrix saved to: {cm_raw_path}")
    print(f"Normalized confusion matrix saved to: {cm_norm_path}")
    print(f"\nClassification Report:\n{report_text}")
    print(f"\nEvaluation metrics saved to: {results_path}")


if __name__ == "__main__":
    main()