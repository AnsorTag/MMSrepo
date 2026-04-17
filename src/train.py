import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_data_loaders
from model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2"],
                        help="Model architecture to use.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (e.g. 0.3, 0.6, 1.0).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_fraction=args.train_fraction
    )

    num_classes = len(class_names)
    model = get_model(args.model, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    fraction_tag = str(args.train_fraction).replace(".", "p")

    save_model_path = f"outputs/{args.model}_img{args.image_size}_frac{fraction_tag}_best.pth"
    save_metrics_path = f"results/{args.model}_img{args.image_size}_frac{fraction_tag}_train_metrics.json"

    best_val_acc = 0.0
    history = []

    training_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch_time_seconds": epoch_time
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            print(f"Saved best model to {save_model_path}")

    total_training_time = time.time() - training_start_time
    average_epoch_time = total_training_time / args.epochs

    final_summary = {
        "model": args.model,
        "image_size": args.image_size,
        "train_fraction": args.train_fraction,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "best_validation_accuracy": best_val_acc,
        "total_training_time_seconds": total_training_time,
        "average_epoch_time_seconds": average_epoch_time,
        "history": history
    }

    with open(save_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=4)

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Average epoch time: {average_epoch_time:.2f}s")
    print(f"Training metrics saved to: {save_metrics_path}")


if __name__ == "__main__":
    main()