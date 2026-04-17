import sys
import torch
from PIL import Image
from torchvision import transforms

from dataset import get_data_loaders
from model import get_model


def predict_image(image_path, model_name="resnet50", image_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, _, class_names = get_data_loaders(batch_size=1, image_size=image_size)
    num_classes = len(class_names)

    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(f"outputs/{model_name}_best.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)

    predicted_class = class_names[pred.item()]
    return predicted_class


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path_to_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    prediction = predict_image(image_path)
    print("Predicted class:", prediction)