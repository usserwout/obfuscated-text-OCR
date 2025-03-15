import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm


class TextSegmentationDataset(Dataset):
    def __init__(self, metadata, image_dir, mask_dir, transform=None):
        with open(metadata, "r") as f:
            self.metadata = json.load(f)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = list(self.metadata.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.metadata[img_name]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Save transformed image for debugging
        # transformed_image_path = os.path.join(self.image_dir, 'transformed.png')
        # pil_image = transforms.ToPILImage()(image)
        # pil_image.save(transformed_image_path)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0.95).float()
        # save transformed mask for debugging
        transformed_mask_path = 'transformed.png'
        pil_mask = transforms.ToPILImage()(mask)
        pil_mask.save(transformed_mask_path)
        
        return image, mask


def create_model(num_classes=1):
    model = deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model


def main():
    # Configuration
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    IMAGE_SIZE = (50, 300)
    LR = 0.0001

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
         #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset and Loaders
    dataset = TextSegmentationDataset(
        metadata="generated/metadata.json",
        image_dir="generated/images",
        mask_dir="generated/masks",
        transform=transform,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_model().to(device)
    if os.path.exists("text_segmenter.pth"):
        model.load_state_dict(torch.load("text_segmenter.pth", map_location=device))
        print("Loaded existing model")

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    best_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)["out"]
                val_loss += criterion(outputs, masks).item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "text_segmenter.pth")
            print("Saved new best model")


if __name__ == "__main__":
    main()
