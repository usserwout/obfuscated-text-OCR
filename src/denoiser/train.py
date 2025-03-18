import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import json
import os
from tqdm import tqdm
import numpy as np


class TextSegmentationDataset(Dataset):
    def __init__(self, metadata, image_dir, mask_dir, transform=None, mask_transform=None):
        with open(metadata, "r") as f:
            self.metadata = json.load(f)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform if mask_transform else transform
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
        # transformed_image_path = os.path.join( 'transformed.png')
        # pil_image = transforms.ToPILImage()(image)
        # pil_image.save(transformed_image_path)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            

        mask = (mask > 0.5).float()
        
        return image, mask


def create_model(num_classes=1):
    model = deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model


def dice_loss(pred, target):
    smooth = 1.0
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return 1 - ((2. * intersection + smooth) / 
                (pred_flat.sum() + target_flat.sum() + smooth))


def combined_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(torch.sigmoid(pred), target)
    return bce + dice


def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]
            val_loss += criterion(outputs, masks).item() * images.size(0)
            
            # Calculate Dice score
            pred = torch.sigmoid(outputs) > 0.5
            target = masks > 0.5
            
            intersection = (pred & target).float().sum((1, 2, 3))
            union = (pred | target).float().sum((1, 2, 3))
            
            # Add small value to prevent division by zero
            dice = (2 * intersection + 1e-7) / (union + intersection + 1e-7)
            dice_score += dice.sum().item()
    
    return val_loss / len(dataloader.dataset), dice_score / len(dataloader.dataset)


def main():
    BATCH_SIZE = 8  # Increased from 4
    NUM_EPOCHS = 100  # Increased from 50
    IMAGE_SIZE = (128, 384)  # Increased resolution
    LR = 0.001  # Increased from 0.0001

    img_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TextSegmentationDataset(
        metadata="generated/metadata.json",
        image_dir="generated/images",
        mask_dir="generated/masks",
        transform=img_transform,
        mask_transform=mask_transform,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataset.dataset.transform = train_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_model().to(device)
    if os.path.exists("text_segmenter.pth"):
        model.load_state_dict(torch.load("text_segmenter.pth", map_location=device,weights_only=True))
        print("Loaded existing model")

    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )
    
    patience = 10
    counter = 0
    best_loss = float("inf")
    
    os.makedirs("model_checkpoints", exist_ok=True)
    
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        
        val_loss, dice_score = evaluate_model(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Dice Score: {dice_score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        scheduler.step(val_loss)

        # Save model if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "text_segmenter.pth")
            # Save checkpoint with epoch information
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': best_loss,
            # }, f"model_checkpoints/checkpoint_epoch_{epoch}.pth")
            print("Saved new best model")
            counter = 0
        else:
            counter += 1
            
        # Early stopping
        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break


if __name__ == "__main__":
    main()
