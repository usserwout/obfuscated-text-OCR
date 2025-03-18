import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import json
import os
from PIL import ImageDraw
from tqdm import tqdm

IMAGE_SIZE = (300,400)

class CouponDataset(Dataset):
    def __init__(self, metadata, image_dir, transform=None):
        self.metadata = metadata
        self.image_dir = image_dir
        self.transform = transform
        self.keys = list(metadata.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        bbox = self.metadata[key]
        x1, y1 = bbox['top_left']
        x2, y2 = bbox['bottom_right']
        
        img_path = os.path.join(self.image_dir, f"{key}.png")
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size

        if self.transform:
            image = self.transform(image)
        
       
        

        sx = IMAGE_SIZE[1] / orig_width
        sy = IMAGE_SIZE[0] / orig_height
        
        x1_scaled = x1 * sx
        y1_scaled = y1 * sy
        x2_scaled = x2 * sx
        y2_scaled = y2 * sy
        
        # Draw bounding box on the transformed image
        # transformed_image = transforms.ToPILImage()(image)
        # draw = ImageDraw.Draw(transformed_image)
        # draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline="red", width=2)
        
        # transformed_img_path = os.path.join(self.image_dir, f"{key}_transformed.png")
        # transformed_image.save(transformed_img_path)

        target = torch.tensor([x1_scaled, y1_scaled, x2_scaled, y2_scaled], dtype=torch.float32)
        return image, target

def create_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    return model

def main():

    with open('generated/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    image_dir = 'generated' 
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CouponDataset(metadata, image_dir, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = create_model().to(device)
    
    if os.path.exists('text_detector.pth'):
        model.load_state_dict(torch.load('text_detector.pth', 
                                map_location=device,
                                weights_only=True))
        print("Loaded existing model weights")
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    best_loss = float('inf')
    num_epochs = 30
    for epoch in range(num_epochs):
      model.train()
      train_loss = 0.0
      for images, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
      
      train_loss /= len(train_loader.dataset)
      
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
          images, targets = images.to(device), targets.to(device)
          outputs = model(images)
          val_loss += criterion(outputs, targets).item() * images.size(0)
      val_loss /= len(val_loader.dataset)
      scheduler.step(val_loss)
      print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

      if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'text_detector.pth')
            print(f"Saved new best model with loss: {best_loss:.4f}")
    

if __name__ == '__main__':
    main()