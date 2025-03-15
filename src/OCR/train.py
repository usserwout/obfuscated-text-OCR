import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from PIL import Image
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from tqdm import tqdm
from PIL import ImageDraw
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.detection.image_list import ImageList

# Configuration
CHAR_SET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789-'
NUM_CLASSES = len(CHAR_SET) + 1  # +1 for background class
DEVICE = "cpu" #torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
IMAGE_SIZE = (256, 50) 

# Custom Dataset
class CharacterDataset(Dataset):
    """Loads images with character annotations and handles preprocessing"""
    
    def __init__(self, image_dir, metadata_path):
        """
        Args:
            image_dir (str): Path to directory containing images
            metadata_path (str): Path to JSON metadata file
        """
        self.image_dir = image_dir
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        # Filter and sort valid image files
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.endswith('.png') and f in self.metadata]#[:250]
        self.char_to_idx = {char: i+1 for i, char in enumerate(CHAR_SET)}  # 0 is background

    def __len__(self):
        return len(self.image_files)

    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('L')
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.unsqueeze(0)  # [1, H, W]

        annotations = self.metadata[img_name]['char_positions']
        boxes = []
        labels = []
        for ann in annotations:
            x1, y1, x2, y2 = ann[1]
            char = ann[0]
            orig_w, orig_h = img.shape[2], img.shape[1]
            scale_x = IMAGE_SIZE[0] / orig_w
            scale_y = IMAGE_SIZE[1] / orig_h
            boxes.append([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ])
            labels.append(self.char_to_idx[char])

        # Resize image to fixed size
        
        if img.shape[2] != IMAGE_SIZE[0] or img.shape[1] != IMAGE_SIZE[1]:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(IMAGE_SIZE[1], IMAGE_SIZE[0]), mode='bilinear')[0]
        
        # Write image to disk for debugging
        # img_pil = Image.fromarray((img.squeeze(0).numpy() * 255).astype(np.uint8)).convert("RGB")
        # draw = ImageDraw.Draw(img_pil)
        # for box in boxes:
        #     draw.rectangle(box, outline="red", width=1)
        # img_pil.save('./gen.png')
        
        # Create dummy masks: one binary mask per box
        masks = []
        img_w, img_h = IMAGE_SIZE
        for box in boxes:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(img_w, int(box[2]))
            y2 = min(img_h, int(box[3]))
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(np.array(masks), dtype=torch.uint8),  # New key!
            'image_id': torch.tensor([idx]),
            'area': (torch.as_tensor(boxes)[:, 3] - torch.as_tensor(boxes)[:, 1]) * 
                    (torch.as_tensor(boxes)[:, 2] - torch.as_tensor(boxes)[:, 0]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }

        return img, target


def collate_fn(batch):
    """Custom collate function to handle variable numbers of boxes"""
    return tuple(zip(*batch))

# Update IdentityTransform to wrap images in an ImageList
class IdentityTransform(GeneralizedRCNNTransform):
    def __init__(self, *, min_size, max_size, image_mean, image_std, **kwargs):
        super().__init__(min_size=min_size, max_size=max_size, image_mean=image_mean, image_std=image_std, **kwargs)
    def __call__(self, images, targets=None):
        image_sizes = [img.shape[-2:] for img in images]
        image_tensor = torch.stack(images, dim=0)
        image_list = ImageList(image_tensor, image_sizes)
        return (image_list, targets)

# Mask R-CNN Model with ResNet-50 FPN backbone
class CharacterModel(MaskRCNN):
    def __init__(self):
        backbone = resnet_fpn_backbone('resnet34', weights="DEFAULT")
        #backbone = resnet_fpn_backbone('resnet50', weights="DEFAULT")

        for param in backbone.body.parameters():
            param.requires_grad = False

        for param in backbone.fpn.parameters():
            param.requires_grad = True

        super().__init__(backbone, NUM_CLASSES)
        
        old_conv = self.backbone.body.conv1
        new_conv = torch.nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
        self.backbone.body.conv1 = new_conv

        self.transform = IdentityTransform(
            min_size=IMAGE_SIZE[1],
            max_size=IMAGE_SIZE[0],
            image_mean=[0.5], 
            image_std=[0.5]   
        )
        
def train_model():
    dataset = CharacterDataset('./generated', './generated/metadata.json')
    
    if len(dataset) == 0:
        print("Warning: Dataset is empty. Check image directory and metadata file.")
        return
    
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, 
        collate_fn=collate_fn, num_workers=4   
    )

    model = CharacterModel().to(DEVICE)
    
    model_path = 'character_detector.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f'Model loaded from {model_path}')
    else:
        print('No existing model found. Training from scratch.')
    
    # Visualize transformed images
    # for images, targets in dataloader:
    #      # Move data to device
    #      images = [img.to(DEVICE) for img in images]
    #      targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
    #      # Apply the identity transformation
    #      transformed_images, _ = model.transform(images=images, targets=targets)
        
    #      # Iterate over the tensor attribute of ImageList
    #      for i, img in enumerate(transformed_images.tensors):
    #          img = img.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    #          img = (img * 0.5 + 0.5) * 255  # De-normalize
    #          img = img.astype(np.uint8)
            
    #          # Display the image
    #          plt.imshow(img, cmap='gray')
    #          plt.title(f'Transformed Image {i}')
    #          plt.show()
        
    #      # Break after visualizing the first batch
    #      break
    
    
    # try:
    #     scripted_model = torch.jit.script(model)
    #     model = scripted_model
    #     print("Model successfully scripted for optimization.")
    # except Exception as e:
    #     print("TorchScript scripting failed:", e)
    
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    
    # Training loop
    model.train()
    for epoch in range(20):  
        print(f'Starting epoch {epoch+1}...')
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}')):

            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            output = model(images, targets)
            if isinstance(output, tuple):
                loss_dict = output[0]
            else:
                loss_dict = output
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if batch_idx % 10 == 0:
                tqdm.write(f'Epoch {epoch+1} | Batch {batch_idx} | Loss: {losses.item():.2f} | Avg: {epoch_loss / (batch_idx+1):.2f}')
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1} | Average Loss: {avg_epoch_loss:.2f}')
        
        scheduler.step(avg_epoch_loss)
        
        torch.save(model.state_dict(), 'character_detector.pth')

    torch.save(model.state_dict(), 'character_detector.pth')
    print('Training complete. Model saved.')

if __name__ == '__main__':
    train_model()