import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image
import numpy as np
import json
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.image_list import ImageList
from PIL import ImageDraw

# Configuration (must match training config)
CHAR_SET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789-'
DEVICE = "cpu"
IMAGE_SIZE = (256, 50)
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed
IOU_THRESHOLD = 0.3  # For non-maximum suppression

# Replicate model architecture from training code
class IdentityTransform(GeneralizedRCNNTransform):
    def __init__(self, *, min_size, max_size, image_mean, image_std, **kwargs):
        super().__init__(min_size=min_size, max_size=max_size, image_mean=image_mean, image_std=image_std, **kwargs)
    
    def __call__(self, images, targets=None):
        image_sizes = [img.shape[-2:] for img in images]
        image_tensor = torch.stack(images, dim=0)
        return ImageList(image_tensor, image_sizes), targets

class CharacterModel(MaskRCNN):
    def __init__(self):
        backbone = resnet_fpn_backbone('resnet34', weights="DEFAULT")
        super().__init__(backbone, len(CHAR_SET)+1)
        
        # Modify first conv layer for grayscale
        old_conv = self.backbone.body.conv1
        new_conv = torch.nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        self.backbone.body.conv1 = new_conv
        
        self.transform = IdentityTransform(
            min_size=IMAGE_SIZE[1],
            max_size=IMAGE_SIZE[0],
            image_mean=[0.5],
            image_std=[0.5]
        )

def load_model(model_path):
    """Load trained model from checkpoint"""
    model = CharacterModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def preprocess_image(img):
    """Process image to match training pipeline"""
    #img = Image.open(image_path).convert('L')  # Grayscale
    original_width, original_height = img.size
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
    
    # Resize to model input size
    img_tensor = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0),  # Add batch dimension
        size=(IMAGE_SIZE[1], IMAGE_SIZE[0]),
        mode='bilinear'
    )[0][0]  # Remove batch dimension
    
    return img_tensor, (original_width, original_height)

def detect_characters(image, model):
    """
    Main detection function
    Returns: List of dictionaries with 'character', 'box', and 'confidence'
    """
    # Preprocess image
    img_tensor, original_size = preprocess_image(image)
    img_tensor = img_tensor.to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        predictions = model([img_tensor])[0]
    
    # Process predictions
    detected_chars = []
    original_w, original_h = original_size
    
    # Apply Non-Maximum Suppression
    keep_indices = torchvision.ops.nms(
        predictions['boxes'],
        predictions['scores'],
        IOU_THRESHOLD
    )
    
    # Scale boxes to original image size
    scale_x = original_w / IMAGE_SIZE[0]
    scale_y = original_h / IMAGE_SIZE[1]
    
    for idx in keep_indices:
        score = predictions['scores'][idx].item()
        if score < CONFIDENCE_THRESHOLD:
            continue
            
        box = predictions['boxes'][idx]
        label = predictions['labels'][idx].item()
        
        # Scale coordinates
        x1 = max(0, int(box[0].item() * scale_x))
        y1 = max(0, int(box[1].item() * scale_y))
        x2 = min(original_w, int(box[2].item() * scale_x))
        y2 = min(original_h, int(box[3].item() * scale_y))
        
        detected_chars.append({
            'character': CHAR_SET[label-1],  # Labels are 1-indexed
            'box': [x1, y1, x2, y2],
            'confidence': round(score, 4)
        })
    # Get the top 12 characters
    detected_chars = sorted(detected_chars, key=lambda x: x['confidence'], reverse=True)[:12]
    
    # Sort left to right
    detected_chars.sort(key=lambda x: x['box'][0])
    
    return detected_chars

def draw_detections(image_path, detections):
    """Helper to visualize results"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    for detection in detections:
        box = detection['box']
        draw.rectangle(box, outline="red", width=1)
       
    
    return img

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python detect.py <image_path>")
        sys.exit(1)
    
    model_path = "./character_detector.pth"
    image_path = sys.argv[1]
    
    image = Image.open(image_path).convert('L')
    
    # Load model and detect
    model = load_model(model_path)
    results = detect_characters(image, model)
    
    # Print results
    print(f"Detected {len(results)} characters:")
    for i, det in enumerate(results, 1):
        print(f"{i:2}. {det['character']} (conf: {det['confidence']:.2f}) "
              f"at [{det['box'][0]}, {det['box'][1]}] -> [{det['box'][2]}, {det['box'][3]}]")
    
    # Save visualization and results
    output_img = draw_detections(image_path, results)
    with open('detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    output_img.save('detection_result.png')
    
    print(f"Visualization saved to {output_img}")
    print("JSON results saved to detection_results.json")