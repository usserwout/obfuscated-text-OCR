import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance
import os

from denoiser.train import create_model

# Use the same image size as in the updated train.py
IMAGE_SIZE = (128, 384)

class TextSegmenter:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        self.model = self.load_model(model_path).to(self.device)
        
        # Match the same transform used in training
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            # Uncomment if you're using normalization in training
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        model = create_model()
        
        # Check if we're loading a checkpoint or just the model state
        if model_path.endswith('.pth'):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint, strict=True)
                    print("Loaded model state dictionary")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        else:
            raise ValueError(f"Invalid model path: {model_path}")
            
        model.eval()
        return model
        
    def predict(self, orig_image, confidence_threshold=0.3, output_dir=None, save_intermediate=False):
        """
        Segment text from an image
        
        Args:
            orig_image: PIL Image to segment
            confidence_threshold: Threshold for binary mask (default: 0.5)
            output_dir: Directory to save intermediate results (default: None)
            save_intermediate: Whether to save intermediate results (default: False)
            
        Returns:
            segmented_image: PIL Image with only the text visible
        """
        # Load and preprocess image
        width, height = orig_image.size
        
        # Transform
        image = self.transform(orig_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image)['out']
            output = torch.sigmoid(output)
        
        # Post-process
        mask = output.squeeze().cpu().numpy()
        
        # Use the confidence threshold
        binary_mask = (mask > confidence_threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary_mask).resize((width, height), Image.BILINEAR)
        
        # Save intermediate results if requested
        if save_intermediate and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the raw confidence map
            confidence_map = (mask * 255).astype(np.uint8)
            confidence_img = Image.fromarray(confidence_map).resize((width, height), Image.BILINEAR)
            confidence_img.save(os.path.join(output_dir, "confidence_map.png"))
            
            # Save the binary mask
            mask_img.save(os.path.join(output_dir, "binary_mask.png"))
            
            # Save the original image
            orig_image.save(os.path.join(output_dir, "original.png"))
        
        # Apply mask to original image
        # Create an RGBA image with the text visible and background transparent
        # result = Image.new('RGBA', (width, height))
        # orig_rgba = orig_image.convert('RGBA')
        
        # # Create mask as alpha channel
        # r, g, b, _ = orig_rgba.split()
        #result = Image.merge('RGBA', (r, g, b, mask_img.convert('L')))
        
        # For applications that need RGB, create version with black background
        rgb_result = Image.new('RGB', (width, height), (0, 0, 0))
        rgb_result.paste(orig_image, (0, 0), mask_img)
        
        return {
          #  'segmented': result,          # RGBA image with transparent background
            'segmented_rgb': rgb_result,  # RGB image with black background
            'mask': mask_img,             # Binary mask as an image
            'confidence': mask            # Raw confidence values as numpy array
        }


if __name__ == '__main__':
    # Example usage
    segmenter = TextSegmenter('text_segmenter.pth')
    
    # Single image processing
    orig_image = Image.open('generated/images/coupon16.png').convert('RGB')
    
    results = segmenter.predict(orig_image, save_intermediate=True, output_dir='output')
    results['segmented'].save('text_only_transparent.png')
    results['segmented_rgb'].save('text_only.png')
    
