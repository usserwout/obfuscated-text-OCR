import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance

from denoiser.train import create_model

IMAGE_SIZE = (50, 300)

class TextSegmenter:
    def __init__(self, model_path, device='mps'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.load_model(model_path).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
          #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        model = create_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        return model
    def predict(self, orig_image):
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
      mask = (mask > 0.5).astype(np.uint8) * 255
      mask = Image.fromarray(mask).resize((width, height), Image.BILINEAR)
      
      # Apply mask to original image
      result = Image.new('RGB', (width, height))
      mask.save("mask.png")
      
      
      result.paste(orig_image, (0, 0), mask)
      
      # Convert to grayscale
      gray = result.convert("L")
      
      return gray

if __name__ == '__main__':
    segmenter = TextSegmenter('text_segmenter.pth')
    
    orig_image = Image.open('generated/images/coupon96.png').convert('RGB')
    
    result = segmenter.predict(orig_image)
    result.save('text_only.png')