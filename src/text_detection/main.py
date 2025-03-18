from PIL import Image
from text_detection.detect import detect
import os


def crop_text(image:Image) -> Image:
  
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'text_detector.pth')
    result = detect(image, model_path)
    result[0] = max(0, result[0] - 6)
    result[1] = max(0, result[1] - 6)
    result[2] = min(image.width, result[2] + 6)
    result[3] = min(image.height, result[3] + 6)
    
    cropped = image.crop(result)
    
    return cropped




if __name__ == '__main__':
    image_path = 'generated/coupon665.png'
    image = Image.open(image_path).convert('RGB')
    cropped = crop_text(image)
    cropped.save('cropped.png')