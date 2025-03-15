import sys
import random
import os
from pathlib import Path
import json

FILE_PATH = Path(__file__).resolve()  
# Add the parent directory of 'text_detection' to the Python path
sys.path.append(str(FILE_PATH.parent.parent))
sys.path.append(str(FILE_PATH.parent.parent.parent))


from src.generate_images.generateCoupon import generate_coupon
from src.text_detection.detect import detect
from src.denoiser.denoiser import denoise

from PIL import ImageDraw

def add_spaces():
  return " " * random.randint(0, 2)
  

def random_coupon_code(with_spaces=False):
  chars = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  
  return "".join(random.choices(chars, k=4)) +add_spaces()+ "-" +add_spaces()+ "".join(random.choices(chars, k=4)) +add_spaces()+ "-" +add_spaces()+ "".join(random.choices(chars, k=4))



if __name__ == "__main__":
  
  output_path='./generated'
  os.makedirs(output_path, exist_ok=True)
  
  metadata = {}
  
  
  
  for i in range(750):
    print(f"Generating coupon {i+1}...")
    x = int(random.gauss(550, 40))
    y = int(random.gauss(447, 30))
    
    opacity = random.uniform(0.5, 1)

    image, meta = generate_coupon(
      text=random_coupon_code(True),
      obfuscate=True,
      text_placement=(x, y),
      opacity=opacity
    )
    # image.save(f'./generated/coupon{i}.png')
    image = image.convert('RGB')
    
    top_left = (1000,10000)
    bottom_right = (0,0)
    
    for char,pos in meta['char_positions']:
      top_left = (min(top_left[0], pos[0]), min(top_left[1], pos[1]))
      bottom_right = (max(bottom_right[0], pos[2]), max(bottom_right[1], pos[3]))
    
    
    result = detect(image, str(FILE_PATH.parent.parent / 'text_detection' / 'text_detector.pth'))
    result[0] = max(0, result[0] - 6)
    result[1] = max(0, result[1] - 6)
    result[2] = min(image.width, result[2] + 6)
    result[3] = min(image.height, result[3] + 6)
    cropped = image.crop(result)
    # Fix the character bounding boxes
    corrected_char_positions = []
    for char, pos in meta['char_positions']:
        pos[0] =max(0, pos[0] - result[0]) 
        pos[1] =max(0, pos[1] - result[1])
        pos[2] =min(result[2], pos[2] - result[0])
        pos[3] =min(result[3], pos[3] - result[1])
        
        corrected_char_positions.append((char, pos))
        
    # save image with bounding box
    # image_with_boxes = cropped.copy()
    # draw = ImageDraw.Draw(image_with_boxes)
    # for char, pos in corrected_char_positions:
    #   draw.rectangle(pos, outline='red')
      
    # image_with_boxes.save(str(FILE_PATH.parent.parent / 'OCR' / 'generated'  / f'coupon{i}.png'))
    
    #cropped.save(str(FILE_PATH.parent.parent / 'OCR' / 'generated' / f'coupon{i}.png'))
    
    denoised_image = denoise(cropped)
    denoised_image.save(str(FILE_PATH.parent / 'generated'  / f'coupon{i}.png'))
    
    
    
    metadata[f'coupon{i}.png'] = {
     'text': meta['text'],
     'char_positions': meta['char_positions'],
      # 'top_left': (round(top_left[0]) -1, round(top_left[1])-1),
      # 'bottom_right': (round(bottom_right[0]) +1, round(bottom_right[1])+1),
    }
    
  with open(str(FILE_PATH.parent / 'generated'  / 'metadata.json'), 'w') as f:
    json.dump(metadata, f, separators=(',', ':'))