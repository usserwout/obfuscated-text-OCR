import sys
import random
import os
from pathlib import Path
import json

# Add the parent directory to the sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Now you can directly import the module
from src.generate_images.generateCoupon import generate_coupon

def add_spaces():
  return " " * random.randint(0, 2)
  

def random_coupon_code(with_spaces=False):
  chars = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  
  return "".join(random.choices(chars, k=4)) +add_spaces()+ "-" +add_spaces()+ "".join(random.choices(chars, k=4)) +add_spaces()+ "-" +add_spaces()+ "".join(random.choices(chars, k=4))



if __name__ == "__main__":
  
  output_path='./generated'
  os.makedirs(output_path, exist_ok=True)
  
  metadata = {}
  
  
  
  for i in range(500):
    print(f"Generating coupon {i+1}...")
    x = int(random.gauss(550, 20))
    y = int(random.gauss(447, 15))
    
    opacity = random.uniform(0.5, 1)

    final_image, meta, mask = generate_coupon(
      text=random_coupon_code(True),
      obfuscate=True,
      text_placement=(x, y),
      opacity=opacity,
      output_mask=True,
    )
    
    top_left = (1000,10000)
    bottom_right = (0,0)
    for char,pos in meta['char_positions']:
      top_left = (min(top_left[0], pos[0]), min(top_left[1], pos[1]))
      bottom_right = (max(bottom_right[0], pos[2]), max(bottom_right[1], pos[3]))
    
    cropped = [max(0, top_left[0]-6), max(0, top_left[1]-6), min(final_image.width, bottom_right[0]+6), min(final_image.height, bottom_right[1]+6)]
    
    final_image.crop(cropped).save(f'./generated/images/coupon{i}.png')
    mask.crop(cropped).save(f'./generated/masks/mask{i}.png')
    
    metadata[f'coupon{i}.png'] = f'mask{i}.png'
    #{
    #  'text': meta['text'],
    #  'char_positions': meta['char_positions'],
    #  'top_left': (round(top_left[0]) -1, round(top_left[1])-1),
    #  'bottom_right': (round(bottom_right[0]) +1, round(bottom_right[1])+1),
    #}
    
  with open('./generated/metadata.json', 'w') as f:
    json.dump(metadata, f, separators=(',', ':'))
      

  