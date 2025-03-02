from PIL import Image, ImageDraw, ImageFont
import os
import random
import math
import json

def get_text_pos(image):
  # find first non transparent pixel from top left, and bottom right
  x_min, y_min = image.size
  x_max, y_max = 0, 0
  found = False

  for y in range(image.height):
    found_line = False
    for x in range(image.width):
      if image.getpixel((x, y))[3] != 0:
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
        found = True
        found_line=True
    if not found_line and found:
      break



  return (x_min, y_min, x_max, y_max)


def apply_line(image, lineCount=None, start=(0,0), end=(0,0)):

  draw = ImageDraw.Draw(image)
  x1 = random.randint(start[0]-5, start[0] + 5)
  
  # Adjust y1 and y2 to have lower chances of being at the top or bottom
  y1 = int(random.triangular(start[1] + (end[1] - start[1]) * 0.1, end[1] - (end[1] - start[1]) * 0.1))
  x2 = random.randint(end[0]-5, end[0]+5)
  y2 = int(random.triangular(start[1] + (end[1] - start[1]) * 0.1, end[1] - (end[1] - start[1]) * 0.1))
  
  draw.line((x1, y1, x2, y2), fill="#3A4460", width=random.randint(10, 20))
    
  return image

def apply_wave(image, waveCount=None, start=(0,0), end=(0,0)):

  draw = ImageDraw.Draw(image)
  amplitude = random.randint(1, 15)
  frequency = random.uniform(0.01, 0.05)
  phase_shift = random.uniform(0, 2 * math.pi)
  
  x1 = random.randint(start[0]-5, start[0] + 5)
  y1 = random.randint(start[1], end[1])
  x2 = random.randint(end[0]-5, end[0]+5)
  y2 = random.randint(start[1], end[1])
    
    
  points = []
  for x in range(start[0], end[0]):
    t = (x - x1) / (x2 - x1)

    y = (1 - t) * y1 + t * y2 + amplitude * math.sin(frequency * x + phase_shift)
    points.append((x, y))
  
  draw.line(points, fill="#3A4460", width=random.randint(10, 20))
  
  return image

def apply_straight_lines(image, start=(0,0), end=(0,0)):
  lineCount = max(1, min(3, int(random.gauss(1, 1)*2.1)))
  diff = (end[1] - start[1] ) / lineCount
  for i in range(lineCount):
    y_max = int(diff * (i+1))
    y_min = int(diff * i)
    y = random.randint(y_min, y_max)
    
    apply_line(image, start=(start[0],start[1]+y), end=(end[0],start[1]+y))
  return image

def generate_coupon(text,template,font, text_color="white", border_color="#3A4460", border_width=10, opacity=None, obfuscate=True):

  if opacity is None:
    opacity = random.uniform(0.75, 0.9)

  # Load template image

  # Create transparent layer for text
  text_layer = Image.new("RGBA",template.size, (255, 255, 255, 0))
  draw = ImageDraw.Draw(text_layer)

  # Font and style configuration
  

  # Define center position (with vertical offset)
  x = template.width // 2
  y = template.height // 2 + 138

  # Draw text with stroke onto text_layer
  draw.text(
    (x, y),
    text,
    font=font,
    fill=text_color,
    anchor="mm",
    stroke_width=border_width,
    stroke_fill=border_color,
  )

  # Calculate and store positions of each character
  char_positions = []
  
  scale = 400 / template.width  
  
  current_x = x - draw.textbbox((0, 0), text, font=font)[2] // 2
  for char in text:
      char_bbox = draw.textbbox((current_x , (y - 115//2)), char, font=font)
      char_bbox = [float(x*scale) for x in char_bbox]
      if char != " " and char != '-':
        char_positions.append((char, char_bbox))
      current_x += (char_bbox[2] - char_bbox[0])/scale


  
  
  x1,y1, x2,y2 = get_text_pos(text_layer)

  if obfuscate:
    rand = random.random() 
    if rand < 0.2:
      apply_straight_lines(text_layer, start=(x1,y1), end=(x2,y2))
    elif rand < 0.25:
      pass #  No obfuscation
    else:
      lineCount = max(1, min(4, int(random.gauss(1, 1)*1.8)))
      for i in range(lineCount):
        if random.random() < 0.3:
          apply_line(text_layer, start=(x1,y1), end=(x2,y2))
        else:
          apply_wave(text_layer, start=(x1,y1), end=(x2,y2))
    

  alpha = text_layer.split()[3]
  alpha = alpha.point(lambda p: p * opacity)
  text_layer.putalpha(alpha)
  

      

  final_image = Image.alpha_composite(template, text_layer)
  
  
  final_image.thumbnail((400, 300), Image.LANCZOS)

  
  # draw = ImageDraw.Draw(final_image)
  # for char, bbox in char_positions:
  #   print(f"Character '{char}' position: top-left {bbox[:2]}, bottom-right {bbox[2:]}")
  #   draw.rectangle(bbox, outline="red", width=1)

  
  metadata = {"char_positions": char_positions, "text": text}


  return final_image, metadata


def random_coupon_code(with_spaces=False):
  chars = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
  extra = ""
  if with_spaces:
    extra = " "
  return "".join(random.choices(chars, k=4)) +extra+ "-" +extra+ "".join(random.choices(chars, k=4)) +extra+ "-" +extra+ "".join(random.choices(chars, k=4))

if __name__ == "__main__":

  output_path='./generated'
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  
  metadata = {}
  template = Image.open("./template.png").convert("RGBA")
  font = ImageFont.truetype("./burbankbigcondensed_bold.otf", 115)

  for i in range(1300):  
    print(f"Generating coupon {i+1}...")
    # Example usage
    final_image, meta = generate_coupon(
      template=template,
      font=font,
      text=random_coupon_code(random.random() < 0.15),
      obfuscate=True
    )
    filename = f'coupon{i}.png'
    final_image.save(f'./generated/{filename}')
    metadata[filename] = meta
  
  # Save metadata to a JSON file
  with open('./generated/metadata.json', 'w') as f:
    json.dump(metadata, f, separators=(',', ':'))
