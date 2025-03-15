
from PIL import Image
from OCR.detect import detect_characters, load_model
import os
from PIL import ImageDraw



class CouponCode:
    
    def __init__(self, detections: list, image: Image):
        self.image = image
        self.characters = detections
    
    def save(self, output_file: str):
        
        img_with_boxes = self.image.copy().convert('RGB')
        draw = ImageDraw.Draw(img_with_boxes)
        
        for char_info in self.characters:
            box = char_info['box']
            # Draw rectangle with some padding
            draw.rectangle(box, outline="red", width=2)
            
            #draw.text((box[0], box[3]+10), char_info['character'], fill="red")
        
        img_with_boxes.save(output_file)


    def print_detected_characters(self):
        for i, det in enumerate(self.characters, 1):
            print(f"{i:2}. {det['character']} (conf: {det['confidence']:.2f}) "
                  f"at [{det['box'][0]}, {det['box'][1]}] -> [{det['box'][2]}, {det['box'][3]}]")

    
    def __str__(self):
        characters = [result['character'] for result in self.characters]
        return ''.join(characters[:4]) + '-' + ''.join(characters[4:8]) + '-' + ''.join(characters[8:12])




def read_coupon(image:Image) -> CouponCode:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'character_detector.pth')
    model = load_model(model_path)
    results = detect_characters(image, model)
    
    return CouponCode(results, image)