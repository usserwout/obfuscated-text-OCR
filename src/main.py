
from PIL import Image

from text_detection.main import crop_text
from denoiser.denoiser import denoise
from OCR.OCR import read_coupon

def get_coupon_code(image:Image) -> str:
  # resize image to a 400x225 image
  
  if image.width > 400 or image.height > 225:
    image = image.resize((400, 225))
  elif image.width < 400 or image.height < 225:
    print('WARNING: Image is too small, results may be inaccurate')
  
  
  image = crop_text(image)
  image.save('cropped.png')
  image = denoise(image)
  image.save('denoised.png')  
  coupon = read_coupon(image)
  
  return coupon
  



def main():
    image_path = './text_detection/generated/coupon122.png'
    image = Image.open(image_path).convert('RGB')
    coupon = get_coupon_code(image)
    coupon.save('annotated.png')
    coupon.print_detected_characters()
    print(coupon)
    print(f"Detected coupon code: {coupon}")
    
if __name__ == "__main__":
    main()