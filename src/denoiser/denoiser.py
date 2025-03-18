from PIL import Image
import numpy as np
import cv2

from denoiser.detect import TextSegmenter
import os



current_dir = os.path.dirname(os.path.abspath(__file__))
segmenter = TextSegmenter(os.path.join(current_dir, 'text_segmenter.pth'))

def denoise(image: Image) -> Image:
    result = segmenter.predict(image)
    return result["segmented_rgb"].convert('L')

if __name__ == '__main__':
    image_path = 'generated/cropped.png'
    image = Image.open(image_path)
    denoised = denoise(image)
    denoised.save('denoised.png')
    print("Denoised image saved as denoised.png")
