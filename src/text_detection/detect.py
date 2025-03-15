import torch
from torchvision import transforms
from PIL import Image
from text_detection.train import create_model
from PIL import ImageDraw

IMAGE_SIZE = (300,400)


def detect(image, model_path):
    # Load model
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    # Load and process image
   
    orig_width, orig_height = image.size

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor).squeeze().numpy()
        
        
    # transformed_image = transforms.ToPILImage()(input_tensor.squeeze(0))
    # draw = ImageDraw.Draw(transformed_image)
    # draw.rectangle(output, outline="red", width=2)

    # transformed_img_path = "./result.png"
    # transformed_image.save(transformed_img_path)


    # Convert to original coordinates
    x1 = int(output[0] * orig_width / IMAGE_SIZE[1])
    y1 = int(output[1] *  orig_height/IMAGE_SIZE[0]) 
    x2 = int(output[2] * orig_width / IMAGE_SIZE[1])
    y2 = int(output[3] * orig_height / IMAGE_SIZE[0])

    return [x1, y1, x2, y2] 

if __name__ == '__main__':
    image_path = 'generated/coupon665.png'
    image = Image.open(image_path).convert('RGB')
    result = detect(image, 'text_detector.pth')
    
    # Draw the bounding box on the original image
    original_image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(original_image)
    draw.rectangle(result, outline="red", width=2)
    
    # Save the image with the bounding box
    original_image.save('detected.png')
    
    print(f"Detected Text Region: {result}")