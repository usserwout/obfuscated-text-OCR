import glob
from difflib import SequenceMatcher
from main import get_coupon_code
from PIL import Image

def main():
    
    score = 0
    not_detected =0
    semi_correct=0
    miss_classified = 0

    image_files = glob.glob("../images/*.png")
    for image_file in image_files:
        
        image = Image.open(image_file).convert('RGB')
        
        coupon = get_coupon_code(image)
        
        coupon_code = str(coupon)
        
        real_coupon_code= image_file.split("/")[-1].split(".")[0]
        rcc = real_coupon_code.replace("-", "")
        cc = coupon_code.replace("-", "")
        
        if len(cc) != len(rcc):
            not_detected += 1
            print("Failed to detect all characters")
        elif cc == rcc:
            semi_correct += 1
            print("Detected and classified all correct")
        else:
            miss_classified += 1
            print(f"Detected all characters but classified some wrong ((expected) {rcc} != {cc} (found))")
            
        
        matcher = SequenceMatcher(None, cc, rcc)
        score += matcher.ratio()
        
        coupon.save(f"../images/annotated/{real_coupon_code}.png")
        print(f"Annotated image saved as ../images/annotated/{real_coupon_code}.png")
        
        # new_file_name =  image_file.replace(".png", f"_{coupon_code}.png").replace("../images/transformed", "./annotated")
        # draw_detections(image_file, detections).save(
        #   new_file_name
        # )
        # print(
        #     f"Annotated image saved as {new_file_name}"
        # )
    
    print(f"At least 1 character not detected: {not_detected}/{len(image_files)} ({not_detected/len(image_files)*100:.2f}%)")
    print(f"Detected and classified all correct: {semi_correct}/{len(image_files)} ({semi_correct/len(image_files)*100:.2f}%)")
    print(f"Detected all characters but classified some wrong: {miss_classified}/{len(image_files)} ({miss_classified/len(image_files)*100:.2f}%)")
    print(f"Average similarity score: {score / len(image_files) if image_files else 0:.2f}")



if __name__ == "__main__":
    main()