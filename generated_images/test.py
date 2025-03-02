from detect import detect_characters, load_model, draw_detections
import glob
from difflib import SequenceMatcher


def main():
    model = load_model("./character_detector.pth")
    score = 0
    not_detected =0
    semi_correct=0
    miss_classified = 0

    image_files = glob.glob("../images/transformed/*.png")
    for image_file in image_files:
        detections = detect_characters(image_file, model)
        print(f"Detected {len(detections)} characters in {image_file}")
        for i, detection in enumerate(detections, 1):
            print(
                f"{i:2}. {detection['character']} "
                f"(conf: {detection['confidence']:.2f}) "
                f"at [{detection['box'][0]:3},{detection['box'][1]:3}] "
                f"to [{detection['box'][2]:3},{detection['box'][3]:3}]"
            )
            
        coupon_code = "".join([detection['character'] for detection in detections])
        
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
            print(f"Detected all characters but classified some wrong ({rcc} != {cc})")
            
        
        matcher = SequenceMatcher(None, cc, rcc)
        score += matcher.ratio()
        
        
        
        new_file_name =  image_file.replace(".png", f"_{coupon_code}.png").replace("../images/transformed", "./annotated")
        draw_detections(image_file, detections).save(
          new_file_name
        )
        print(
            f"Annotated image saved as {new_file_name}"
        )
    
    print(f"At least 1 character not detected: {not_detected}/{len(image_files)} ({not_detected/len(image_files)*100:.2f}%)")
    print(f"Detected and classified all correct: {semi_correct}/{len(image_files)} ({semi_correct/len(image_files)*100:.2f}%)")
    print(f"Detected all characters but classified some wrong: {miss_classified}/{len(image_files)} ({miss_classified/len(image_files)*100:.2f}%)")
    print(f"Average similarity score: {score / len(image_files) if image_files else 0:.2f}")



if __name__ == "__main__":
    main()