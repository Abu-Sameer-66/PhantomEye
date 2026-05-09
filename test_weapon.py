import cv2
from core.weapon import load_weapon_model, detect_weapons

model = load_weapon_model()
print("Model ready!")

# Test on a sample image from downloads
import os
test_images = [
    r'C:\Users\DELL\Downloads\licensed-image.jpg',
    r'C:\Users\DELL\Downloads\download.jpg',
    r'C:\Users\DELL\Downloads\soft.jpg',
]

for img_path in test_images:
    if os.path.exists(img_path):
        frame = cv2.imread(img_path)
        annotated, detections = detect_weapons(frame, model)
        cv2.imwrite('outputs/weapon_test.jpg', annotated)
        print(f"Tested: {os.path.basename(img_path)}")
        print(f"Detections: {len(detections)}")
        for d in detections:
            print(f"  -> {d['class_name']} ({d['confidence']:.0%})")
        break

print("Done! Check outputs/weapon_test.jpg")