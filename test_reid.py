import cv2
import numpy as np
from core.reid import load_reid_model, extract_feature, match_person

model = load_reid_model()
print("Model ready!")

# Load same image — crop two regions to simulate same person
img = cv2.imread(r'C:\Users\DELL\Downloads\profile suit photo.jpeg')
h, w = img.shape[:2]

# Crop 1 — top half (same person)
crop1 = img[0:h//2, w//4:3*w//4]
# Crop 2 — slightly different crop (same person)
crop2 = img[h//8:5*h//8, w//4:3*w//4]
# Crop 3 — bottom region (different appearance)
crop3 = img[h//2:h, 0:w//2]

feat1 = extract_feature(model, crop1)
feat2 = extract_feature(model, crop2)
feat3 = extract_feature(model, crop3)

from core.reid import cosine_similarity
sim_same = cosine_similarity(feat1, feat2)
sim_diff = cosine_similarity(feat1, feat3)

print(f"Same person similarity  : {sim_same:.4f}")
print(f"Different region sim    : {sim_diff:.4f}")
print(f"ReID working correctly  : {sim_same > sim_diff}")

# Gallery match test
gallery = [
    {"id": 1, "feature": feat1},
    {"id": 2, "feature": feat3},
]
result = match_person(crop2, gallery, model)
print(f"Match result: {result}")