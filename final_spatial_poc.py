import cv2
import torch
import numpy as np
import math
import os
import sys

# Depth Anything V2 ka path set karna
sys.path.append(os.path.join(os.getcwd(), "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

points = []

# 1. Model Configuration
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
model = DepthAnythingV2(**model_configs['vits'])
weights_path = os.path.join("Depth-Anything-V2", "checkpoints", "depth_anything_v2_vits.pth")

print("AI Model load ho raha hai...")
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

# 2. Image Load Karna
image_path = "object1.jpg"
img_array = np.fromfile(image_path, dtype=np.uint8)
raw_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if raw_img is None:
    print("Error: object2.jpg nahi mili!")
    sys.exit()

# Proportional Resize
original_height, original_width = raw_img.shape[:2]
new_width = 800
new_height = int((new_width / original_width) * original_height)
frame = cv2.resize(raw_img, (new_width, new_height))
clone = frame.copy()

print("AI 3D Depth Compute kar raha hai...")
with torch.no_grad():
    depth_map = model.infer_image(frame)

# -------------------------------------------------------------
# 🔥 OPTION A: MATH CALIBRATION BLOCK (Relative to Metric)
# -------------------------------------------------------------
# Hamara Card center mein hai. Hum wahan ki relative depth ki value uthate hain
# Aur usay real-world 42.0 CM (Camera Height) par map karte hain.
card_y, card_x = int(new_height / 2), int(new_width / 2)
relative_depth_at_card = depth_map[card_y, card_x]

# Formula: Relative depth ko invert kar ke real centimeters mein convert karna
depth_in_cm = (1.0 / (depth_map + 1e-5))
# Scaling factor jo relative map ko asli 42 CM ke barabar le aayega
scaling_factor = 42.0 / (1.0 / (relative_depth_at_card + 1e-5))
metric_depth_map = depth_in_cm * scaling_factor
# -------------------------------------------------------------

# Fixed Pixel ratio for X and Y mapping based on calibrated frame
pixel_to_cm_ratio = 8.56 / 209

def mouse_click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

cv2.namedWindow("DreamVerse - Calibrated Spatial AI")
cv2.setMouseCallback("DreamVerse - Calibrated Spatial AI", mouse_click)

while True:
    temp_frame = clone.copy()

    for point in points:
        cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)

    if len(points) == 2:
        pt1, pt2 = points[0], points[1]
        
        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0], pt2[1]
        
        # Sahi calibrated Metric Depth (Z) uthana
        z1 = metric_depth_map[y1, x1]
        z2 = metric_depth_map[y2, x2]
        
        # Real-world X and Y projection using pixel ratio and specific depth
        real_x1 = (x1 - new_width/2) * pixel_to_cm_ratio * (z1 / 42.0)
        real_y1 = (y1 - new_height/2) * pixel_to_cm_ratio * (z1 / 42.0)
        
        real_x2 = (x2 - new_width/2) * pixel_to_cm_ratio * (z2 / 42.0)
        real_y2 = (y2 - new_height/2) * pixel_to_cm_ratio * (z2 / 42.0)
        
        # 3D Euclidean Distance Formula (Ab yeh sahi CM nikaalega)
        distance_3d = math.sqrt((real_x2 - real_x1)**2 + (real_y2 - real_y1)**2 + (z2 - z1)**2)
        
        cv2.line(temp_frame, pt1, pt2, (0, 255, 0), 2)
        text = f"Calibrated 3D AI: {distance_3d:.1f} CM"
        cv2.putText(temp_frame, text, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    elif len(points) > 2:
        points = []

    cv2.imshow("DreamVerse - Calibrated Spatial AI", temp_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()