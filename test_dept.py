import cv2
import torch
import numpy as np
import os
import sys

# Depth Anything V2 ka rasta system ko batana
sys.path.append(os.path.join(os.getcwd(), "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

# 1. Model Configuration (Lightweight Small Model for CPU)
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
encoder = 'vits'
model = DepthAnythingV2(**model_configs[encoder])

# Weight file ka path
video_path_weights = os.path.join("Depth-Anything-V2", "checkpoints", "depth_anything_v2_vits.pth")
print("AI Model load ho raha hai, thora intezar karein...")
model.load_state_dict(torch.load(video_path_weights, map_location='cpu'))
model.eval()

# 2. Image Load Karna
image_path = "object2.jpg"  # Aap ki mobile aur card wali image
if not os.path.exists(image_path):
    print(f"Error: {image_path} nahi mili! Check karein ke yeh PoC_Measurement folder mein mojud hai.")
    sys.exit()

img_array = np.fromfile(image_path, dtype=np.uint8)
raw_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# 3. AI Prediction (Depth Map Generate Karna)
print("AI tasweer ka 3D Depth Map nikaal raha hai...")
with torch.no_grad():
    # Model image ko dekh kar depth predict karega
    depth = model.infer_image(raw_img)

# Depth map ko screen par dekhne ke liye normalize aur colour karna
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth_normalized = depth_normalized.astype(np.uint8)
depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

# 4. Result Dikhana
print("Mubarak ho! Depth Map ready hai. 'q' daba kar window band karein.")

# Dono images ko chota kar ke sath sath dikhana
raw_img_resized = cv2.resize(raw_img, (600, 450))
depth_color_resized = cv2.resize(depth_color, (600, 450))
combined = np.hstack((raw_img_resized, depth_color_resized))

cv2.imshow("Original Image vs AI Depth Map", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()