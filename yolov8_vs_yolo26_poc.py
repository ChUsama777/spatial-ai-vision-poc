import cv2
from ultralytics import YOLO

# Load both models (Ensure you have yolo26n.pt downloaded or let the code download it)
model_v8 = YOLO('yolov8n.pt')
model_26 = YOLO('yolo26n.pt') 

# Path to your NEW CLEAN image (Without Iron Scale)
image_path = r"D:\Project_DreamVerse\April – June 2026 Goals\03_Spatial_AI_Architecture\[DV S04] Sprint Planning & Work Breakdown\PoC_Measurement\object0.jpg"

# Run inference on both models
# We use conf=0.1 to ensure it draws a box even if it hallucinates the class
results_v8 = model_v8(image_path, conf=0.1)
results_26 = model_26(image_path, conf=0.1)

# Get annotated frames from both models
frame_v8 = results_v8[0].plot()
frame_26 = results_26[0].plot()

# Resize function to fit laptop screen
def resize_for_display(frame, max_width=600, max_height=500):
    orig_h, orig_w = frame.shape[:2]
    scale = min(max_width / orig_w, max_height / orig_h)
    if scale < 1.0:
        return cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    return frame

# Apply resizing
display_v8 = resize_for_display(frame_v8)
display_26 = resize_for_display(frame_26)

# Show both results in separate windows for direct comparison
cv2.imshow("YOLOv8 - Legacy Model", display_v8)
cv2.imshow("YOLO26 - Latest End-to-End Model", display_26)

cv2.waitKey(0)
cv2.destroyAllWindows()