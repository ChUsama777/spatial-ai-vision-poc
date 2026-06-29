import cv2
import math
import numpy as np

points = []
# Nayi tasweer ka naam 'object2.jpg' set kar diya hai
image_path = r"D:\Project_DreamVerse\April – June 2026 Goals\03_Spatial_AI_Architecture\[DV S04] Sprint Planning & Work Breakdown\PoC_Measurement\Google_Pixel_pro_(W).jpg"

img_array = np.fromfile(image_path, dtype=np.uint8)
frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if frame is None:
    print("Error: object2.jpg load nahi hui. Path check karein!")
else:
    original_height, original_width = frame.shape[:2]
    new_width = 800 
    new_height = int((new_width / original_width) * original_height)
    frame = cv2.resize(frame, (new_width, new_height))
    
    clone = frame.copy()

    # === PRODUCTION-READY CALIBRATION FOR PIXEL 9 PRO SERIES (Phase 3) ===
    camera_distance = 35.56
    object_height = 0.85
    lens_distortion_factor = 1.0  # Planar baseline uses standard geometry
    
    # Mathematical Re-Calibration based on Phase 3 empirical ground truth
    # We calibrate based on 7.20 cm (Width) producing 414 px
    ratio = (7.20 / 414) * ((camera_distance - object_height) / camera_distance) * lens_distortion_factor  

    def mouse_click(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow("Measurement PoC")
    cv2.setMouseCallback("Measurement PoC", mouse_click)

    while True:
        temp_frame = clone.copy()

        for point in points:
            cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)

        if len(points) == 2:
            pt1 = points[0]
            pt2 = points[1]
            
            distance_pixels = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            distance_cm = distance_pixels * ratio
            
            cv2.line(temp_frame, pt1, pt2, (0, 255, 0), 2)
            text_to_show = f"{distance_cm:.2f} CM ({int(distance_pixels)} px)"
            cv2.putText(temp_frame, text_to_show, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        elif len(points) > 2:
            points = []

        cv2.imshow("Measurement PoC", temp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()