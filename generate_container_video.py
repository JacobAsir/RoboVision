import cv2
import numpy as np
import math

output_path = "container-loading-demo.mp4"
width, height = 1280, 720
fps = 30
duration_sec = 10
total_frames = fps * duration_sec

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame_idx in range(total_frames):
    t = frame_idx / float(fps)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Dark high-tech background with subtle grid
    img[:] = (15, 20, 32)
    grid_size = 40
    for x in range(0, width, grid_size):
        cv2.line(img, (x, 0), (x, height), (22, 30, 48), 1)
    for y in range(0, height, grid_size):
        cv2.line(img, (0, y), (width, y), (22, 30, 48), 1)
        
    # --- LEFT SIDE: Cargo Bay Scanner Feed ---
    cv2.rectangle(img, (40, 60), (600, 660), (35, 45, 70), -1)
    cv2.rectangle(img, (40, 60), (600, 660), (77, 107, 255), 2)
    cv2.putText(img, "CARGO BAY 01 - INBOUND CARGO SCANNER", (55, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (147, 197, 253), 2)
    
    # Conveyor belt platform in bay
    cv2.rectangle(img, (70, 480), (570, 620), (50, 60, 85), -1)
    cv2.polylines(img, [np.array([[70,480], [570,480], [540,620], [100,620]])], True, (77, 107, 255), 2)
    
    # Laser scanning line moving back and forth
    scan_y = int(200 + 250 * (0.5 + 0.5 * math.sin(t * 3)))
    cv2.line(img, (70, scan_y), (570, scan_y), (0, 255, 255), 2)
    
    # Moving cargo boxes on conveyor
    progress = (t * 0.8) % 3.0 # Cycle through 3 boxes
    
    # Box A (Heavy - Green)
    bx_a = int(80 + (progress * 160))
    cv2.rectangle(img, (bx_a, 340), (bx_a + 110, 460), (16, 185, 129), -1)
    cv2.rectangle(img, (bx_a, 340), (bx_a + 110, 460), (255, 255, 255), 2)
    cv2.putText(img, "BOX A [HEAVY]", (bx_a + 5, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 2)
    
    # Box B (Medium - Blue)
    bx_b = int(240 + (progress * 160))
    if bx_b < 540:
        cv2.rectangle(img, (bx_b, 360), (bx_b + 95, 460), (255, 107, 77), -1)
        cv2.rectangle(img, (bx_b, 360), (bx_b + 95, 460), (255, 255, 255), 2)
        cv2.putText(img, "BOX B [MED]", (bx_b + 5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 107, 77), 2)
        
    # Box C (Fragile - Yellow)
    bx_c = int(400 + (progress * 160))
    if bx_c < 540:
        cv2.rectangle(img, (bx_c, 380), (bx_c + 85, 460), (8, 179, 234), -1)
        cv2.rectangle(img, (bx_c, 380), (bx_c + 85, 460), (255, 255, 255), 2)
        cv2.putText(img, "BOX C [TOP]", (bx_c + 5, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (8, 179, 234), 2)

    # --- RIGHT SIDE: Real-time 3D Container ❶ Stacking Animation ---
    cv2.rectangle(img, (640, 60), (1240, 660), (35, 45, 70), -1)
    cv2.rectangle(img, (640, 60), (1240, 660), (16, 185, 129), 2)
    cv2.putText(img, "CONTAINER 1 - 3D REAL-TIME PACKING MAP", (655, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 231, 183), 2)

    # Wireframe Container ❶ Outlines
    cv2.rectangle(img, (680, 160), (1200, 600), (77, 107, 255), 2)
    
    # Animated Filling of Container Layers based on frame progress
    fill_stage = min(3, int(t * 0.4) + 1)
    
    # Floor Layer (Box A - Heavy Green)
    cv2.rectangle(img, (690, 460), (1190, 590), (16, 185, 129), -1)
    cv2.rectangle(img, (690, 460), (1190, 590), (255, 255, 255), 2)
    cv2.putText(img, "FLOOR LAYER: BOX A [HEAVY BASE]", (800, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    
    # Middle Layer (Box B - Medium Blue)
    if fill_stage >= 2:
        cv2.rectangle(img, (690, 320), (1190, 450), (255, 107, 77), -1)
        cv2.rectangle(img, (690, 320), (1190, 450), (255, 255, 255), 2)
        cv2.putText(img, "MID LAYER: BOX B [MEDIUM STACK]", (800, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Top Layer (Box C - Fragile Yellow)
    if fill_stage >= 3:
        cv2.rectangle(img, (690, 180), (1190, 310), (8, 179, 234), -1)
        cv2.rectangle(img, (690, 180), (1190, 310), (255, 255, 255), 2)
        cv2.putText(img, "TOP LAYER: BOX C [FRAGILE SURFACE]", (790, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Capacity Banner overlay
    fill_pct = 50.0 if fill_stage == 1 else (82.0 if fill_stage == 2 else 94.2)
    cv2.rectangle(img, (640, 610), (1240, 660), (10, 15, 26), -1)
    cv2.putText(img, f"3D PACKING COMPLETE | FILL RATIO: {fill_pct:.1f}% | AXLE WEIGHT: EVEN", (660, 642), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 185, 129), 2)

    out.write(img)

out.release()
print("Generated container-loading-demo.mp4 successfully!")
