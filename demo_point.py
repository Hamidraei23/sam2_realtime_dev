import os
import cv2
import numpy as np
import torch
import time

# -----------------------
# Precision / CUDA setup
# -----------------------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# -----------------------
# Camera
# -----------------------
cap = cv2.VideoCapture(0 , cv2.CAP_V4L2)  # force V4L2 backend (Linux)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30.00)


# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # force V4L2 backend (Linux)

# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)

print("W,H,FPS,FOURCC:",
      cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
      cap.get(cv2.CAP_PROP_FPS),
      int(cap.get(cv2.CAP_PROP_FOURCC)))
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (VideoCapture(0)). Check /dev/video0 mapping in Docker.")

HEADLESS = (os.environ.get("DISPLAY", "") == "")

# -----------------------
# 1) Grab initial 30 frames, keep the last
# -----------------------
last_frame_rgb = None
for i in range(30):
    ret, frame_bgr = cap.read()
    # frame_bgr = frame_bgr[80:300, 180:460, :]
    if not ret:
        raise RuntimeError(f"Failed to read frame {i+1}/30 from webcam.")
    last_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

h, w = last_frame_rgb.shape[:2]

# -----------------------
# 2) Get click (x, y) on the last frame
# -----------------------
clicked = {"done": False, "x": None, "y": None}

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked["x"], clicked["y"] = x, y
        clicked["done"] = True

if HEADLESS:
    # No GUI available: save image and ask for coordinates
    out_path = "last_frame.png"
    cv2.imwrite(out_path, cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR))
    print(f"[HEADLESS] Saved the last frame to: {out_path}")
    print("[HEADLESS] Open it on your machine and pick a point to track.")
    clicked["x"] = int(input("Enter x: ").strip())
    clicked["y"] = int(input("Enter y: ").strip())
    clicked["done"] = True
else:
    # Show image and capture click
    win = "Click target to track (Left click). Press q to quit."
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    display = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR)
    
    while True:
        vis = display.copy()

        # draw crosshair if clicked
        if clicked["done"]:
            cv2.drawMarker(
                vis,
                (clicked["x"], clicked["y"]),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            cv2.putText(vis, f"Selected: ({clicked['x']}, {clicked['y']})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Quit.")
        if clicked["done"]:
            break
        

    cv2.destroyWindow(win)

x, y = clicked["x"], clicked["y"]
print(f"Using click point: ({x}, {y})")

# -----------------------
# 3) Initialize SAM2 on that last frame + your click prompt
# -----------------------
predictor.load_first_frame(last_frame_rgb)

ann_frame_idx = 0
ann_obj_id = 1

points = np.array([[x, y]], dtype=np.float32)
labels = np.array([1], dtype=np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# -----------------------
# 4) Track live
# -----------------------
tracking_i = 0
win_track = "Tracking (press q to quit)"

if not HEADLESS:
    cv2.namedWindow(win_track, cv2.WINDOW_NORMAL)
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
t = time.time()
n = 0
while True:
    ret, frame_bgr = cap.read()
    # frame_bgr = frame_bgr[80:300, 180:460, :]
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out_obj_ids, out_mask_logits = predictor.track(frame_rgb)
    tracking_i += 1

    # Overlay masks (same idea as your original)
    H, W = frame_rgb.shape[:2]
    all_mask = np.zeros((H, W, 3), dtype=np.uint8)
    all_mask[..., 1] = 255
    # print(f"shape of frame_rgb: {frame_rgb.shape}")

    for i in range(len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        hue = int((i + 3) / (len(out_obj_ids) + 3) * 255)
        all_mask[out_mask[..., 0] == 255, 0] = hue
        all_mask[out_mask[..., 0] == 255, 2] = 255
#         out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 
# 0).cpu().numpy().astype(np.uint8) * 255
#         all_mask[out_mask[..., 0] == 255] = [0, 255, 255]

    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    vis_rgb = cv2.addWeighted(frame_rgb, 1.0, all_mask, 0.5, 0)

    # draw selected point for reference
    cv2.circle(vis_rgb, (x, y), 4, (0, 255, 0), -1)

    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    n+=1
    if n%30==0:
        elapsed = time.time() - t
        fps = n / elapsed if elapsed > 0 else 0.0
        print(f" approx. {fps:.2f} FPS")

        t = time.time()
        n = 0

    if HEADLESS:
        # In headless mode, just run tracking; optionally save frames/video if you want
        # print(f"Frame {tracking_i} tracked...")
        pass
    else:
        cv2.imshow(win_track, vis_bgr)
        cv2.imshow("original", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()
