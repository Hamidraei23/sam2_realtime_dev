import os
import cv2
import numpy as np
import torch
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from rclpy.qos import QoSProfile

# ADD THIS CLASS (somewhere above the main loop)
class CenterPublisher(Node):
    def __init__(self):
        super().__init__("sam2_centers_publisher")
        self._pubs = {}
        self._qos = QoSProfile(depth=10)

    def _get_pub(self, topic_name: str):
        if topic_name not in self._pubs:
            self._pubs[topic_name] = self.create_publisher(PointStamped, topic_name, self._qos)
            self.get_logger().info(f"Created publisher: {topic_name} [PointStamped]")
        return self._pubs[topic_name]

    def publish_center(self, topic_name: str, cx: float, cy: float, frame_id: str = "image"):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = float(cx)
        msg.point.y = float(cy)
        msg.point.z = 0.0
        self._get_pub(topic_name).publish(msg)

# -----------------------
# Precision / CUDA setup
# -----------------------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# -----------------------
# Camera
# -----------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # force V4L2 backend (Linux)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120.101)
print(
    "W,H,FPS,FOURCC:",
    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    cap.get(cv2.CAP_PROP_FPS),
    int(cap.get(cv2.CAP_PROP_FOURCC)),
)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam (VideoCapture(0)). Check /dev/video0 mapping in Docker.")

HEADLESS = (os.environ.get("DISPLAY", "") == "")

# -----------------------
# 1) Grab initial 30 frames, keep the last
# -----------------------
last_frame_rgb = None
last_frame_bgr = None
for i in range(30):
    ret, frame_bgr = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {i+1}/30 from webcam.")
    last_frame_bgr = frame_bgr
    last_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

H0, W0 = last_frame_rgb.shape[:2]

# -----------------------
# 2) Multi-click selection
#    Left click: add a new target
#    Right click / Backspace: undo last
#    Enter: start tracking
#    q / Esc: quit
# -----------------------
click_points = []  # list of (x, y) points

def on_mouse_multi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((int(x), int(y)))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if click_points:
            click_points.pop()

if HEADLESS:
    out_path = "last_frame.png"
    cv2.imwrite(out_path, last_frame_bgr)
    print(f"[HEADLESS] Saved the last frame to: {out_path}")
    print("[HEADLESS] Enter points manually: 'x y' per line. Blank line to finish.")
    while True:
        line = input("point> ").strip()
        if not line:
            break
        xs, ys = line.split()
        click_points.append((int(xs), int(ys)))
else:
    win = "Select targets: Left=add | Right/Backspace=undo | Enter=start | q/Esc=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse_multi)
    
    while True:
        vis = last_frame_bgr.copy()

        # draw selected points with ids
        for idx, (px, py) in enumerate(click_points, start=1):
            cv2.drawMarker(
                vis,
                (px, py),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.putText(
                vis,
                f"{idx}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            vis,
            f"Targets: {len(click_points)}  (Enter=start, RightClick/Backspace=undo)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(win, vis)
        key = cv2.waitKey(10) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Quit.")

        if key in (8, 127):  # Backspace/Delete
            if click_points:
                click_points.pop()

        if key in (13, 10):  # Enter
            if len(click_points) == 0:
                print("No points selected. Click at least one target.")
                continue
            break

        # if n%30==0:
        #     elapsed = time.time() - t
        #     fps = n / elapsed if elapsed > 0 else 0.0
        #     print(f" approx. {fps:.2f} FPS")

        #     t = time.time()
        #     n = 0

    cv2.destroyWindow(win)

print("Selected points:", click_points)

# -----------------------
# 3) Initialize SAM2 on that last frame with multiple prompts
# -----------------------
predictor.load_first_frame(last_frame_rgb)

ann_frame_idx = 0
pos_label = np.array([1], dtype=np.int32)

for obj_id, (px, py) in enumerate(click_points, start=1):
    px = int(np.clip(px, 0, W0 - 1))
    py = int(np.clip(py, 0, H0 - 1))
    points = np.array([[px, py]], dtype=np.float32)
    predictor.add_new_prompt(
        frame_idx=ann_frame_idx,
        obj_id=obj_id,
        points=points,
        labels=pos_label,
    )

# -----------------------
# 4) Track live
# -----------------------
win_track = "Tracking (press q to quit)"
if not HEADLESS:
    cv2.namedWindow(win_track, cv2.WINDOW_NORMAL)

t0 = time.time()
n = 0
rclpy.init(args=None)
ros_node = CenterPublisher()
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

    # Overlay masks (multi-object HSV coloring)
    H, W = frame_rgb.shape[:2]
    all_mask = np.zeros((H, W, 3), dtype=np.uint8)
    all_mask[..., 1] = 255

    for i in range(len(out_obj_ids)):
        out_mask = (
            (out_mask_logits[i] > 0.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
            * 255
        )

        hue = int((i + 3) / (len(out_obj_ids) + 3) * 255)
        all_mask[out_mask[..., 0] == 255, 0] = hue
        all_mask[out_mask[..., 0] == 255, 2] = 255


    ########### ADDED CODE ############
    out_obj_ids_list = (
        out_obj_ids.tolist()
        if hasattr(out_obj_ids, "tolist")
        else list(out_obj_ids)
    )

    # if len(out_obj_ids_list) != 2 or set(out_obj_ids_list) != {1, 2}:
    #     print(f"Stopping: expected obj_ids {{1,2}}, got {out_obj_ids_list}")
    #     break

    # Compute centroids (center of segment) for each object
    centers = {}
    for i, oid in enumerate(out_obj_ids_list):
        # out_mask_logits[i] is typically [1,H,W] (based on your permute usage)
        m = (out_mask_logits[i] > 0.0).squeeze()  # -> [H,W]

        ys_xs = torch.nonzero(m, as_tuple=False)  # [N,2] with (y,x)
        if ys_xs.numel() == 0:
            centers[oid] = None
            continue

        cy, cx = ys_xs.float().mean(dim=0)  # (y,x)
        centers[oid] = (float(cx.item()), float(cy.item()))  # (x,y)

    # Print with your requested names:
    for oid in out_obj_ids_list:
        c = centers.get(oid)
        if c is None:
            continue  # skip if segmentation vanished
        cx, cy = c
        if oid == 1:
            topic = "/hand_center"
        else:
            topic = f"/obj_{oid-1}_center"
        ros_node.publish_center(topic, cx, cy, frame_id="image")

    # Optional: allow ROS to process internal work
    rclpy.spin_once(ros_node, timeout_sec=0.0)




    ########### ADDED CODE ############

    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    vis_rgb = cv2.addWeighted(frame_rgb, 1.0, all_mask, 0.5, 0)

    # draw the original selected points as ids (reference only)
    for idx, (px, py) in enumerate(click_points, start=1):
        cv2.circle(vis_rgb, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(
            vis_rgb,
            str(idx),
            (px + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    # FPS estimate (optional)
    n += 1
    if n % 60 == 0:
        dt = time.time() - t0
        fps = n / dt if dt > 0 else 0.0
        print(f"Approx FPS: {fps:.2f}")
        t0 = time.time()
        n = 0

    if HEADLESS:
        # Headless mode: do nothing (or add a VideoWriter if you want)
        pass
    else:
        cv2.imshow(win_track, vis_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()

ros_node.destroy_node()
rclpy.shutdown()
