import os
import cv2
import numpy as np
import torch
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image


# ✅ RealSense
import pyrealsense2 as rs
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ADD THIS CLASS (somewhere above the main loop)
class CenterPublisher(Node):
    def __init__(self):
        super().__init__("sam2_centers_publisher")
        self._pubs = {}
        self._qos = QoSProfile(depth=10)

    def _get_pub(self, topic_name: str):
        if topic_name not in self._pubs:
            self._pubs[topic_name] = self.create_publisher(PointStamped, topic_name, self._qos)
            q = QoSProfile(depth=1)
            q.reliability = ReliabilityPolicy.RELIABLE
            q.history = HistoryPolicy.KEEP_LAST
            self._img_pub = self.create_publisher(Image, "/sam2/vis_image", q)
        return self._pubs[topic_name]

    def publish_center(self, topic_name: str, cx: float, cy: float, frame_id: str = "image"):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = float(cx)
        msg.point.y = float(cy)
        msg.point.z = 0.0
        self._get_pub(topic_name).publish(msg)


    def publish_vis_image(self, bgr: np.ndarray, frame_id: str = "image"):
        if bgr is None:
            return

        # Ensure contiguous memory (important for correct .tobytes())
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)

        h, w = bgr.shape[:2]

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.height = h
        msg.width = w
        msg.encoding = "bgr8"          # OpenCV BGR 8-bit
        msg.is_bigendian = False
        msg.step = w * 3               # 3 bytes per pixel
        msg.data = bgr.tobytes()       # bytes

        self._img_pub.publish(msg)

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
# Camera (RealSense)  ✅ REPLACED
# -----------------------
# Requested stream mode (must be supported by your camera)
REQ_W, REQ_H, REQ_FPS = 640, 480, 30

ctx = rs.context()
if len(ctx.query_devices()) == 0:
    raise RuntimeError("No Intel RealSense device found. Check USB + permissions (/dev/bus/usb).")

pipeline = rs.pipeline()
config = rs.config()

# Request COLOR stream as BGR8 so the rest of your code stays identical
config.enable_stream(rs.stream.color, REQ_W, REQ_H, rs.format.bgr8, REQ_FPS)

# Try requested mode, fallback if unsupported
try:
    profile = pipeline.start(config)
except RuntimeError as e:
    print("Failed to start RealSense with requested mode:", (REQ_W, REQ_H, REQ_FPS), "\n", e)
    print("Trying fallbacks...")

    fallbacks = [
        (1280, 720, 30),
        (640, 480, 30),
        (640, 480, 60),
    ]
    started = False
    for (w, h, fps) in fallbacks:
        try:
            config = rs.config()
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            profile = pipeline.start(config)
            REQ_W, REQ_H, REQ_FPS = w, h, fps
            started = True
            break
        except RuntimeError:
            pass
    if not started:
        raise

# Print actual mode
cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
print(
    "W,H,FPS,FOURCC:",
    cprof.width(),
    cprof.height(),
    cprof.fps(),
    "RS(BGR8)",
)

HEADLESS = (os.environ.get("DISPLAY", "") == "")

# -----------------------
# 1) Grab initial 30 frames, keep the last
# -----------------------
last_frame_rgb = None
last_frame_bgr = None
for i in range(30):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError(f"Failed to read frame {i+1}/30 from RealSense color stream.")

    frame_bgr = np.asanyarray(color_frame.get_data())  # already BGR8
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
            pipeline.stop()
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


# ------------------------------------------------------------
# BEFORE the loop (put these somewhere above `while True:`)
# ------------------------------------------------------------
ref_areas = {}            # oid -> pixel area from first tracking frame (excluding hand)
ref_areas_set = False
last_good_centers = {}    # oid -> (cx, cy) last published "good" center

AREA_MIN_RATIO = 0.5      # if area < ref_area * ratio -> publish last_good center (non-hand)

# ------------------------------------------------------------
# WHILE LOOP (rewritten)
# ------------------------------------------------------------
frame_idx = 0
ros_visualize = False    # set to False to disable visualization publishing
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        break

    frame_bgr = np.asanyarray(color_frame.get_data())  # BGR8
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

    # -----------------------
    # Overlay masks (multi-object HSV coloring)  (UNCHANGED behavior)
    # -----------------------
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

    # -----------------------
    # Centers + area tracking + "sticky center publish" (NEW)
    # -----------------------
    out_obj_ids_list = (
        out_obj_ids.tolist()
        if hasattr(out_obj_ids, "tolist")
        else list(out_obj_ids)
    )

    current_area = {}
    centers = {}

    for i, oid in enumerate(out_obj_ids_list):
        m = (out_mask_logits[i] > 0.0).squeeze()  # [H,W] bool

        area = int(m.sum().item())  # pixel area
        current_area[oid] = area

        if area == 0:
            centers[oid] = None
            continue

        ys_xs = torch.nonzero(m, as_tuple=False)  # [N,2] (y,x)
        cy, cx = ys_xs.float().mean(dim=0)
        centers[oid] = (float(cx.item()), float(cy.item()))  # (x,y)

    # Capture reference areas ONCE (first tracking frame only), excluding hand (oid==1)
    if not ref_areas_set:
        for oid in out_obj_ids_list:
            if oid == 1:
                continue  # skip hand
            a = current_area.get(oid, 0)
            if a > 0:
                ref_areas[oid] = a
        ref_areas_set = True
        ros_node.get_logger().info(
            f"Reference areas (first frame only, excluding hand): {ref_areas}"
        )

    # Publish: if non-hand area falls below threshold -> publish last_good center (but keep mask as-is)
    for oid in out_obj_ids_list:
        if oid == 1:
            topic = "/hand_center"
        else:
            topic = f"/obj_{oid-1}_center"

        c_now = centers.get(oid)         # (cx,cy) or None
        a_now = current_area.get(oid, 0)

        if oid == 1:
            # Hand: publish current center if available
            if c_now is None:
                continue
            cx, cy = c_now
            last_good_centers[oid] = (cx, cy)
            ros_node.publish_center(topic, cx, cy, frame_id="image")
            continue

        # Non-hand: "too small" => publish last good
        a0 = ref_areas.get(oid, None)
        too_small = (a0 is not None) and (a_now < a0 * AREA_MIN_RATIO)

        if (c_now is not None) and (not too_small):
            # Good update
            cx, cy = c_now
            last_good_centers[oid] = (cx, cy)
            ros_node.publish_center(topic, cx, cy, frame_id="image")
        else:
            # Fallback to last known good center (if exists)
            if oid in last_good_centers:
                cx, cy = last_good_centers[oid]
                ros_node.publish_center(topic, cx, cy, frame_id="image")
            # else: nothing to publish yet

    # Optional: allow ROS to process internal work
    

    # -----------------------
    # Visualization + UI (UNCHANGED behavior)
    # -----------------------
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    vis_rgb = cv2.addWeighted(frame_rgb, 1.0, all_mask, 0.5, 0)

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

    n += 1
    if n % 60 == 0:
        dt = time.time() - t0
        fps = n / dt if dt > 0 else 0.0
        print(f"Approx FPS: {fps:.2f}")
        t0 = time.time()
        n = 0

    if HEADLESS:
        pass
    else:
        cv2.imshow(win_track, vis_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if ros_visualize:
        PUBLISH_EVERY = 1  # publish 1 out of 3 frames
        if (frame_idx % PUBLISH_EVERY) == 0:
            ros_node.publish_vis_image(vis_bgr)
        frame_idx += 1
    rclpy.spin_once(ros_node, timeout_sec=0.0)

pipeline.stop()
if not HEADLESS:
    cv2.destroyAllWindows()

ros_node.destroy_node()
rclpy.shutdown()
