import os
import cv2
import numpy as np
import torch
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from rclpy.qos import QoSProfile

import numpy as np

import numpy as np

import numpy as np
import cv2

import numpy as np

import cv2

def rect_angle_deg_from_mask(mask_u8):
    """
    mask_u8: 0/255 binary
    returns angle in degrees, roughly in [0,180)
    """
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None

    (cx, cy), (w, h), a = cv2.minAreaRect(c)  # a typically in [-90,0)
    # Convert to an edge direction
    if w < h:
        a = a + 90.0
    # normalize to [0,180)
    a = a % 180.0
    return float(a)

def ang_diff_deg(a, b):
    """Shortest signed difference a-b in degrees, result in [-180,180)."""
    return (a - b + 180.0) % 360.0 - 180.0

def closest_periodic(curr_deg, prev_deg, period_deg):
    """
    Return curr_deg shifted by +/- period_deg*k so it's closest to prev_deg.
    Works for squares (period=90) and lines/rectangles (period=180).
    """
    if prev_deg is None:
        return curr_deg

    k = round((prev_deg - curr_deg) / period_deg)
    candidates = [curr_deg + period_deg*(k-1),
                  curr_deg + period_deg*k,
                  curr_deg + period_deg*(k+1)]
    return min(candidates, key=lambda x: abs(ang_diff_deg(x, prev_deg)))


class MaskOrientation:
    """
    Per-object orientation from mask moments, with direction continuity + confidence gating.
    angle is returned in degrees in [-180, 180].
    """
    def __init__(self, min_area=50, min_conf=0.15):
        self.min_area = min_area
        self.min_conf = min_conf
        self.prev_v = {}        # oid -> (vx, vy)
        self.prev_angle = {}    # oid -> angle_deg

    def update(self, oid: int, mask_u8: np.ndarray):
        # mask_u8 can be 0/1 or 0/255
        if mask_u8.dtype != np.uint8:
            mask_u8 = mask_u8.astype(np.uint8)

        # Make sure it's binary 0/255
        if mask_u8.max() == 1:
            mask_bin = (mask_u8 * 255)
        else:
            mask_bin = mask_u8

        m = cv2.moments(mask_bin, binaryImage=True)
        area = m["m00"]
        if area < self.min_area:
            return None, 0.0

        # Covariance-like terms (normalized central moments)
        mu20 = m["mu20"] / area
        mu02 = m["mu02"] / area
        mu11 = m["mu11"] / area

        # Principal axis angle in radians (same formula you used)
        theta = 0.5 * np.arctan2(2.0 * mu11, (mu20 - mu02))
        v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        # Confidence: based on eigenvalues of covariance matrix
        # lambda = (tr ± sqrt((mu20-mu02)^2 + 4*mu11^2))/2
        tr = mu20 + mu02
        disc = np.sqrt((mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2))
        lam1 = 0.5 * (tr + disc)
        lam2 = 0.5 * (tr - disc)

        # Elongation confidence in [0..1], near 0 means round / ambiguous axis
        conf = float((lam1 - lam2) / (lam1 + lam2 + 1e-9))

        # If ambiguous (nearly round), freeze last angle if we have it
        if conf < self.min_conf and oid in self.prev_angle:
            return self.prev_angle[oid], conf

        # Direction continuity: flip v if it suddenly points opposite
        if oid in self.prev_v:
            pv = self.prev_v[oid]
            if float(v[0] * pv[0] + v[1] * pv[1]) < 0.0:
                v = -v
                theta += np.pi

        angle_deg = float(np.degrees(np.arctan2(v[1], v[0])))
        angle_deg = (angle_deg + 180.0) % 360.0 - 180.0  # wrap to [-180,180]

        self.prev_v[oid] = (float(v[0]), float(v[1]))
        self.prev_angle[oid] = angle_deg
        return angle_deg, conf


def moment_directed_angle_deg(mask_u8, prev_v=None, history_angle_deg=None):
    """
    mask_u8: [H,W] 0/1 or 0/255
    prev_v:  (vx,vy) from previous frame to disambiguate sign
    returns: (angle_deg, v) where angle_deg is signed in [-90, +90]
    """
    ys, xs = np.nonzero(mask_u8)
    n = len(xs)
    if n < 10:
        return None, None

    x = xs.astype(np.float32)
    y = ys.astype(np.float32)
    cx, cy = x.mean(), y.mean()
    x0, y0 = x - cx, y - cy

    mu20 = (x0 * x0).mean()
    mu02 = (y0 * y0).mean()
    mu11 = (x0 * y0).mean()

    # This already gives an axis angle in [-90, +90]
    theta = 0.5 * np.arctan2(2.0 * mu11, (mu20 - mu02))  # radians
    v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)  # unit axis direction

    # Make it a directed vector using previous frame (resolve v vs -v)
    if prev_v is not None:
        prev_v = np.asarray(prev_v, dtype=np.float32)
        prev_v /= (np.linalg.norm(prev_v) + 1e-9)
        if float(np.dot(v, prev_v)) < 0.0:
            v = -v
            theta = theta + np.pi  # same direction flip in angle space

    # Signed angle relative to +x
    angle_deg = float(np.degrees(np.arctan2(v[1], v[0])))
    print(f"Raw angle: {angle_deg:.1f} degrees")
    if len(history_angle_deg)>1:
        array_h = np.array(history_angle_deg)
        if array_h is not None and len(array_h) > 0:
            avg = np.mean(array_h)
            print(f"History angles: average: {avg:.1f} degrees")
        # Fold to [-90, +90] (optional, but usually what people want)
            if avg > 45 and angle_deg * history_angle_deg[-1] < 0:
                angle_deg += 180.0
            elif avg < -45 and angle_deg * history_angle_deg[-1] < 0:
                angle_deg -= 180.0

    return angle_deg, (float(v[0]), float(v[1]))



# ADD THIS CLASS (somewhere above the main loop)
class CenterPublisher(Node):
    def __init__(self):
        super().__init__("sam2_centers_publisher")
        self._qos = QoSProfile(depth=10)
        self._pose_pub = self.create_publisher(PoseArray, "/tracked_objects", self._qos)
        self.get_logger().info("Publishing tracked objects on: /tracked_objects [PoseArray]")
        self._hand_pub = self.create_publisher(PointStamped, "/hand_center", self._qos)
        self.get_logger().info("Publishing hand center on: /hand_center [PointStamped]")

    def publish_poses(self, poses_list: list, frame_id: str = "image"):
        """
        poses_list: list of (cx, cy, yaw_rad) tuples, ordered by object index.
        """
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        for cx, cy, yaw_rad in poses_list:
            p = Pose()
            p.position.x = float(cx)
            p.position.y = float(cy)
            p.position.z = 0.0
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            p.orientation.z = float(np.sin(yaw_rad / 2.0))
            p.orientation.w = float(np.cos(yaw_rad / 2.0))
            msg.poses.append(p)

        self._pose_pub.publish(msg)

    def publish_hand(self, cx: float, cy: float, frame_id: str = "image"):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = float(cx)
        msg.point.y = float(cy)
        msg.point.z = 0.0
        self._hand_pub.publish(msg)

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
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


# ------------------------------------------------------------
# BEFORE the loop (put these somewhere above `while True:`)
# ------------------------------------------------------------
ref_areas = {}            # oid -> pixel area from first tracking frame (excluding hand)
ref_areas_set = False
last_good_centers = {}    # oid -> (cx, cy) last published "good" center
last_good_yaws = {}       # oid -> last published yaw (optional)

AREA_MIN_RATIO = 0.5      # if area < ref_area * ratio -> publish last_good center (non-hand)

# ------------------------------------------------------------
# WHILE LOOP (rewritten)
# ------------------------------------------------------------
history_two = []  # keeps last two angle_deg values for smoothing (optional)

TARGET_OID = 2
prev_angle_by_oid = {}     # continuous angle in degrees (can go beyond 180)
MAX_STEP_DEG = 20.0
orient = MaskOrientation(min_area=200, min_conf=0.12)

prev_unwrapped_by_oid = {}  # oid -> continuous angle (can exceed ±180)

def unwrap_deg(curr, prev):
    """Unwrap curr (wrapped to [-180,180]) to be continuous relative to prev."""
    if prev is None:
        return curr
    d = (curr - prev + 180.0) % 360.0 - 180.0
    return prev + d
# --------------------------------
prev_angle_by_oid = {}   # oid -> continuous angle
cal_by_oid = {}          # oid -> bool
angle_f_by_oid = {}      # oid -> float
angle_vis_by_oid = {} 

angle_f = None
cal = False

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

    


    ############TESTING: print debug info about tracked objects
    # TARGET_OID = 2
    # orient = MaskOrientation(min_area=200, min_conf=0.12)
    # prev_v = None  # remembers the directed axis from the previous frame

    # --- inside your while-loop, right after predictor.track(...) ---
    # meas = rect_angle_deg_from_mask(mask_u8)
    out_ids = out_obj_ids.tolist() if hasattr(out_obj_ids, "tolist") else list(out_obj_ids)

    for oid in out_ids:
        if oid == 1:
            continue  # skip hand

        

        j = out_ids.index(oid)

        mask_u8 = (out_mask_logits[j] > 0.0).squeeze().detach().cpu().numpy().astype(np.uint8) * 255

        meas = rect_angle_deg_from_mask(mask_u8)   # mask_u8 must be 0/255

        if meas is not None:
            # For a square, symmetry period is 90 degrees
            prev = prev_angle_by_oid.get(oid, None)
            meas_cont = closest_periodic(meas, prev, period_deg=90.0)

            # Optional: clamp how fast it can change per frame (prevents sudden jumps)
            if prev is not None:
                d = ang_diff_deg(meas_cont, prev)
                d = float(np.clip(d, -MAX_STEP_DEG, MAX_STEP_DEG))
                meas_cont = prev + d

            prev_angle_by_oid[oid] = meas_cont

            if not cal_by_oid.get(oid, False):
                angle_f_by_oid[oid] = meas_cont
                cal_by_oid[oid] = True

            meas_cont -= angle_f_by_oid[oid]

            angle_vis_by_oid[oid] = meas_cont
            print(f"oid={oid}: angle={meas_cont:.1f} deg (meas={meas:.1f})")


        # ang_wrapped, conf = orient.update(TARGET_OID, mask_u8)
        # if ang_wrapped is not None:
        #     ang_unwrapped = unwrap_deg(ang_wrapped, prev_unwrapped_by_oid.get(TARGET_OID))
        #     prev_unwrapped_by_oid[TARGET_OID] = ang_unwrapped

        #     print(
        #         f"Angle oid={TARGET_OID}: {ang_unwrapped:.1f} deg "
        #         f"(wrapped {ang_wrapped:.1f}, conf={conf:.2f})"
        #     )


    # print(f"Angle is {ang:.1f} degrees")

    ############TESTING: print debug info about tracked objects

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

    # Build poses list for non-hand objects; publish hand separately
    poses_to_publish = []
    for oid in out_obj_ids_list:
        c_now = centers.get(oid)
        a_now = current_area.get(oid, 0)

        yaw = np.deg2rad(angle_vis_by_oid.get(oid, 0.0))

        if oid == 1:
            # Hand: publish on separate /hand_center topic
            if c_now is not None:
                cx, cy = c_now
                last_good_centers[oid] = (cx, cy)
                ros_node.publish_hand(cx, cy, frame_id="image")
            elif oid in last_good_centers:
                cx, cy = last_good_centers[oid]
                ros_node.publish_hand(cx, cy, frame_id="image")
            continue

        a0 = ref_areas.get(oid, None)
        too_small = (a0 is not None) and (a_now < a0 * AREA_MIN_RATIO)

        if (c_now is not None) and (not too_small):
            cx, cy = c_now
            last_good_centers[oid] = (cx, cy)
            if yaw is not None:
                last_good_yaws[oid] = yaw
            poses_to_publish.append((cx, cy, yaw))
        else:
            if oid in last_good_centers:
                cx, cy = last_good_centers[oid]
                yaw = last_good_yaws.get(oid, 0.0)
                poses_to_publish.append((cx, cy, yaw))

    if poses_to_publish:
        ros_node.publish_poses(poses_to_publish, frame_id="image")

    # Optional: allow ROS to process internal work
    rclpy.spin_once(ros_node, timeout_sec=0.0)

    # -----------------------
    # Visualization + UI (UNCHANGED behavior)
    # -----------------------
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    vis_rgb = cv2.addWeighted(frame_rgb, 1.0, all_mask, 0.5, 0)

    AXIS_LEN = 80  # pixels, tune as you like

    for oid in out_obj_ids_list:
        if oid == 1:
            continue  # skip hand

        c = centers.get(oid, None)
        if c is None:
            continue

        ang_deg = angle_vis_by_oid.get(oid, None)
        if ang_deg is None:
            continue

        cx, cy = c
        cx_i, cy_i = int(round(cx)), int(round(cy))

        th = np.deg2rad(ang_deg)

        # x-axis direction (cos, sin)
        dx = float(np.cos(th))
        dy = float(np.sin(th))

        # y-axis direction = x rotated +90deg
        dx2 = -dy
        dy2 = dx

        x1 = int(round(cx + AXIS_LEN * dx))
        y1 = int(round(cy + AXIS_LEN * dy))
        x2 = int(round(cx - AXIS_LEN * dx))
        y2 = int(round(cy - AXIS_LEN * dy))

        yx1 = int(round(cx + AXIS_LEN * dx2))
        yy1 = int(round(cy + AXIS_LEN * dy2))
        yx2 = int(round(cx - AXIS_LEN * dx2))
        yy2 = int(round(cy - AXIS_LEN * dy2))

        # draw axes (RGB colors because vis_rgb is RGB)
        cv2.line(vis_rgb, (x2, y2), (x1, y1), (255, 0, 0), 3)   # X axis = red
        cv2.line(vis_rgb, (yx2, yy2), (yx1, yy1), (0, 255, 0), 3) # Y axis = green
        cv2.circle(vis_rgb, (cx_i, cy_i), 4, (255, 255, 255), -1)

        cv2.putText(
            vis_rgb,
            f"{oid}:{ang_deg:.1f}",
            (cx_i + 6, cy_i - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

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


cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()

ros_node.destroy_node()
rclpy.shutdown()
