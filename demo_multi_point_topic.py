import os
import cv2
import numpy as np
import torch
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image

import time

class CenterPublisher(Node):
    """
    - Subscribes to image_topic (expects bgr8) WITHOUT cv_bridge (NumPy2-safe)
    - Publishes centers as PointStamped on per-object topics (same as before)
    - Publishes visualization/seg overlay as sensor_msgs/Image (bgr8) on /image_seg
    """
    def __init__(self, image_topic: str = "/camera/camera/color/image_raw"):
        super().__init__("sam2_tracker_node")
        self._pubs = {}
        self._qos = QoSProfile(depth=10)

        # Image subscription state
        self._latest_bgr = None
        self._new_frame = False
        self._last_header = None
        self._image_topic = image_topic

        # Subscribe to camera topic
        self._img_sub = self.create_subscription(
            Image,
            self._image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Subscribed to image topic: {self._image_topic} [sensor_msgs/Image, bgr8]"
        )

        # Publish segmented/overlay image
        self._seg_pub = self.create_publisher(Image, "/image_seg", qos_profile_sensor_data)
        self.get_logger().info("Publishing segmentation overlay on: /image_seg [sensor_msgs/Image, bgr8]")

    # -----------------------
    # Center publishers (unchanged behavior)
    # -----------------------
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
    # NumPy-only Image conversions (NO cv_bridge)
    # -----------------------
    def _imgmsg_to_bgr8(self, msg: Image) -> np.ndarray:
        enc = (msg.encoding or "").lower()
        if enc != "rgb8":
            raise ValueError(f"Expected bgr8, got: {msg.encoding}")

        h, w, step = int(msg.height), int(msg.width), int(msg.step)
        buf = np.frombuffer(msg.data, dtype=np.uint8)

        # step may include padding; handle both cases
        if step == w * 3:
            frame = buf.reshape(h, w, 3)
        else:
            # reshape as (rows, step_bytes), then crop to w*3 bytes
            frame = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)

        # Make contiguous/writable for OpenCV + downstream ops
        return np.ascontiguousarray(frame)
    


    def _imgmsg_to_rgb8(self, msg: Image) -> np.ndarray:
        enc = (msg.encoding or "").lower()
        if enc != "rgb8":
            raise ValueError(f"Expected rgb8, got: {msg.encoding}")

        h, w, step = int(msg.height), int(msg.width), int(msg.step)
        buf = np.frombuffer(msg.data, dtype=np.uint8)

        if step == w * 3:
            frame = buf.reshape(h, w, 3)
        else:
            frame = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)

        return np.ascontiguousarray(frame)  # RGB uint8


    def _bgr8_to_imgmsg(self, frame_bgr: np.ndarray, header=None) -> Image:
        frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        h, w = frame_bgr.shape[:2]

        msg = Image()
        if header is not None:
            msg.header = header
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "image"

        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(w * 3)
        msg.data = frame_bgr.tobytes()
        return msg
    

    

    # -----------------------
    # ROS callbacks / frame queue
    # -----------------------
    def _on_image(self, msg: Image):
        try:
            frame_bgr = self._imgmsg_to_rgb8(msg)
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        self._latest_bgr = frame_bgr
        self._last_header = msg.header
        self._new_frame = True

    def pop_latest_frame(self):
        """Return latest frame once (and clear 'new' flag)."""
        if self._latest_bgr is None or not self._new_frame:
            return None
        self._new_frame = False
        return self._latest_bgr

    def wait_for_frame(self, timeout_sec: float = 2.0):
        """Spin until a new frame arrives or timeout."""
        t_end = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)
            f = self.pop_latest_frame()
            if f is not None:
                return f
        return None

    # -----------------------
    # Publish overlay image to /image_seg
    # -----------------------
    def publish_seg(self, bgr_img: np.ndarray):
        msg = self._bgr8_to_imgmsg(bgr_img, header=self._last_header)
        self._seg_pub.publish(msg)


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

HEADLESS = (os.environ.get("DISPLAY", "") == "")

# -----------------------
# 2) Multi-click selection
# -----------------------
click_points = []  # list of (x, y) points

def on_mouse_multi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((int(x), int(y)))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if click_points:
            click_points.pop()


def main():
    rclpy.init(args=None)

    # topic name exactly as you asked:
    ros_node = CenterPublisher(image_topic="/camera/camera/color/image_raw")

    # -----------------------
    # 1) Grab initial 30 frames from ROS topic, keep the last
    # -----------------------
    last_frame_bgr = None
    last_frame_rgb = None
    for i in range(30):
        frame_bgr = ros_node.wait_for_frame(timeout_sec=2.0)
        if frame_bgr is None:
            ros_node.destroy_node()
            rclpy.shutdown()
            raise RuntimeError(f"Timeout waiting for frame {i+1}/30 from topic /image")

        last_frame_bgr = frame_bgr
        last_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    H0, W0 = last_frame_rgb.shape[:2]

    # -----------------------
    # 2) Select points (same UI logic as before)
    # -----------------------
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

            visrgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            for idx, (px, py) in enumerate(click_points, start=1):
                cv2.drawMarker(
                    visrgb, (px, py), (0, 255, 0),
                    markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2
                )
                cv2.putText(
                    visrgb, f"{idx}", (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

            cv2.putText(
                visrgb,
                f"Targets: {len(click_points)}  (Enter=start, RightClick/Backspace=undo)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(win, visrgb)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                cv2.destroyAllWindows()
                ros_node.destroy_node()
                rclpy.shutdown()
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
    # 4) Track live (frames from /image)
    # -----------------------
    win_track = "Tracking (press q to quit)"
    # if not HEADLESS:
    #     cv2.namedWindow(win_track, cv2.WINDOW_NORMAL)

    t0 = time.time()
    n = 0

    ref_areas = {}         # oid -> pixel area from first tracking frame (excluding hand)
    ref_areas_set = False
    last_good_centers = {} # oid -> (cx, cy)
    AREA_MIN_RATIO = 0.5

    while rclpy.ok():
        # Pump ROS to receive images
        rclpy.spin_once(ros_node, timeout_sec=0.01)

        frame_bgr = ros_node.pop_latest_frame()
        if frame_bgr is None:
            continue

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

        # Centers + area tracking + sticky publish
        out_obj_ids_list = out_obj_ids.tolist() if hasattr(out_obj_ids, "tolist") else list(out_obj_ids)

        current_area = {}
        centers = {}

        for i, oid in enumerate(out_obj_ids_list):
            m = (out_mask_logits[i] > 0.0).squeeze()  # [H,W] bool

            area = int(m.sum().item())
            current_area[oid] = area

            if area == 0:
                centers[oid] = None
                continue

            ys_xs = torch.nonzero(m, as_tuple=False)  # [N,2] (y,x)
            cy, cx = ys_xs.float().mean(dim=0)
            centers[oid] = (float(cx.item()), float(cy.item()))  # (x,y)

        # Reference areas ONCE (first tracking frame), excluding hand (oid==1)
        if not ref_areas_set:
            for oid in out_obj_ids_list:
                if oid == 1:
                    continue
                a = current_area.get(oid, 0)
                if a > 0:
                    ref_areas[oid] = a
            ref_areas_set = True
            ros_node.get_logger().info(f"Reference areas (first frame only, excluding hand): {ref_areas}")

        # Publish centers
        for oid in out_obj_ids_list:
            if oid == 1:
                topic = "/hand_center"
            else:
                topic = f"/obj_{oid-1}_center"

            c_now = centers.get(oid)
            a_now = current_area.get(oid, 0)

            if oid == 1:
                if c_now is None:
                    continue
                cx, cy = c_now
                last_good_centers[oid] = (cx, cy)
                ros_node.publish_center(topic, cx, cy, frame_id="image")
                continue

            a0 = ref_areas.get(oid, None)
            too_small = (a0 is not None) and (a_now < a0 * AREA_MIN_RATIO)

            if (c_now is not None) and (not too_small):
                cx, cy = c_now
                last_good_centers[oid] = (cx, cy)
                ros_node.publish_center(topic, cx, cy, frame_id="image")
            else:
                if oid in last_good_centers:
                    cx, cy = last_good_centers[oid]
                    ros_node.publish_center(topic, cx, cy, frame_id="image")

        # Visualization
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

        # vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        n += 1
        if n % 60 == 0:
            dt = time.time() - t0
            fps = n / dt if dt > 0 else 0.0
            print(f"Approx FPS: {fps:.2f}")
            t0 = time.time()
            n = 0

        # ros_node.publish_seg(vis_bgr)
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        if not HEADLESS:
            cv2.imshow(win_track, vis_rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if not HEADLESS:
        cv2.destroyAllWindows()

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
