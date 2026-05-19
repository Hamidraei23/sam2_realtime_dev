import os
import cv2
import numpy as np
import torch
import time
from collections import defaultdict
from contextlib import contextmanager

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Pose, PoseArray, PointStamped
from sensor_msgs.msg import Image


PROFILE_TIMERS = os.environ.get("SAM2_PROFILE_TIMERS", "1") != "0"
PROFILE_SYNC_CUDA = os.environ.get("SAM2_PROFILE_SYNC_CUDA", "1") != "0"
PROFILE_PRINT_EVERY = int(os.environ.get("SAM2_PROFILE_PRINT_EVERY", "60"))

def _profile_sync_cuda():
    if PROFILE_SYNC_CUDA and torch.cuda.is_available():
        torch.cuda.synchronize()

@contextmanager
def profile_section(profile, key):
    if not PROFILE_TIMERS:
        yield
        return
    _profile_sync_cuda()
    start = time.perf_counter()
    try:
        yield
    finally:
        _profile_sync_cuda()
        profile[key] = profile.get(key, 0.0) + (time.perf_counter() - start) * 1000.0

class RollingProfiler:
    def __init__(self):
        self.totals = defaultdict(float)
        self.count = 0

    def update(self, profile):
        if not PROFILE_TIMERS:
            return
        self.count += 1
        for key, value in profile.items():
            if isinstance(value, (int, float)):
                self.totals[key] += float(value)

    def reset(self):
        self.totals.clear()
        self.count = 0

    def format_line(self, fps):
        if not PROFILE_TIMERS or self.count == 0:
            return None

        def avg(key):
            return self.totals.get(key, 0.0) / self.count

        keys = [
            "loop_total_ms",
            "ros_spin_ms",
            "pop_frame_ms",
            "predictor_wall_ms",
            "track_total_ms",
            "prepare_data_ms",
            "get_feature_total_ms",
            "image_to_cuda_ms",
            "image_encoder_ms",
            "feature_expand_ms",
            "prepare_backbone_features_ms",
            "track_step_total_ms",
            "track_step_core_ms",
            "high_res_features_ms",
            "memory_conditioning_ms",
            "memory_attention_ms",
            "sam_heads_total_ms",
            "sam_prompt_encoder_ms",
            "sam_mask_decoder_ms",
            "sam_mask_postprocess_ms",
            "obj_ptr_ms",
            "encode_memory_output_ms",
            "memory_encoder_forward_ms",
            "memory_encoder_post_ms",
            "compact_output_ms",
            "orig_res_output_ms",
            "masks_cpu_ms",
            "overlay_mask_ms",
            "centers_ms",
            "publish_ms",
            "visualization_ms",
            "gui_ms",
        ]
        parts = [
            f"TIMERS avg{self.count}",
            f"fps={fps:.2f}",
            f"objs={avg('num_objects'):.1f}",
        ]
        parts.extend(
            f"{key[:-3] if key.endswith('_ms') else key}={avg(key):.2f}ms"
            for key in keys
            if key in self.totals
        )
        return " | ".join(parts)

class CenterPublisher(Node):
    """
    - Subscribes to image_topic (expects bgr8) WITHOUT cv_bridge (NumPy2-safe)
    - Publishes all tracked object poses as PoseArray on /tracked_objects
    - Publishes visualization/seg overlay as sensor_msgs/Image (bgr8) on /image_seg
    """
    def __init__(self, image_topic: str = "/camera/camera/color/image_raw"):
        super().__init__("sam2_tracker_node")
        self._qos = QoSProfile(depth=10)

        # Image subscription state
        self._latest_bgr = None
        self._new_frame = False
        self._last_header = None
        self._image_topic = image_topic

        # Subscribe to camera topic (use default reliable QoS to match most publishers)
        self._img_sub = self.create_subscription(
            Image,
            self._image_topic,
            self._on_image,
            10,
        )
        self.get_logger().info(
            f"Subscribed to image topic: {self._image_topic} [sensor_msgs/Image, bgr8]"
        )

        # Publish segmented/overlay image
        self._seg_pub = self.create_publisher(Image, "/image_seg", qos_profile_sensor_data)
        self.get_logger().info("Publishing segmentation overlay on: /image_seg [sensor_msgs/Image, bgr8]")

        # Single PoseArray publisher for all tracked objects (excluding hand)
        self._pose_pub = self.create_publisher(PoseArray, "/tracked_objects_a", self._qos)
        self.get_logger().info("Publishing tracked objects on: /tracked_objects_a [PoseArray]")

        # Separate hand publisher
        self._hand_pub = self.create_publisher(PointStamped, "/hand_center_a", self._qos)
        self.get_logger().info("Publishing hand center on: /hand_center_a [PointStamped]")

        # Subscribe to external tracked objects from another system
        self._ext_poses = []  # list of (cx, cy) pixel coords
        self._ext_sub = self.create_subscription(
            PoseArray,
            "/tracked_objects_a",
            self._on_tracked_objects_a,
            self._qos,
        )
        self.get_logger().info("Subscribed to external objects on: /tracked_objects_a [PoseArray]")

    # -----------------------
    # Publish all object poses in one PoseArray
    # -----------------------
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
            # Encode yaw as quaternion (rotation about z)
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            p.orientation.z = float(np.sin(yaw_rad / 2.0))
            p.orientation.w = float(np.cos(yaw_rad / 2.0))
            msg.poses.append(p)

        self._pose_pub.publish(msg)

    def _on_tracked_objects_a(self, msg: PoseArray):
        self._ext_poses = [(p.position.x, p.position.y) for p in msg.poses]

    def publish_hand(self, cx: float, cy: float, frame_id: str = "image"):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = float(cx)
        msg.point.y = float(cy)
        msg.point.z = 0.0
        self._hand_pub.publish(msg)

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
from sam2_realtime_config import SAM2_REALTIME_PROFILE


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _append_compile_overrides(
    hydra_overrides_extra,
    profile,
    *,
    env_prefix: str,
    profile_key: str,
    model_key: str,
):
    """Append Hydra overrides for an optional torch.compile target.

    The SAM2 YAML does not explicitly declare these fork-specific compile keys,
    so all compile options are added with ++model.* overrides.
    """
    if not _env_bool(env_prefix, profile.get(profile_key, False)):
        return

    hydra_overrides_extra.append(f"++model.{model_key}=true")

    fullgraph = _env_bool(
        f"{env_prefix}_FULLGRAPH",
        profile.get(f"{profile_key}_fullgraph", False),
    )
    dynamic = _env_bool(
        f"{env_prefix}_DYNAMIC",
        profile.get(f"{profile_key}_dynamic", False),
    )
    mode = os.environ.get(
        f"{env_prefix}_MODE",
        profile.get(f"{profile_key}_mode", "max-autotune-no-cudagraphs"),
    )

    hydra_overrides_extra.append(
        f"++model.{model_key}_fullgraph={str(bool(fullgraph)).lower()}"
    )
    hydra_overrides_extra.append(
        f"++model.{model_key}_dynamic={str(bool(dynamic)).lower()}"
    )
    hydra_overrides_extra.append(f"++model.{model_key}_mode={mode}")


def _build_hydra_overrides(profile):
    hydra_overrides_extra = []

    if _env_bool(
        "SAM2_COMPILE_IMAGE_ENCODER",
        profile["compile_image_encoder"],
    ):
        hydra_overrides_extra.append("++model.compile_image_encoder=true")

    _append_compile_overrides(
        hydra_overrides_extra,
        profile,
        env_prefix="SAM2_COMPILE_MEMORY_ATTENTION",
        profile_key="compile_memory_attention",
        model_key="compile_memory_attention",
    )
    _append_compile_overrides(
        hydra_overrides_extra,
        profile,
        env_prefix="SAM2_COMPILE_MEMORY_ENCODER",
        profile_key="compile_memory_encoder",
        model_key="compile_memory_encoder",
    )
    _append_compile_overrides(
        hydra_overrides_extra,
        profile,
        env_prefix="SAM2_COMPILE_SAM_MASK_DECODER",
        profile_key="compile_sam_mask_decoder",
        model_key="compile_sam_mask_decoder",
    )

    image_size = _env_int("SAM2_IMAGE_SIZE", profile["image_size"])
    if image_size is not None:
        hydra_overrides_extra.append(f"model.image_size={int(image_size)}")

    memory_temporal_stride = _env_int(
        "SAM2_MEMORY_TEMPORAL_STRIDE",
        profile["memory_temporal_stride_for_eval"],
    )
    if memory_temporal_stride is not None:
        memory_temporal_stride = int(memory_temporal_stride)
        if memory_temporal_stride < 1:
            raise ValueError(
                f"SAM2_MEMORY_TEMPORAL_STRIDE must be >= 1; "
                f"got {memory_temporal_stride}"
            )
        # This key is accepted by SAM2Base.__init__, but it is not explicitly
        # present in sam2.1_hiera_s.yaml. Use ++ so Hydra can add it cleanly.
        hydra_overrides_extra.append(
            f"++model.memory_temporal_stride_for_eval={memory_temporal_stride}"
        )

    return hydra_overrides_extra


def _apply_runtime_num_maskmem(predictor, value):
    """
    Reduce SAM2's runtime memory window after checkpoint loading.

    Do not pass model.num_maskmem as a Hydra override before loading the
    checkpoint: SAM2.1 checkpoints contain maskmem_tpos_enc with first dimension
    equal to the training/config value, normally 7. Reducing it before loading
    causes a checkpoint shape mismatch.

    Here we load the checkpoint normally, then keep the most recent temporal
    positional embeddings:
      old shape [7, 1, 1, C]
      new shape [N, 1, 1, C] = old[-N:]
    """
    if value in (None, ""):
        return

    new_num_maskmem = int(value)
    old_num_maskmem = int(predictor.num_maskmem)

    if new_num_maskmem == old_num_maskmem:
        print(f"SAM2 runtime num_maskmem: {old_num_maskmem} unchanged")
        return

    if new_num_maskmem < 1:
        raise ValueError(
            f"SAM2_NUM_MASKMEM must be >= 1 for video tracking; got {new_num_maskmem}"
        )

    if new_num_maskmem > old_num_maskmem:
        raise ValueError(
            f"SAM2_NUM_MASKMEM={new_num_maskmem} is larger than checkpoint/config "
            f"num_maskmem={old_num_maskmem}. This runtime override only supports "
            f"reducing num_maskmem."
        )

    old_tpos = predictor.maskmem_tpos_enc
    new_tpos = old_tpos[-new_num_maskmem:].detach().clone()

    predictor.num_maskmem = new_num_maskmem
    predictor.maskmem_tpos_enc = torch.nn.Parameter(
        new_tpos,
        requires_grad=old_tpos.requires_grad,
    )

    print(
        "SAM2 runtime num_maskmem override:",
        f"{old_num_maskmem} -> {new_num_maskmem}",
        "| maskmem_tpos_enc:",
        tuple(old_tpos.shape),
        "->",
        tuple(predictor.maskmem_tpos_enc.shape),
    )


sam2_checkpoint = os.environ.get(
    "SAM2_CHECKPOINT",
    SAM2_REALTIME_PROFILE["checkpoint"],
)
model_cfg = os.environ.get(
    "SAM2_MODEL_CFG",
    SAM2_REALTIME_PROFILE["model_cfg"],
)
hydra_overrides_extra = _build_hydra_overrides(SAM2_REALTIME_PROFILE)

print("SAM2 checkpoint:", sam2_checkpoint)
print("SAM2 config:", model_cfg)
print("SAM2 runtime profile:", SAM2_REALTIME_PROFILE["name"])
print("SAM2 Hydra overrides:", hydra_overrides_extra)

predictor = build_sam2_camera_predictor(
    model_cfg,
    sam2_checkpoint,
    hydra_overrides_extra=hydra_overrides_extra,
)
_apply_runtime_num_maskmem(
    predictor,
    _env_int("SAM2_NUM_MASKMEM", SAM2_REALTIME_PROFILE["runtime_num_maskmem"]),
)

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
    ros_node = CenterPublisher(image_topic="/image")

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
    win_track = "Tracking (press q to quit | S=record)"
    # if not HEADLESS:
    #     cv2.namedWindow(win_track, cv2.WINDOW_NORMAL)

    t0 = time.time()
    n = 0
    profile_acc = RollingProfiler()
    video_writer = None
    is_recording = False
    record_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmented.mp4")

    track_hand = False  # if False, first click is treated as a regular object, not hand

    ref_areas = {}         # oid -> pixel area from first tracking frame (excluding hand)
    ref_areas_set = False
    last_good_centers = {} # oid -> (cx, cy)
    AREA_MIN_RATIO = 0.5


    while rclpy.ok():
        frame_profile = {}
        loop_start = time.perf_counter()

        # Pump ROS to receive images
        with profile_section(frame_profile, "ros_spin_ms"):
            rclpy.spin_once(ros_node, timeout_sec=0.0)

        with profile_section(frame_profile, "pop_frame_ms"):
            frame_bgr = ros_node.pop_latest_frame()
        if frame_bgr is None:
            continue

        frame_rgb = frame_bgr  # _imgmsg_to_rgb8 already returns RGB
        with profile_section(frame_profile, "predictor_wall_ms"):
            out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

        # Add internal SAM2 timings collected inside SAM2CameraPredictor.track().
        frame_profile.update(getattr(predictor, "last_track_profile", {}) or {})

        out_ids = out_obj_ids.tolist() if hasattr(out_obj_ids, "tolist") else list(out_obj_ids)
        frame_profile["num_objects"] = float(len(out_ids))

        # Download all masks from GPU once per frame
        with profile_section(frame_profile, "masks_cpu_ms"):
            masks_np = {}   # oid -> (H, W) uint8 numpy array (0 or 255)
            masks_bool = {} # oid -> (H, W) bool CPU tensor
            for j, oid in enumerate(out_ids):
                m_bool = (out_mask_logits[j] > 0.0).squeeze().cpu()
                masks_bool[oid] = m_bool
                masks_np[oid] = m_bool.numpy().astype(np.uint8) * 255

        # Overlay masks (multi-object HSV coloring)
        with profile_section(frame_profile, "overlay_mask_ms"):
            H, W = frame_rgb.shape[:2]
            all_mask = np.zeros((H, W, 3), dtype=np.uint8)
            all_mask[..., 1] = 255

            for j, oid in enumerate(out_ids):
                out_mask = masks_np[oid]
                hue = int((j + 3) / (len(out_ids) + 3) * 255)
                all_mask[out_mask == 255, 0] = hue
                all_mask[out_mask == 255, 2] = 255

        # Centers + area tracking + sticky publish
        with profile_section(frame_profile, "centers_ms"):
            current_area = {}
            centers = {}

            for oid in out_ids:
                m = masks_bool[oid]
                area = int(m.sum().item())
                current_area[oid] = area

                if area == 0:
                    centers[oid] = None
                    continue

                ys_xs = torch.nonzero(m, as_tuple=False)  # [N,2] (y,x)
                cy, cx = ys_xs.float().mean(dim=0)
                centers[oid] = (float(cx.item()), float(cy.item()))  # (x,y)

            # Reference areas ONCE (first tracking frame)
            if not ref_areas_set:
                for oid in out_ids:
                    if track_hand and oid == 1:
                        continue
                    a = current_area.get(oid, 0)
                    if a > 0:
                        ref_areas[oid] = a
                ref_areas_set = True
                ros_node.get_logger().info(f"Reference areas (first frame only{', excluding hand' if track_hand else ''}): {ref_areas}")

        # Build poses list for non-hand objects; publish hand separately
        with profile_section(frame_profile, "publish_ms"):
            poses_to_publish = []
            for oid in out_ids:
                c_now = centers.get(oid)
                a_now = current_area.get(oid, 0)

                if track_hand and oid == 1:
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
                    poses_to_publish.append((cx, cy, 0.0))
                else:
                    cx, cy = last_good_centers.get(oid, (0.0, 0.0))
                    poses_to_publish.append((cx, cy, 0.0))

            if poses_to_publish:
                ros_node.publish_poses(poses_to_publish, frame_id="image")

        # Visualization
        with profile_section(frame_profile, "visualization_ms"):
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

            # Draw external poses from /tracked_objects_a (orange circles, RGB space)
            # for ext_cx, ext_cy in ros_node._ext_poses:
            #     cv2.circle(vis_rgb, (int(ext_cx), int(ext_cy)), 10, (255, 140, 0), 2)

            vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

            if is_recording and video_writer is not None:
                video_writer.write(vis_rgb)

        # ros_node.publish_seg(vis_bgr)
        with profile_section(frame_profile, "gui_ms"):
            if not HEADLESS:
                cv2.imshow(win_track, vis_rgb)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("s"), ord("S")):
                    if not is_recording:
                        H_rec, W_rec = vis_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(record_path, fourcc, 5.5, (W_rec, H_rec))
                        is_recording = True
                        print(f"[REC] Recording started -> {record_path}")
                    else:
                        is_recording = False
                        video_writer.release()
                        video_writer = None
                        print(f"[REC] Recording stopped. Saved: {record_path}")

        frame_profile["loop_total_ms"] = (time.perf_counter() - loop_start) * 1000.0
        profile_acc.update(frame_profile)

        n += 1
        if n % PROFILE_PRINT_EVERY == 0:
            dt = time.time() - t0
            fps = n / dt if dt > 0 else 0.0
            print(f"Approx FPS: {fps:.2f}")
            timer_line = profile_acc.format_line(fps)
            if timer_line:
                print(timer_line)
            t0 = time.time()
            n = 0
            profile_acc.reset()
    if is_recording and video_writer is not None:
        video_writer.release()
        print(f"[REC] Recording saved on exit: {record_path}")

    if not HEADLESS:
        cv2.destroyAllWindows()

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
