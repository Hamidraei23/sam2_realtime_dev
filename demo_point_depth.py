import os
import time
from datetime import datetime
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn

from new.depth_anything_v2.dpt import DepthAnythingV2
from sam2.build_sam import build_sam2_camera_predictor

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2

def fill_holes_cv(mask_bool: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary mask using flood fill from the border.
    mask_bool: HxW bool
    returns: HxW bool with holes filled
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    h, w = mask_u8.shape

    # Invert: holes become foreground in inv
    inv = cv2.bitwise_not(mask_u8)

    # Flood fill from border on inv to mark "true background"
    flood = inv.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 0)  # remove border-connected background

    # Remaining white in flood are the holes (in inverted space)
    holes = flood

    # Invert holes back and OR with original to fill
    holes_inv = cv2.bitwise_not(holes)
    filled = cv2.bitwise_or(mask_u8, holes_inv)

    return filled > 0


def keep_component_containing_point(mask_bool: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Keeps only the connected component that contains (x,y).
    If (x,y) is not inside the mask, falls back to largest component.
    """
    mask_u8 = mask_bool.astype(np.uint8)
    num, labels = cv2.connectedComponents(mask_u8, connectivity=8)

    if num <= 1:
        return mask_bool

    if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
        lab = labels[y, x]
    else:
        lab = 0

    if lab != 0:
        return labels == lab

    # fallback: largest non-zero component
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    lab = int(np.argmax(counts))
    return labels == lab


def postprocess_mask(mask_bool: np.ndarray, x: int, y: int,
                     close_ksize: int = 11,
                     close_iters: int = 1,
                     do_fill_holes: bool = True,
                     do_keep_seed_cc: bool = True) -> np.ndarray:
    """
    Robust post-process:
    - closing to seal gaps
    - hole filling
    - keep connected component containing the click
    """
    m = mask_bool.astype(np.uint8) * 255

    # Closing seals small cracks/gaps (good for corner holes)
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iters)

    m = m > 0

    if do_fill_holes:
        m = fill_holes_cv(m)

    if do_keep_seed_cc:
        m = keep_component_containing_point(m, x, y)

    return m

def robust_bg_value(depth_raw: np.ndarray,
                    mask: np.ndarray,
                    trim: float = 0.05,
                    method: str = "trimmed_mean") -> float:
    """
    depth_raw: HxW float32
    mask: HxW bool (True = object)
    method: 'median' or 'trimmed_mean'
    """
    bg = depth_raw[~mask]
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        allv = depth_raw[np.isfinite(depth_raw)]
        return float(np.median(allv)) if allv.size else 0.0

    if method == "median":
        return float(np.median(bg))

    # trimmed mean (robust against outliers)
    lo = np.quantile(bg, trim)
    hi = np.quantile(bg, 1.0 - trim)
    bg_clip = np.clip(bg, lo, hi)
    return float(bg_clip.mean())


def ema(prev: float | None, new: float, alpha: float = 0.90) -> float:
    """alpha close to 1.0 = more smoothing (less jitter)."""
    return new if prev is None else (alpha * prev + (1.0 - alpha) * new)


# -----------------------
# DepthAnythingV2 helper
# -----------------------
class DepthEstimator:
    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(self, ckpt_path: str, encoder: str = "vits", device: str | None = None):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )

        if encoder not in self.model_configs:
            raise ValueError(f"Encoder '{encoder}' not supported. Choose from {list(self.model_configs.keys())}.")

        cfg = self.model_configs[encoder]
        model = DepthAnythingV2(**cfg)

        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device)

        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"[Depth] Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        self.model = model.eval()
        print(f"[Depth] Loaded on {self.device} ({encoder})")

    @torch.no_grad()
    @torch.no_grad()
    def infer_depth_raw(self, frame_bgr: np.ndarray) -> np.ndarray:
        base = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Disable autocast for DepthAnything so numpy conversion won't see bfloat16
        ctx = torch.autocast(device_type="cuda", enabled=False) if self.device == "cuda" else nullcontext()

        with ctx:
            depth = base.infer_image(frame_bgr)  # this returns numpy in your DepthAnything code

        return np.asarray(depth, dtype=np.float32).squeeze()


    @staticmethod
    def depth_to_u8(depth_f32: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
        """
        Robust normalization to uint8 for display.
        """
        lo, hi = np.percentile(depth_f32, [p_lo, p_hi])
        d = np.clip(depth_f32, lo, hi)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        return (d * 255).astype(np.uint8)


# -----------------------
# SAM2 + Depth two-stream
# -----------------------
def main(
    cam_index: int = 2,
    width: int = 640,
    height: int = 480,
    fps: int = 120.101,
    process_scale: float = 1.0,   # set to 0.5 if you want speed, click+tracking+depth will all use the scaled image
    save_dir: str = "./two_stream_saved",
):
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------
    # Precision / CUDA setup
    # -----------------------
    if torch.cuda.is_available():
        # prefer bfloat16 autocast for SAM2
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        # TF32 speedups on Ampere+
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # -----------------------
    # Build models
    # -----------------------
    # SAM2
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    print("[SAM2] Predictor ready")

    # DepthAnythingV2
    depth_ckpt = "./new/depth_anything_v2_vits.pth"
    depth_est = DepthEstimator(depth_ckpt, encoder="vits")

    # -----------------------
    # Camera
    # -----------------------
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print("W,H,FPS,FOURCC:",
          cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
          cap.get(cv2.CAP_PROP_FPS),
          int(cap.get(cv2.CAP_PROP_FOURCC)))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {cam_index}.")

    HEADLESS = (os.environ.get("DISPLAY", "") == "")

    ####### ROS NODE SETUP #######
    rclpy.init(args=None)
    ros_node = Node("sam_depth_pub")
    ros_pub = ros_node.create_publisher(Image, "/depth_bgr_vis", 10)
    ros_bridge = CvBridge()
    ##############################

    # -----------------------
    # Warm up and grab last frame for click
    # -----------------------
    last_frame_rgb = None
    K = np.array([
        [269.82775554,   0.0,        312.94152473],
        [  0.0,        269.49528176, 220.30212927],
        [  0.0,          0.0,          1.0       ],
    ], dtype=np.float64)

    D = np.array([[-0.04928117],
                  [ 0.02756253],
                  [-0.15501842],
                  [ 0.17920585]], dtype=np.float64)
    image_size = (width, height)
    R = np.eye(3, dtype=np.float64)

    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, R, balance=0.0, fov_scale=1.0
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, newK, image_size, m1type=cv2.CV_16SC2
    )
    for i in range(30):
        ret, frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {i+1}/30 from webcam.")

        # Your existing crop
        frame_bgr = cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
        frame_bgr = frame_bgr[20:320, 150:490, :]
        # print(f"frame shape after ROI: {frame_bgr.shape}")

        h, w, _ = frame_bgr.shape
        side = min(h, w)

        # Center square crop
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        frame_bgr = frame_bgr[y0:y0 + side, x0:x0 + side]

        # Resize to final square size
        target_size = 224  # or whatever you want
        frame_bgr = cv2.resize(
            frame_bgr,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA
        )

        last_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------
    # Click (x, y)
    # -----------------------
    clicked = {"done": False, "x": None, "y": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["x"], clicked["y"] = x, y
            clicked["done"] = True

    if HEADLESS:
        out_path = os.path.join(save_dir, "last_frame_for_click.png")
        cv2.imwrite(out_path, cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR))
        print(f"[HEADLESS] Saved last frame to: {out_path}")
        clicked["x"] = int(input("Enter x: ").strip())
        clicked["y"] = int(input("Enter y: ").strip())
        clicked["done"] = True
    else:
        win = "Click target to track (Left click). Press q to quit."
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

        display = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR)
        while True:
            vis = display.copy()

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
    print(f"[Click] Using point: ({x}, {y})")

    # -----------------------
    # Init SAM2 with first frame + prompt
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
    print("[SAM2] Prompt added. Tracking starts now.")

    # -----------------------
    # Live loop
    # -----------------------
    if not HEADLESS:
        cv2.namedWindow("RGB+Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth preserved(obj) + avg(background)", cv2.WINDOW_NORMAL)

    t_fps = time.time()
    n_fps = 0

    

    print("Controls: q=quit, s=save current outputs")
    bg_val_smoothed = None
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {i+1}/30 from webcam.")
        frame_bgr = cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
        # Your existing crop
        frame_bgr = frame_bgr[20:320, 150:490, :]
        # print(f"frame shape after ROI: {frame_bgr.shape}")

        h, w, _ = frame_bgr.shape
        side = min(h, w)

        # Center square crop
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        frame_bgr = frame_bgr[y0:y0 + side, x0:x0 + side]

        # Resize to final square size
        target_size = 224  # or whatever you want
        frame_bgr = cv2.resize(
            frame_bgr,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA
        )

        # last_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- SAM2 track ---
        out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

        # choose the mask for our object id if present; else fallback to first
        chosen_idx = None
        for i, oid in enumerate(out_obj_ids):
            if int(oid) == ann_obj_id:
                chosen_idx = i
                break
        if chosen_idx is None:
            chosen_idx = 0

        # mask: HxW bool
        mask_t = (out_mask_logits[chosen_idx] > -0.5)  # shape usually [1,H,W]
        mask = mask_t.squeeze().detach().cpu().numpy().astype(bool)

        mask_u8 = (mask.astype(np.uint8) * 255)
        cnts,_ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        mask_u8[:] = 0; cv2.fillConvexPoly(mask_u8, hull, 255); mask = mask_u8 > 0

        # --- Depth ---
        depth_raw = depth_est.infer_depth_raw(frame_bgr)  # HxW float32

        depth_vis = depth_est.depth_to_u8(depth_raw)

        # Safety: ensure same shape (should be, but just in case)
        if depth_raw.shape != mask.shape:
            # resize mask to depth shape
            mask = cv2.resize(mask.astype(np.uint8), (depth_raw.shape[1], depth_raw.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        # mask_u8 = mask.astype(np.uint8) * 255
        kernel = np.ones((7, 7), np.uint8)           # tweak size: 5..11
        mask_dil = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)

        # 2) Robust background value (median or trimmed mean)
        bg_val_frame = robust_bg_value(depth_raw, mask_dil, trim=0.10, method="trimmed_mean")

        # 3) Temporal smoothing (keep bg_val_smoothed across frames!)
        bg_val_smoothed = ema(bg_val_smoothed, bg_val_frame, alpha=0.92)

        depth_modified = depth_raw.copy()
        depth_modified[~mask] = bg_val_smoothed # background becomes average depth

        # Visualize depth_modified as grayscale
        depth_u8 = depth_est.depth_to_u8(depth_modified)
        

        depth_u8[mask] = depth_vis[mask]


        depth_bgr_vis = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

        # Mask overlay on RGB for sanity-check
        overlay = frame_rgb.copy()
        overlay_mask = np.zeros_like(overlay, dtype=np.uint8)
        overlay_mask[..., 1] = 255  # green
        overlay[mask] = cv2.addWeighted(overlay[mask], 0.5, overlay_mask[mask], 0.5, 0)

        H, W = mask.shape[:2]
        ys, xs = np.where(mask)
        # cx = int(xs.mean()) if xs.size else (W // 2)

        # side = min(H, W)  # will be H if W >= H
        # x0 = np.clip(cx - side // 2, 0, W - side)
        # x1 = x0 + side

        # overlay = overlay[:, x0:x1]
        depth_bgr_vis = depth_bgr_vis

        # Publish depth_bgr_vis via ROS
        img = np.ascontiguousarray(depth_bgr_vis)   # important!
        msg = Image()
        msg.header.stamp = ros_node.get_clock().now().to_msg()
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        ros_pub.publish(msg)
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        msg.header.stamp = ros_node.get_clock().now().to_msg()
        ros_pub.publish(msg)
        rclpy.spin_once(ros_node, timeout_sec=0.0)


        # draw the original clicked point
        # cv2.circle(overlay, (x, y), 6, (0, 255, 0), -1)

        # show
        if not HEADLESS:
            cv2.imshow("RGB+Mask", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imshow("Depth preserved(obj) + avg(background)", depth_bgr_vis)
            # print(f"image shape is {depth_bgr_vis.shape}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fn_rgb = os.path.join(save_dir, f"rgb_{ts}.png")
                fn_overlay = os.path.join(save_dir, f"overlay_{ts}.png")
                fn_depth = os.path.join(save_dir, f"depth_modified_{ts}.png")

                cv2.imwrite(fn_rgb, frame_bgr)
                cv2.imwrite(fn_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite(fn_depth, depth_bgr_vis)
                print(f"[Saved] {fn_rgb}")
                print(f"[Saved] {fn_overlay}")
                print(f"[Saved] {fn_depth}")

        # FPS print every ~30 frames
        n_fps += 1
        if n_fps % 30 == 0:
            elapsed = time.time() - t_fps
            fps_now = n_fps / elapsed if elapsed > 0 else 0.0
            print(f"[FPS] ~{fps_now:.2f}")
            print(f"shape info: frame {frame_bgr.shape}, depth {depth_raw.shape}")
            t_fps = time.time()
            n_fps = 0

    cap.release()
    ros_node.destroy_node()
    rclpy.shutdown()
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        cam_index=2,
        width=840,
        height=480,
        fps=60,
        process_scale=1.0,   # try 0.5 if you need speed
        save_dir="./two_stream_saved",
    )
