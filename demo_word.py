import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

# -----------------------
# Device / precision
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# -----------------------
# SAM2 camera predictor
# -----------------------
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# -----------------------
# GroundingDINO (local)
# -----------------------
from grounding_dino.groundingdino.util.inference import load_model, predict
import grounding_dino.groundingdino.datasets.transforms as T

GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = "face."  # GroundingDINO caption helper expects/ensures trailing '.' :contentReference[oaicite:3]{index=3}

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)

# Same preprocessing used by GroundingDINO inference helpers :contentReference[oaicite:4]{index=4}
gdino_transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def detect_hand_boxes_xyxy(frame_rgb: np.ndarray):
    """
    Returns Nx4 xyxy boxes in pixel coords (float32).
    """
    pil = Image.fromarray(frame_rgb)
    img_t, _ = gdino_transform(pil, None)  # torch.Tensor, CHW

    boxes_cxcywh, scores, phrases = predict(
        model=grounding_model,
        image=img_t,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    if boxes_cxcywh is None or boxes_cxcywh.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32), [], []

    h, w, _ = frame_rgb.shape
    boxes_abs = boxes_cxcywh * torch.tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
    return boxes_xyxy.astype(np.float32), scores.cpu().numpy().tolist(), phrases

# -----------------------
# Camera + (optional) headless output
# -----------------------
cap = cv2.VideoCapture(0)

HEADLESS = os.environ.get("DISPLAY", "") == ""
writer = None

if_init = False
tracking_i = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape

    if not if_init:
        predictor.load_first_frame(frame_rgb)

        # Detect "hand" on the first frame, then register each detection as a tracked object
        boxes_xyxy, scores, phrases = detect_hand_boxes_xyxy(frame_rgb)

        if boxes_xyxy.shape[0] == 0:
            print("No 'hand' detected on first frame. Try adjusting thresholds or lighting.")
            # You can either continue looping until it detects, or break.
            # continue
        else:
            # Create an output writer if we're in a headless container
            if HEADLESS and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if fps and fps > 1 else 30
                writer = cv2.VideoWriter("hand_tracking.mp4", fourcc, fps, (w, h))

            for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy, start=1):
                bbox_prompt = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=i,
                    bbox=bbox_prompt,
                )

            if_init = True

        out_obj_ids = []
        out_mask_logits = []

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame_rgb)
        tracking_i += 1

    # --- visualize masks (same spirit as your original code) ---
    vis_rgb = frame_rgb.copy()

    if if_init and len(out_obj_ids) > 0:
        all_mask = np.zeros((h, w, 3), dtype=np.uint8)
        all_mask[..., 1] = 255  # saturation

        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255

            hue = (i + 3) / (len(out_obj_ids) + 3) * 255
            all_mask[out_mask[..., 0] == 255, 0] = hue
            all_mask[out_mask[..., 0] == 255, 2] = 255

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
        vis_rgb = cv2.addWeighted(vis_rgb, 1.0, all_mask, 0.5, 0)

    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    # Headless-safe: write video instead of imshow
    if writer is not None:
        writer.write(vis_bgr)

    if not HEADLESS:
        cv2.imshow("frame", vis_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
