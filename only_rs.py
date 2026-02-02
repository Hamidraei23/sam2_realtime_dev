import pyrealsense2 as rs
import numpy as np
import cv2

W, H, FPS = 640, 480, 30   # <-- change these

pipeline = rs.pipeline()
config = rs.config()

# Request exact modes (must be supported by your device)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)  # depth is usually Z16
# Start streaming
try:
    profile = pipeline.start(config)
except RuntimeError as e:
    print("Failed to start with requested mode:", e)
    print("Tip: print supported profiles (see function below) and pick one.")
    raise

# Optional: align depth to color
align = rs.align(rs.stream.color)

# Verify what you actually got
cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
print("COLOR:", cprof.width(), cprof.height(), cprof.fps(), cprof.format())
print("DEPTH:", dprof.width(), dprof.height(), dprof.fps(), dprof.format())

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
        )
        both = np.hstack((color, depth_vis))
        cv2.imshow("RealSense (color | depth)", both)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
