#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__("webcam_publisher")

        # Parameters (override with ROS params if you want)
        self.declare_parameter("device_index", 0)
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)
        self.declare_parameter("fps", 30)
        self.declare_parameter("topic", "/image")

        device_index = int(self.get_parameter("device_index").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)
        topic = str(self.get_parameter("topic").value)

        self.pub = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()

        # Open camera with V4L2 backend
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)

        # Apply requested settings
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam (VideoCapture({device_index})). "
                f"Check /dev/video{device_index} mapping (esp. in Docker)."
            )

        self.get_logger().info(
            "Opened camera: "
            f"W={self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, "
            f"H={self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, "
            f"FPS={self.cap.get(cv2.CAP_PROP_FPS)}, "
            f"FOURCC={int(self.cap.get(cv2.CAP_PROP_FOURCC))}"
        )

        # Timer rate: try to publish at target fps (best-effort)
        period = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.timer = self.create_timer(period, self.timer_cb)

    def timer_cb(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn("Failed to read frame from camera.")
            return

        # OpenCV gives BGR; publish as bgr8
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        self.pub.publish(msg)

    def destroy_node(self):
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
