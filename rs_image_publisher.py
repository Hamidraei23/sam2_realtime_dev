#!/usr/bin/env python3
import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class RealSensePublisher(Node):
    def __init__(self):
        super().__init__("realsense_image_publisher")

        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("topic", "/image")

        w = int(self.get_parameter("width").value)
        h = int(self.get_parameter("height").value)
        fps = int(self.get_parameter("fps").value)
        topic = str(self.get_parameter("topic").value)

        # RealSense pipeline
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)

        try:
            profile = self._pipeline.start(config)
        except RuntimeError as e:
            self.get_logger().error(f"Failed to start RealSense: {e}")
            raise

        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.get_logger().info(
            f"RealSense color stream: {cprof.width()}x{cprof.height()} @ {cprof.fps()} fps"
        )

        # Publisher
        self._pub = self.create_publisher(Image, topic, 10)
        self.get_logger().info(f"Publishing on: {topic} [sensor_msgs/Image, rgb8]")

        # Timer at target fps
        period = 1.0 / fps
        self._timer = self.create_timer(period, self._timer_cb)

    def _timer_cb(self):
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        frame = np.asanyarray(color_frame.get_data())  # rgb8
        h, w = frame.shape[:2]

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = frame.tobytes()

        self._pub.publish(msg)

    def destroy_node(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = RealSensePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
