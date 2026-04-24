#!/usr/bin/env python3
import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class RealSensePublisher(Node):
    def __init__(self):
        super().__init__("realsense_image_publisher")

        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 60)
        self.declare_parameter("topic", "/image")
        self.declare_parameter("camera_info_topic", "/camera_info")

        w = int(self.get_parameter("width").value)
        h = int(self.get_parameter("height").value)
        fps = int(self.get_parameter("fps").value)
        topic = str(self.get_parameter("topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)

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
        self._camera_info_msg = self._make_camera_info(cprof)
        self.get_logger().info(
            f"RealSense color stream: {cprof.width()}x{cprof.height()} @ {cprof.fps()} fps"
        )

        # Publishers
        self._pub = self.create_publisher(Image, topic, 10)
        self._camera_info_pub = self.create_publisher(CameraInfo, camera_info_topic, 10)
        self.get_logger().info(f"Publishing on: {topic} [sensor_msgs/Image, rgb8]")
        self.get_logger().info(
            f"Publishing on: {camera_info_topic} [sensor_msgs/CameraInfo]"
        )

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

        self._camera_info_msg.header.stamp = msg.header.stamp
        self._camera_info_msg.header.frame_id = msg.header.frame_id

        self._pub.publish(msg)
        self._camera_info_pub.publish(self._camera_info_msg)

    def _make_camera_info(self, color_profile):
        intr = color_profile.get_intrinsics()

        msg = CameraInfo()
        msg.width = intr.width
        msg.height = intr.height
        msg.distortion_model = self._distortion_model_name(intr.model)
        msg.d = list(intr.coeffs)
        msg.k = [
            intr.fx,
            0.0,
            intr.ppx,
            0.0,
            intr.fy,
            intr.ppy,
            0.0,
            0.0,
            1.0,
        ]
        msg.r = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        msg.p = [
            intr.fx,
            0.0,
            intr.ppx,
            0.0,
            0.0,
            intr.fy,
            intr.ppy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return msg

    @staticmethod
    def _distortion_model_name(model):
        if model == rs.distortion.kannala_brandt4:
            return "equidistant"
        return "plumb_bob"

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
