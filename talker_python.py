#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Talker(Node):
    def __init__(self) -> None:
        super().__init__("py_talker")
        self.pub = self.create_publisher(String, "chatter", 10)
        self.i = 0
        self.timer = self.create_timer(0.5, self.on_timer)  # 2 Hz
        self.get_logger().info("Talker started: publishing on /chatter")

    def on_timer(self) -> None:
        msg = String()
        msg.data = f"hello {self.i}"
        self.pub.publish(msg)
        self.get_logger().info(f"published: {msg.data}")
        self.i += 1


def main() -> None:
    rclpy.init()
    node = Talker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

