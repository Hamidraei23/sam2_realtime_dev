import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

class Enc(Node):
    def __init__(self):
        super().__init__("print_image_encoding")
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.cb, qos_profile_sensor_data)

    def cb(self, msg: Image):
        self.get_logger().info(f"encoding = {msg.encoding}, step = {msg.step}, WxH = {msg.width}x{msg.height}")
        rclpy.shutdown()

def main():
    rclpy.init()
    rclpy.spin(Enc())

if __name__ == "__main__":
    main()
