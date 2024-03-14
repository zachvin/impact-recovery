#!/home/zach/ros2_ws/venv/lib/python3.10/site-packages

# Zach Vincent
# Some code copied from ROS docs at https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html

import rclpy
from rclpy.node import Node

# INPUT
# IMU:
#   linear velocity (x, y, z)
#   linear acceleration (x, y, z)
#   angular velocity (r, p, w)

# OUTPUT
# PWM? 

# numpy, Torch

from sensor_msgs.msg import Imu
import torch.nn as nn

class ImuListener(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Imu,
            '/copter_imu',
            self.listener_callback,
            1)
        self.subscription  # prevent unused variable warning
        self.f = open("testdata.txt", "w")
        self.f = open("testdata.txt", "a")
        

    def listener_callback(self, msg):
        # self.get_logger().info('IMU listener: "%s"' % msg.data)
        # imu: orientation, angular_velocity, linear_acceleration: x, y, z
        print(msg.linear_acceleration.z)
        self.f.write(str(msg.linear_acceleration.z) + "\n")

def main(args=None):
    rclpy.init(args=args)

    listener = ImuListener()

    rclpy.spin(listener)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
