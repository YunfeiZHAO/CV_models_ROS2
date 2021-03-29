import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from interfaces.msg import CameraStream



class ResultSubscriber(Node):
    def __init__(self):
        super().__init__('result_detector')
        self.subscription = self.create_subscription(
            CameraStream,
            'camera_stream',
            self.result_callback,
            10)
        self.subscription  # prevent unused variable warning

    def result_callback(self, msg):
        # deserialize image
        im_shape = (msg.height, msg.width, msg.channel)
        im = np.frombuffer(msg.img, dtype=msg.pixel_type).reshape(im_shape)
        # Stream results
        cv2.imshow('webcam', im)
        cv2.waitKey(1)  # 1 millisecond


def main(args=None):

    # run in ros
    rclpy.init(args=args)
    result_subscriber = ResultSubscriber()
    rclpy.spin(result_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    result_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
