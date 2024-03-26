#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import math
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from tf2_ros import TransformBroadcaster


class OccupancyGridPublisher(Node):

    def __init__(self):
        super().__init__('occupancy_grid_publisher')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10)
        self.publisher_ = self.create_publisher(
            OccupancyGrid, '/occupancy_grid', 10)
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 50  
        self.grid = np.zeros((self.grid_size, self.grid_size),
                             dtype=np.int8)  # Occupancy grid data
        self.angular = Twist().angular

        # Transformation broadcaster for setting the orientation
        self.tf_broadcaster = TransformBroadcaster(self)

    def odom_callback(self, msg):
        self.angular = msg.twist.twist.angular

    def lidar_callback(self, msg):
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Convert ranges to x, y coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * -np.sin(angles)

        # Convert coordinates to grid indices
        grid_x = np.clip(np.floor((x_coords / self.grid_resolution) +
                         (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)
        grid_y = np.clip(np.floor((y_coords / self.grid_resolution) +
                         (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)

        # Update occupancy grid
        self.grid.fill(0)
        self.grid[grid_x, grid_y] = 100  # Mark occupied cells as 100

        # Publish occupancy grid
        self.publish_grid()

    def publish_grid(self):
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = 'ego_racecar/base_link'
        occupancy_grid_msg.info.map_load_time = self.get_clock().now().to_msg()
        occupancy_grid_msg.info.resolution = self.grid_resolution
        occupancy_grid_msg.info.width = self.grid_size
        occupancy_grid_msg.info.height = self.grid_size
        occupancy_grid_msg.info.origin.position.x = - \
            self.grid_size / 2 * self.grid_resolution
        occupancy_grid_msg.info.origin.position.y = - \
            self.grid_size / 2 * self.grid_resolution
        occupancy_grid_msg.info.origin.position.z = 0.0

        # Set the orientation using a Quaternion
        angular_x = self.angular.x
        angular_y = self.angular.y
        angular_z = self.angular.z

        roll = 0.0
        pitch = 0.0
        yaw = math.atan2(angular_y, angular_x)
        q = self.euler_to_quaternion(roll, pitch, yaw)
        occupancy_grid_msg.info.origin.orientation = q

        # Rotate the grid data based on the car's orientation
        rotated_grid = np.rot90(self.grid, k=1)  # Rotate 180 degrees

        occupancy_grid_msg.data = np.ravel(rotated_grid).tolist()

        self.publisher_.publish(occupancy_grid_msg)

        # Broadcast the transform with the corrected orientation
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'ego_racecar/base_link'
        transform.child_frame_id = 'ego_racecar/base_link' 
        transform.transform.rotation = q
        self.tf_broadcaster.sendTransform(transform)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr

        quaternion = Quaternion()
        quaternion.w = qw
        quaternion.x = qx
        quaternion.y = qy
        quaternion.z = qz

        return quaternion
def main(args=None):
    rclpy.init(args=args)
    occupancy_grid_publisher = OccupancyGridPublisher()
    rclpy.spin(occupancy_grid_publisher)
    occupancy_grid_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
