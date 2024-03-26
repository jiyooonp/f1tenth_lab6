#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid

from occupancy_grid import OccupancyGridPublisher

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Quaternion, TransformStamped, Twist


# TODO: import as you need

# class def for tree nodes
# It's up to you if you want to use this
class MyNode(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# class def for RRT
class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        pose_topic = "ego_racecar/odom"
        scan_topic = "/scan"

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # TODO: create subscribers
        self.pose_sub_ = self.create_subscription(
            #PoseStamped,
            Odometry,
            pose_topic,
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need

        # class attributes
        # TODO: maybe create your occupancy grid here
        self.publisher_ = self.create_publisher(
            OccupancyGrid, '/occupancy_grid', 10)
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 150
        self.grid = np.zeros((self.grid_size, self.grid_size),
                             dtype=np.int8)  # Occupancy grid data
        self.angular = Twist().angular

        # Transformation broadcaster for setting the orientation
        self.tf_broadcaster = TransformBroadcaster(self)

    def scan_callback(self, msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
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

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """

        return None

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        x = None
        y = None
        return (x, y)

    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        nearest_node = 0
        return nearest_node

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        new_node = None
        return new_node

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        return True

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        return False

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = []
        return path



    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
