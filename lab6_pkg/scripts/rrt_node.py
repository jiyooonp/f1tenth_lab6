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
from geometry_msgs.msg import PoseStamped, PointStamped, Pose, Point, Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from tf2_ros import TransformBroadcaster

import time


class MyTree(object):
    def __init__(self):
        self.vertices = {}


class MyPoint(object):
    def __init__(self):
        self.id = None
        self.x = None
        self.y = None
        self.parent = None


class MyEdge(object):
    def __init__(self):
        self.id = None
        self.start = None
        self.end = None


class RRT(Node):

    def __init__(self):
        super().__init__('rrt_node')

        # topics, not saved as attributes
        pose_topic = "ego_racecar/odom"
        scan_topic = "/scan"

        # Subscribers
        self.pose_sub_ = self.create_subscription(
            Odometry,
            pose_topic,
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)
        
        self.pure_pursuit_goal_sub = self.create_subscription(
            Point,
            '/pure_pursuit_goal',
            self.pure_pursuit_goal_callback,
            10)

        # Publishers
        self.goal_publisher = self.create_publisher(
            PointStamped, '/local_goal', 10)
        
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)
    
        # Visualization
        self.publisher_ = self.create_publisher(
            OccupancyGrid, 
            '/occupancy_grid', 
            10)
        self.random_point_pub = self.create_publisher(
            Marker,
            '/random_point',
            10)
        self.next_point_pub = self.create_publisher(
            Marker,
            '/next_point',
            10)
        self.target_line_pub = self.create_publisher(
            Marker,
            '/target_line',
            10)
        self.marker_array_pub = self.create_publisher(
            MarkerArray,
            '/marker_array',
            10)
        
        self.marker_history = []

        # Occupancy grid variables
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 60
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8) 

        # self.local_goal = np.random.rand(2) * self.grid_size*self.grid_resolution - self.grid_size*self.grid_resolution/2
        self.local_goal = [2., 0.]
        
        # Transformation broadcaster for setting the orientation
        self.tf_broadcaster = TransformBroadcaster(self)
        self.angular = Twist().angular

        # RRT variables
        self.move_percentage = 0.3

        self.clean_grid()

        # Driving variables
        self.L = 1.0
        self.speed = 0.2
        self.steering_angle = 0.0
        self.max_steering_angle = np.pi / 3

    def pure_pursuit_goal_callback(self, msg):
        self.local_goal = [msg.x, msg.y]
        print(f"New goal: {self.local_goal}")

    def clean_grid(self):

        self.tree = MyTree()

        # Initialize the tree with the initial point
        initial_point = MyPoint()
        initial_point.x = 0.0 #self.grid_size * self.grid_resolution / 2
        initial_point.y = 0.0 #self.grid_size * self.grid_resolution / 2
        initial_point.id = 0
        initial_point.parent = None
        self.tree.vertices[0] = initial_point

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
        
        # publish the local goal
        self.publish_goal(self.local_goal)

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

        # Update the occupancy grid with the rotated grid
        self.grid = rotated_grid

        occupancy_grid_msg.data = np.ravel(rotated_grid).tolist()

        self.publisher_.publish(occupancy_grid_msg)

        # Broadcast the transform with the corrected orientation
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'ego_racecar/base_link'
        transform.child_frame_id = 'ego_racecar/base_link'
        transform.transform.rotation = q
        self.tf_broadcaster.sendTransform(transform)

    def publish_goal(self, goal, color=[0, 0, 1]):
        """
        Publishes the goal as a PointStamped message

        Args:
            goal (tuple): the goal as a tuple (x, y)
        Returns:
        """
        goal_msg = PointStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'ego_racecar/base_link'
        goal_msg.point.x = goal[0]
        goal_msg.point.y = goal[1]
        goal_msg.point.z = 0.0

        self.goal_publisher.publish(goal_msg)

    def publish_marker_one(self, goal_point, frame='map', color=(1.0, 0.0, 0.0), size=1.0, publisher = 0):
        # Publish a marker for the goal point
        marker = Marker()
        marker.header.frame_id = frame
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.ns = 'random_point'
        marker.pose.position.x = goal_point.x
        marker.pose.position.y = goal_point.y
        marker.pose.position.z = goal_point.z
        marker.scale.x = 0.2 * size
        marker.scale.y = 0.2 * size
        marker.scale.z = 0.2 * size
        marker.color.a = 0.8
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        if publisher == 0:
            self.random_point_pub.publish(marker)
        elif publisher == 1:
            self.next_point_pub.publish(marker)

    def publish_marker(self, goal_point, frame='map', color=(1.0, 0.0, 0.0), size=1.0):
        # Create a new marker
        marker = Marker()
        marker.header.frame_id = frame
        marker.id = len(self.marker_history)  # Unique ID based on history size
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.ns = 'random_point'
        marker.pose.position.x = goal_point.x
        marker.pose.position.y = goal_point.y
        marker.pose.position.z = goal_point.z
        marker.scale.x = 0.2 * size
        marker.scale.y = 0.2 * size
        marker.scale.z = 0.2 * size
        marker.color.a = 0.8
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]

        # Add the marker to the history
        self.marker_history.append(marker)

        # Publish the entire history
        marker_array = MarkerArray(markers=self.marker_history)
        self.marker_array_pub.publish(marker_array)

    def publish_line_strip(self, points):

        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'  # Change to your desired frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'line_strip'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1  # Line width
        marker.color.r = 1.0  # Red color
        marker.color.a = 1.0  # Fully opaque

        # Define the line strip points
        for point in points:
            marker.points.append(point)

        self.target_line_pub.publish(marker)

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
        self.pose_msg = pose_msg

        # clean tree
        self.clean_grid()

        # while new_node is not in goal_range
        while True:
            self.publish_goal(self.local_goal)
            
        #   sample_free_space and choose one point
            self.sample_free_space() # this will set self.chosen_point
        #   choose the nearest node 
        #       nearest_node = nearest(tree, sampled_point)
            self.get_nearest_node() # this will set self.nearest_node_id

        #   go move_percentage of the way from nearest_node to chosen_point, add to tree
            new_node = self.update_step() # this adds to the tree
        
        # for my own visualization
            path = self.find_path(new_node)
            self.publish_line_strip(path)

        #  check if new_node is in goal_range
            if self.is_goal(new_node):
                #  if yes, find_path
                time.sleep(3)
                self.marker_history = []
                break
        # visualize the path


        # Steer for path

    def is_straight_line_clear(self, grid, a1, b1, a2, b2):
        # Convert coordinates to integers
        a1 = int(a1)
        b1 = int(b1)
        a2 = int(a2)
        b2 = int(b2)
        # Bresenham's line algorithm
        dx = abs(a2 - a1)
        dy = abs(b2 - b1)
        if dx > dy:
            p_inc = 2 * dy - dx
            y = b1
            for x in range(a1, a2 + 1 if a2 > a1 else a2 - 1, 1 if a2 > a1 else -1):
                if grid[x, y] == 100:
                    return False
                if p_inc >= 0:
                    y += 1 if b2 > b1 else -1
                    p_inc -= 2 * dx
                p_inc += 2 * dy
        else:
            p_inc = 2 * dx - dy
            x = a1
            for y in range(b1, b2 + 1 if b2 > b1 else b2 - 1, 1 if b2 > b1 else -1):
                if grid[x, y] == 100:
                    return False
                if p_inc >= 0:
                    x += 1 if a2 > a1 else -1
                    p_inc -= 2 * dy
                p_inc += 2 * dx
        return True

    def sample_free_space(self):
        """
        This method should randomly sample the free space, and returns a viable point
        """
        random_point = np.random.rand(2) * self.grid_size*self.grid_resolution - self.grid_size*self.grid_resolution/2
        # random_point = np.random.rand(
        #     2) * self.grid_size * self.grid_resolution * 5 - self.grid_size * self.grid_resolution * 5 / 2
        # random_point = np.clip(
        #     random_point, -self.grid_size*self.grid_resolution/2, self.grid_size*self.grid_resolution/2)

        self.chosen_point = Point()
        self.chosen_point.x = random_point[0]
        self.chosen_point.y = random_point[1]
        self.chosen_point.z = 0.0

        self.publish_marker(
            self.chosen_point, frame='ego_racecar/base_link', color=(1.0, 0.0, 0.0), size=0.5)
        
    def get_nearest_node(self):
        """
        This method should return the nearest node on the tree to the sampled point
        """

        self.nearest_node_id = 0
        min_dist = 1000000

        for i, point in self.tree.vertices.items():
            if LA.norm([point.x - self.chosen_point.x, point.y - self.chosen_point.y]) < min_dist:
                self.nearest_node_id = i
                min_dist = LA.norm([point.x - self.chosen_point.x, point.y - self.chosen_point.y])
        
    def update_step(self):
        
        # Get the nearest node from chosen node
        nearest_node = self.tree.vertices[self.nearest_node_id]

        # from nearest_node to chosen_point, draw a line and go move_percentage of the way
        new_node = MyPoint()
        new_node.x = nearest_node.x + self.move_percentage * (self.chosen_point.x - nearest_node.x)
        new_node.y = nearest_node.y + self.move_percentage * (self.chosen_point.y - nearest_node.y)
        new_node.id = len(self.tree.vertices)

        # check if parent should be nearest_node or it's parent 
        if nearest_node.parent and LA.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y]) > LA.norm([nearest_node.x - self.tree.vertices[nearest_node.parent].x, nearest_node.y - self.tree.vertices[nearest_node.parent].y]):

            new_node.parent = nearest_node.parent
        else:
            new_node.parent = nearest_node.id

        self.tree.vertices[new_node.id] = new_node

        # Publish the new node
        # self.publish_marker(
        #     Point(x=new_node.x, y=new_node.y, z=0.0), frame='ego_racecar/base_link', color=(0.0, 1.0, 0.0), size=0.5)

        # self.steer(new_node)

        return new_node

    def steer(self, new_node: MyPoint):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """

        # Calculate curvature/steering angle
        curvature = 2 * new_node.y / self.L ** 2
        curvature = curvature * 0.4
        steering_angle = max(-self.max_steering_angle,
                             min(self.max_steering_angle, curvature))

        # Publish the drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.pose_msg.header.stamp
        drive_msg.header.frame_id = self.pose_msg.header.frame_id
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.speed
        self.drive_pub.publish(drive_msg)

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

    def is_goal(self, current_point):
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
        dist = LA.norm([current_point.x - self.local_goal[0], current_point.y - self.local_goal[1]])
        if dist < 0.2:
            return True
        return False

    def find_path(self, current_point):
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

        while current_point.parent is not None:
            path.append(Point(x=current_point.x, y=current_point.y, z=0.0))
            print(current_point.id, end=", ")
            current_point = self.tree.vertices[current_point.parent]

        print(f" Path is {len(path)} long")

        return path

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
