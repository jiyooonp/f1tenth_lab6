#!/usr/bin/env python3

import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, Pose, Point, Quaternion, TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point

from scipy.ndimage import binary_dilation

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

class MyViz(Node):
    def __init__(self):
        super().__init__('viz_node')
        # Visualization
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid,
            '/occupancy_grid',
            10)
        self.marker_array_pub = self.create_publisher(
            MarkerArray,
            '/marker_array',
            10)
        self.waypoint_array_pub = self.create_publisher(
            MarkerArray,
            '/waypoint_array',
            10)
        self.target_point_pub = self.create_publisher(
            Marker,
            '/target_point',
            10)
        self.target_point_pp_pub = self.create_publisher(
            Marker,
            '/target_pp_point',
            10)
        self.target_line_pub = self.create_publisher(
            Marker,
            '/target_line',
            10)
        self.target_line_history_pub = self.create_publisher(
            MarkerArray,
            '/target_line_history',
            10)
        self.target_waypoint_pub = self.create_publisher(
            Marker,
            '/target_waypoint',
            10)
        self.waypoint_pub = self.create_publisher(
            MarkerArray,
            '/waypoints',
            10)
        self.what_are_you_doing_pub = self.create_publisher(
            Marker,
            '/what_are_you_doing',
            10)
        
        self.randomly_sampled_history = []
        self.line_strip_history = MarkerArray()
        self.frame_base_link = 'ego_racecar/base_link'
        self.frame_map = 'map'

        self.angular = Twist().angular
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 100
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
    def publish_waypoints(self, waypoints):

        markers = MarkerArray()

        for i, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.id = i
            marker.pose.position.x = waypoint[0]
            marker.pose.position.y = waypoint[1]
            marker.pose.position.z = 0.0
            markers.markers.append(marker)
        self.waypoint_pub.publish(markers)

    def publish_marker_history(self, goal_point, frame='map', color=(1.0, 0.0, 0.0, 0.3), size=0.1):

        # Create a new marker
        marker = Marker()
        marker.header.frame_id = frame
        marker.id = len(self.randomly_sampled_history)  # Unique ID based on history size
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.ns = 'random_point'
        marker.pose.position.x = goal_point.x
        marker.pose.position.y = goal_point.y
        marker.pose.position.z = goal_point.z
        marker.scale = Vector3(x=size, y=size, z=size)
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        
        # Add the marker to the history
        self.randomly_sampled_history.append(marker)
        # self.randomly_sampled_history = [marker]

        # Publish the entire history
        marker_array = MarkerArray(markers=self.randomly_sampled_history)

        self.marker_array_pub.publish(marker_array)

    def publish_grid(self, time_stamp):

        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = time_stamp
        occupancy_grid_msg.header.frame_id = self.frame_base_link
        occupancy_grid_msg.info.map_load_time = time_stamp
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
        # rotated_grid = np.rot90(self.grid, k=1)  # Rotate 180 degrees

        occupancy_grid_msg.data = np.ravel(self.grid).tolist()

        self.occupancy_pub.publish(occupancy_grid_msg)

    def publish_point_marker(self, goal_point, frame='ego_racecar/base_link', color=(1.0, 0.0, 0.0, 1.0), size=0.3):

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
        marker.scale = Vector3(x=size, y=size, z=size)
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])

        self.target_point_pub.publish(marker)

    def what_are_you(self, goal_point, frame='ego_racecar/base_link', color=(1.0, 0.0, 0.0, 1.0), size=0.3):

        # Publish a marker for the goal point
        marker = Marker()
        marker.header.frame_id = frame
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.ns = 'what_are_you_doing'
        marker.pose.position.x = goal_point.x
        marker.pose.position.y = goal_point.y
        marker.pose.position.z = 0.0
        marker.scale = Vector3(x=size, y=size, z=size)
        marker.color = ColorRGBA(
            r=color[0], g=color[1], b=color[2], a=color[3])

        self.what_are_you_doing_pub.publish(marker)

    def publish_pp_marker(self, goal_point, frame='ego_racecar/base_link', color=(1.0, 0.0, 0.0, 1.0), size=0.3):

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
        marker.scale = Vector3(x=size, y=size, z=size)
        marker.color = ColorRGBA(
            r=color[0], g=color[1], b=color[2], a=color[3])

        self.target_point_pp_pub.publish(marker)

    def publish_line_strip(self, points, time_stamp):

        marker = Marker()
        marker.header.frame_id = self.frame_base_link  # Change to your desired frame
        marker.header.stamp = time_stamp
        marker.ns = 'line_strip'
        marker.id = len(self.line_strip_history.markers)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1  # Line width
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)    

        # Define the line strip points
        for point in points:
            marker.points.append(point)

        self.line_strip_history.markers.append(marker)

        self.target_line_history_pub.publish(self.line_strip_history)

    def publish_waypoint_strip(self, points, time_stamp):

        marker = Marker()
        marker.header.frame_id = self.frame_map
        marker.header.stamp = time_stamp
        marker.ns = 'waypoint'
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  # Line width
        marker.color = ColorRGBA(r=0.2, g=0.2, b=0.5, a=1.0)

        # Define the line strip points
        for point in points:
            marker.points.append(Point(x=point[0], y=point[1], z=0.0))

        self.target_waypoint_pub.publish(marker)

    def publish_waypoint_sphere(self, points, frame='map', color=(1.0, 0.0, 0.0, 0.3), size=0.3):
        waypoints = []
        for goal_point in points:
            # Create a new marker
            marker = Marker()
            marker.header.frame_id = frame
            # Unique ID based on history size
            marker.id = len(waypoints)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.ns = 'random_point'
            marker.pose.position.x = goal_point[0]
            marker.pose.position.y = goal_point[1]
            marker.pose.position.z = 0.0
            marker.scale = Vector3(x=size, y=size, z=size)
            marker.color = ColorRGBA(
                r=color[0], g=color[1], b=color[2], a=color[3])

            # Add the marker to the history
            waypoints.append(marker)

        # Publish the entire history
        marker_array = MarkerArray(markers=waypoints)

        self.waypoint_array_pub.publish(marker_array)
        print(f"published {len(waypoints)} points")

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