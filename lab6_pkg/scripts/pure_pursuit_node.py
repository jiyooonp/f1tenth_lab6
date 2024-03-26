#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point


class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ROS publishers
        self.marker_pub = self.create_publisher(
            Marker,
            '/goal_point',
            10)
        
        self.goal_pub = self.create_publisher(
            Point,
            '/pure_pursuit_goal',
            10)

        self.waypoint_pub = self.create_publisher(
            MarkerArray,
            '/waypoints',
            10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.frame_base_link = 'ego_racecar/base_link'
        self.frame_map = 'map'

        self.obom_subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.pose_callback,
            10)

        # load all the waypoints from the csv file
        self.waypoints = np.genfromtxt(
            '/sim_ws/src/f1tenth_lab5/logs/waypoints.csv', delimiter=',')

        # only take every 50th waypoint
        self.waypoints = self.waypoints[::100]

        self.goal_index = 0
        self.L = 2.0
        self.speed = 1.5
        self.steering_angle = 0.0
        self.max_steering_angle = np.pi / 3

    def get_next_point(self, current_position):

        # Find the closest waypoint to the current position
        distances = np.linalg.norm(
            self.waypoints[:, :2] - current_position, axis=1)
        closest_waypoint_index = np.argmin(distances)

        # Find the next waypoint to track
        while True:
            next_waypoint_index = (
                closest_waypoint_index + 1) % len(self.waypoints)
            distance_to_next_waypoint = np.linalg.norm(
                self.waypoints[next_waypoint_index, :2] - current_position)
            if distance_to_next_waypoint > self.L:
                break
            closest_waypoint_index = next_waypoint_index

        # Set the goal index to the next waypoint to track
        self.goal_index = next_waypoint_index

        self.goal_point_map = Point()
        self.goal_point_map.x = self.waypoints[self.goal_index, 0]
        self.goal_point_map.y = self.waypoints[self.goal_index, 1]
        self.goal_point_map.z = 0.0

        # Publish the waypoints
        self.publish_waypoints()

        # self.publish_goal_point(self.goal_point_map,
        #                         frame='map', color=(1.0, 0.0, 1.0), size=0.5)

    def pose_callback(self, pose_msg):

        current_position = np.array(
            [pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])

        self.get_next_point(current_position)

        # Transform the goal point to the vehicle frame of reference
        transformation = self.transform_goal_point(self.goal_point_map)

        goal_point_base_link = do_transform_point(
            PointStamped(point=self.goal_point_map), transformation)

        goal_point_base_link = goal_point_base_link.point

        # Publish the goal point
        self.goal_pub.publish(goal_point_base_link)


    def transform_goal_point(self, current_position):
        try:
            t = self.tf_buffer.lookup_transform(
                self.frame_base_link,
                self.frame_map,
                rclpy.time.Time())
            return t
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.frame_map} to {self.frame_base_link}: {ex}')
            return

    def publish_goal_point(self, goal_point, frame='map', color=(1.0, 0.0, 0.0), size=1.0):
        # Publish a marker for the goal point
        marker = Marker()
        marker.header.frame_id = frame
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal_point.x
        marker.pose.position.y = goal_point.y
        marker.pose.position.z = goal_point.z
        marker.scale.x = 0.2 * size
        marker.scale.y = 0.2 * size
        marker.scale.z = 0.2 * size
        marker.color.a = 0.5
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        self.marker_pub.publish(marker)

    def publish_waypoints(self):
        markers = MarkerArray()

        for i, waypoint in enumerate(self.waypoints):
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


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
