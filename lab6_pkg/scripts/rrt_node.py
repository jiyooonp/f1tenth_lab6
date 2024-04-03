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


class RRT(Node):

    def __init__(self):
        super().__init__('rrt_node')

        # Subscribers
        self.pose_sub_ = self.create_subscription(
            Odometry,
            "ego_racecar/odom",
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            1)
        
        # Publishers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)
    
        # Visualization
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid, 
            '/occupancy_grid', 
            10)
        self.marker_array_pub = self.create_publisher(
            MarkerArray,
            '/marker_array',
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
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Occupancy grid variables
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 100
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8) 

        self.local_goal = [2., 0.]
        
        self.angular = Twist().angular

        # RRT variables
        self.move_percentage = 0.5
        self.goal_dist_threshold = 0.3

        self.clean_grid()

        # Driving variables
        self.frame_base_link = 'ego_racecar/base_link'
        self.frame_map = 'map'


        self.L = 2.5
        self.steer_L = 0.3
        self.speed = 0.1
        self.steering_angle = 0.0
        self.max_steering_angle = np.pi / 3

        # load all the waypoints from the csv file
        self.waypoints = np.genfromtxt(
            '/sim_ws/src/f1tenth_lab5/logs/waypoints.csv', delimiter=',')[:, :2]

        # only take every 50th waypoint
        self.waypoints = self.waypoints[::50]
        
        self.publish_waypoints()

        self.local_waypoints = None
        self.local_count = 0

        self.line_strip_history = MarkerArray()
        self.randomly_sampled_history = []
        
    def clean_grid(self):

        self.tree = MyTree()

        # Initialize the tree with the initial point
        initial_point = MyPoint()
        initial_point.x = 0.0 
        initial_point.y = 0.0
        initial_point.z = 0.0
        initial_point.id = 0
        initial_point.parent = None
        self.tree.vertices[0] = initial_point

    def scan_callback(self, msg):

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Convert ranges to x, y coordinates
        self.x_coords = ranges * np.cos(angles)
        self.y_coords = ranges * -np.sin(angles)

        # Convert coordinates to grid indices
        grid_x = np.clip(np.floor((self.x_coords / self.grid_resolution) + (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)
        grid_y = np.clip(np.floor((self.y_coords / self.grid_resolution) + (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)

        # Update occupancy grid
        self.grid.fill(0)
        self.grid[grid_x, grid_y] = 1  

        # Perform erosion on the occupancy grid
        self.grid = binary_dilation(
            self.grid, structure=np.ones((4, 4))).astype(np.int8) 
        
        self.grid *= 100

        # Publish occupancy grid
        self.publish_grid()
        
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

        # Publish the entire history
        marker_array = MarkerArray(markers=self.randomly_sampled_history)

        self.marker_array_pub.publish(marker_array)

    def publish_grid(self):

        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = self.frame_base_link
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

    def publish_line_strip(self, points):

        marker = Marker()
        marker.header.frame_id = self.frame_base_link  # Change to your desired frame
        marker.header.stamp = self.get_clock().now().to_msg()
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

    def publish_waypoint_strip(self, points):

        marker = Marker()
        marker.header.frame_id = self.frame_map
        marker.header.stamp = self.get_clock().now().to_msg()
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

    def get_next_point(self, current_position):

        # Find the closest waypoint to the current position
        distances = np.linalg.norm(
            self.waypoints - current_position, axis=1)
        closest_waypoint_index = np.argmin(distances)

        # Find the next waypoint to track
        while True:
            next_waypoint_index = (
                closest_waypoint_index + 1) % len(self.waypoints)
            distance_to_next_waypoint = np.linalg.norm(
                self.waypoints[next_waypoint_index] - current_position)
            if distance_to_next_waypoint > self.L:
                break
            closest_waypoint_index = next_waypoint_index

        # Set the goal index to the next waypoint to track
        self.goal_index = next_waypoint_index

        self.goal_point_map = Point()
        self.goal_point_map.x = self.waypoints[self.goal_index, 0]
        self.goal_point_map.y = self.waypoints[self.goal_index, 1]
        self.goal_point_map.z = 0.0

        self.local_goal = [self.goal_point_map.x, self.goal_point_map.y]

        self.publish_point_marker(self.goal_point_map,
                                frame='ego_racecar/base_link', color=(1.0, 0.0, 1.0, 1.0), size=0.4)
        
    def get_next_steer_point(self, current_position):

        # Find the closest waypoint to the current position
        distances = np.linalg.norm(
            self.local_waypoints - current_position, axis=1)
        closest_waypoint_index = np.argmin(distances)

        # Find the next waypoint to track
        while True:
            next_waypoint_index = (
                closest_waypoint_index + 1) % len(self.local_waypoints)
            distance_to_next_waypoint = np.linalg.norm(
                self.local_waypoints[next_waypoint_index] - current_position)
            if distance_to_next_waypoint > self.steer_L:
                break
            closest_waypoint_index = next_waypoint_index

        # Set the goal index to the next waypoint to track
        self.steer_goal_index = next_waypoint_index

        self.steer_goal_point_map = Point()
        self.steer_goal_point_map.x = self.local_waypoints[self.steer_goal_index, 0]
        self.steer_goal_point_map.y = self.local_waypoints[self.steer_goal_index, 1]
        self.steer_goal_point_map.z = 0.0

        self.steer_local_goal = [
            self.steer_goal_point_map.x, self.steer_goal_point_map.y]

        # self.publish_point_marker(self.steer_goal_point_map,
        #                           frame='ego_racecar/base_link', color=(0.2, 1.0, 0.0, 1.0), size=0.4)

    def pose_callback(self, pose_msg):
        # Publish the waypoints
        self.publish_waypoints()
        
        self.pose_msg = pose_msg

        current_position = np.array(
            [pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        
        self.current_position = current_position
        
        self.get_next_point(current_position)

        if self.local_waypoints is not None:
            print("steering")
            self.steer()
        
        new_node = None
        while new_node is None:
            self.sample_free_space() # this will set self.chosen_point
            self.get_nearest_node() # this will set self.nearest_node_id

        #   go move_percentage of the way from nearest_node to chosen_point, add to tree
            new_node = self.update_step_collision()  # this adds to the tree
    
        path = self.find_path(new_node)
        self.publish_line_strip(path)

    #  check if new_node is in goal_range
        if self.is_goal(new_node):

            # generate path 
            path = self.find_path(new_node)
            print(f"Path is {len(path)} long")

            # interpolate the path to have more points
            interpolated_path = []
            for i in range(len(path) - 1):
                interpolated_path.append([path[i].x, path[i].y])
                interpolated_path.append([(path[i].x + path[i+1].x)/2, (path[i].y + path[i+1].y)/2])
            interpolated_path.append([path[-1].x, path[-1].y])

            # add the current_position to make it in the map frame 
            for point in interpolated_path:
                point[0] += current_position[0]
                point[1] += current_position[1]

            self.local_waypoints = np.array(interpolated_path).reshape(-1, 2)
            self.local_count = 0

            # self.publish_line_strip(interpolated_path)
            self.publish_waypoint_strip(self.local_waypoints)

            # steer to path using pure pursuit

            print("found path")
            time.sleep(1)
            self.line_strip_history = MarkerArray()

            self.clean_grid()
            
    def sample_free_space(self):
        # choose a random point that is not in an occupied cell
        while True:
            random_point = np.random.rand(2) * self.grid_size*self.grid_resolution - self.grid_size*self.grid_resolution/2
            self.chosen_point = Point()
            self.chosen_point.x = random_point[0]
            self.chosen_point.y = random_point[1]
            self.chosen_point.z = 0.0
            break

        self.publish_marker_history(
            self.chosen_point, frame=self.frame_base_link, color=(0.0, 1.0, 0.0, 0.5), size=0.1)


    def get_nearest_node(self):
        self.nearest_node_id = 0
        min_dist = 1000000

        for i, point in self.tree.vertices.items():
            if LA.norm([point.x - self.chosen_point.x, point.y - self.chosen_point.y]) < min_dist:
                self.nearest_node_id = i
                min_dist = LA.norm([point.x - self.chosen_point.x, point.y - self.chosen_point.y])


    def check_collision(self, nearest_node, new_node):
        # Define the line equation parameters (y = mx + c)
        x1, y1 = nearest_node.x, nearest_node.y
        x2, y2 = new_node.x, new_node.y

        # Calculate slope (m)
        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = np.inf  # Vertical line, handle separately

        # Calculate y-intercept (c)
        c = y1 - m * x1 if m != np.inf else np.nan  # Handle vertical line case

        # Traverse along the line segment and check each cell in the occupancy grid
        num_steps = int(max(abs(x2 - x1), abs(y2 - y1))/self.grid_resolution)

        x_step = (x2 - x1) / num_steps
        y_step = (y2 - y1) / num_steps

        if num_steps == 0:
            return True
        for i in range(num_steps + 1):
            x = x1 + i * x_step
            y = y1 + i * y_step

            grid_x = np.clip(np.floor((x / self.grid_resolution) + (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)
            grid_y = np.clip(np.floor((y / self.grid_resolution) + (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)

            # Check if the cell is occupied
            if self.grid[grid_x, grid_y] != 0:
                return False  # Path is not collision-free

        return True  # Path is collision-free
    
    def update_step_collision(self):
        # Get the nearest node from chosen node
        nearest_node = self.tree.vertices[self.nearest_node_id]

        # Check if the path to the new node is collision-free
        if self.check_collision(nearest_node, self.chosen_point):
            # Path is collision-free, proceed with adding the new node
            new_node = MyPoint()
            new_node.x = nearest_node.x + self.move_percentage * (self.chosen_point.x - nearest_node.x)
            new_node.y = nearest_node.y + self.move_percentage * (self.chosen_point.y - nearest_node.y)
            new_node.id = len(self.tree.vertices)

            # no optimization
            # new_node.parent = nearest_node.id

            # check if parent should be nearest_node or it's parent 
            if nearest_node.parent and LA.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y]) > LA.norm([nearest_node.x - self.tree.vertices[nearest_node.parent].x, nearest_node.y - self.tree.vertices[nearest_node.parent].y]):

                new_node.parent = nearest_node.parent
            else:
                new_node.parent = nearest_node.id

            # check if parent should be nearest_node or one of it's ancestors
            # prev = new_node
            # dist = 0
            # if nearest_node.parent:
            #     curr_point = self.tree.vertices[nearest_node.parent]

            #     while curr_point.parent is not None:
            #         # print(f"Current point is {curr_point.id}")
            #         dist += LA.norm([prev.x - curr_point.x, prev.y - curr_point.y])
            #         new_dist = LA.norm([new_node.x - curr_point.x, new_node.y - curr_point.y])
            #         if dist >= new_dist:
            #             dist = new_dist
            #             new_node.parent = curr_point.id
            #         prev = curr_point
            #         curr_point = self.tree.vertices[curr_point.parent]
            # else:
            #     new_node.parent = nearest_node.id

            self.tree.vertices[new_node.id] = new_node
            return new_node
        else:
            # Path is not collision-free, return None or handle accordingly
            return None

    def steer(self):

        self.get_next_steer_point(self.current_position)

        # Transform the goal point to the vehicle frame of reference
        transformation = self.transform_goal_point()

        goal_point_base_link = do_transform_point(
            PointStamped(point=self.steer_goal_point_map), transformation)

        goal_point_base_link = goal_point_base_link.point

        self.publish_pp_marker(goal_point_base_link,
            frame='map', color=(0.2, 1.0, 0.0, 1.0), size=0.4)

        # Calculate curvature/steering angle
        print(f"Goal point is {goal_point_base_link.x}, {goal_point_base_link.y}")
        curvature = 2 * goal_point_base_link.y / self.steer_L ** 2
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
        print("steering done")

    def transform_goal_point(self):
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
        
    def is_goal(self, current_point):
        dist = LA.norm([current_point.x - self.local_goal[0], current_point.y - self.local_goal[1]])
        if dist < self.goal_dist_threshold:
            return True
        return False

    def find_path(self, current_point):
        path = []

        while current_point and (current_point.parent is not None):
            path.append(Point(x=current_point.x, y=current_point.y, z=0.0))
            current_point = self.tree.vertices[current_point.parent]
        path.append(Point(x = 0.0, y = 0.0, z = 0.0))
        # print(f" Path is {len(path)} long")

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
