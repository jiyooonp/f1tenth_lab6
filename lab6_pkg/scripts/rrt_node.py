#!/usr/bin/env python3

import numpy as np
from numpy import linalg as LA

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, Pose, Point, Quaternion, TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point

from scipy.ndimage import binary_dilation

import time

from my_utils import MyTree, MyPoint, MyViz

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
        
        self.my_viz = MyViz()
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Occupancy grid variables
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 100
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8) 

        self.local_goal = [2., 0.]
        
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
        
        self.my_viz.publish_waypoints(self.waypoints)

        self.local_waypoints = None

        self.done_steering = False

        self.collision_check_step = 50

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
            self.grid, structure=np.ones((3, 3))).astype(np.int8) 
        
        self.grid *= 100

        self.grid_viz = np.rot90(self.grid, k=1)  # Rotate 180 degrees

        self.my_viz.grid = self.grid_viz
        print("grid updated")

        # print the indexes
        for i in range(len(grid_x)):
            print(f"grid_x: {grid_x[i]}, grid_y: {grid_y[i]}, {self.grid[grid_x[i], grid_y[i]]}")

        # Publish occupancy grid
        self.my_viz.publish_grid(self.get_clock().now().to_msg())

    def get_next_point(self, current_position):

        # Find the closest waypoint to the current position
        distances = np.linalg.norm( self.waypoints - current_position, axis=1)
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

        self.my_viz.publish_point_marker(self.goal_point_map,
                                frame='map', color=(1.0, 0.0, 1.0, 1.0), size=0.4)
        
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

    def pose_callback(self, pose_msg):
        # Publish the waypoints
        # self.my_viz.publish_waypoints(self.waypoints)
        
        self.pose_msg = pose_msg

        self.current_position = np.array(
            [pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        
        self.get_next_point(self.current_position)

        if self.local_waypoints is not None:
            if not self.done_steering:
                # print("steering")
                self.steer()
            else:
                self.stop()
                print("have waypoint but done steering, thus will rrt")
                self.rrt()
        else:
            print("don't have waypoints, thus will rrt")
            self.stop()
            self.rrt()
        
    def rrt(self):
        new_node = None
        while new_node is None:
            self.sample_free_space() # this will set self.chosen_point
            self.get_nearest_node() # this will set self.nearest_node_id

        #   go move_percentage of the way from nearest_node to chosen_point, add to tree
            new_node = self.update_step_collision()  # this adds to the tree
    
        path = self.find_path(new_node)

        self.my_viz.publish_line_strip(path, self.get_clock().now().to_msg())

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

            print(f"interp path is {len(interpolated_path)} long")

            # reverse the path
            interpolated_path = interpolated_path[::-1]

            # add the current_position to make it in the map frame 
            for point in interpolated_path:
                point[0] += self.current_position[0]
                point[1] += self.current_position[1]

            self.local_waypoints = np.array(interpolated_path).reshape(-1, 2)
            # self.local_waypoints = np.array(path)

            # self.publish_line_strip(interpolated_path)
            # self.my_viz.publish_waypoint_strip(self.local_waypoints, self.get_clock().now().to_msg())
            self.my_viz.publish_waypoint_sphere(self.local_waypoints, frame='map', color=(0.0, 1.0, 1.0, 1.0), size=0.3)

            # steer to path using pure pursuit

            print("found path")
            time.sleep(1)

            self.my_viz.line_strip_history = MarkerArray()

            self.clean_grid()
            
    def sample_free_space(self):

        # choose a random point that is not in an occupied cell

        while True:
            random_point = np.random.rand(2) * self.grid_size*self.grid_resolution - self.grid_size*self.grid_resolution/2
            self.chosen_point = Point()
            self.chosen_point.x = random_point[0]
            self.chosen_point.y = random_point[1]
            self.chosen_point.z = 0.0
            # if not self.check_collision(Point(x = self.current_position[0], y = self.current_position[1], z = 0.0), self.chosen_point):
            break

        self.my_viz.publish_marker_history(
            self.chosen_point, frame=self.frame_base_link, color=(1.0, 1.0, 0.0, 1.0), size=0.1)

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
        # num_steps = int(abs(x2 - x1)/self.grid_resolution)
        # print(f"num_steps is {num_steps}, 1 is {round(x1, 3), round(y1, 3)}, 2 is {round(x2, 3), round(y2, 3)}")

        x_step = (x2 - x1) / self.collision_check_step
        y_step = (y2 - y1) / self.collision_check_step

        # if num_steps == 0:
        #     return False
        
        for i in range(self.collision_check_step + 1):
            x = x1 + i * x_step
            y = y1 + i * y_step
            
            grid_x = np.clip(np.floor((x / self.grid_resolution) + (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)
            grid_y = np.clip(np.floor((y / self.grid_resolution) + (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)


            # self.my_viz.grid = self.grid
            rotated_x, rotated_y = grid_y,  grid_x
            # print("checking point", grid_x, grid_y,
            #       self.grid[grid_x, grid_y], " rot: ", rotated_x, rotated_y, self.grid[rotated_x, rotated_y])


            # Check if the cell is occupied
            # if self.grid[rotated_x, rotated_y] != 0 :
            #     print(
            #         f"collision at {rotated_x}, {rotated_y}, => {self.grid[rotated_x, rotated_y]}")
            #     return True  # Path is not collision-free
            if self.my_viz.grid[rotated_x, rotated_y] != 0:
                print(
                    f"collision at {rotated_x}, {rotated_y}, => {self.my_viz.grid[rotated_x, rotated_y]} ")
                return True  # Path is not collision-free
            # self.my_viz.grid[rotated_x, rotated_y] = 100
            # self.my_viz.publish_grid(self.get_clock().now().to_msg())
            # self.my_viz.grid[rotated_x, rotated_y] = 0

            
            # self.my_viz.grid[self.grid_size - grid_x, grid_y] = 100

            # wait for user input 
            # input("Press Enter to continue...")

        return False  # Path is collision-free
    
    def update_step_collision(self):

        # Get the nearest node from chosen node
        nearest_node = self.tree.vertices[self.nearest_node_id]

        new_node = MyPoint()
        new_node.x = nearest_node.x + self.move_percentage * (self.chosen_point.x - nearest_node.x)
        new_node.y = nearest_node.y + self.move_percentage * (self.chosen_point.y - nearest_node.y)
        new_node.id = len(self.tree.vertices)

        # Check if the path to the new node is collision-free
        if not self.check_collision(nearest_node, new_node):
            # Path is collision-free, proceed with adding the new node

            # no optimization
            # new_node.parent = nearest_node.id

            # check if parent should be nearest_node or it's parent 
            # if nearest_node.parent and LA.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y]) > LA.norm([nearest_node.x - self.tree.vertices[nearest_node.parent].x, nearest_node.y - self.tree.vertices[nearest_node.parent].y]):

            #     new_node.parent = nearest_node.parent
            # else:
            #     new_node.parent = nearest_node.id

            # check if parent should be nearest_node or one of it's ancestors
            prev = new_node
            dist = 0
            if nearest_node.parent:
                curr_point = self.tree.vertices[nearest_node.parent]

                while curr_point.parent is not None:
                    # print(f"Current point is {curr_point.id}")
                    dist += LA.norm([prev.x - curr_point.x, prev.y - curr_point.y])
                    new_dist = LA.norm([new_node.x - curr_point.x, new_node.y - curr_point.y])
                    if dist >= new_dist:
                        dist = new_dist
                        new_node.parent = curr_point.id
                    prev = curr_point
                    curr_point = self.tree.vertices[curr_point.parent]
            else:
                new_node.parent = nearest_node.id

            self.tree.vertices[new_node.id] = new_node
            return new_node
        else:
            # Path is not collision-free, return None or handle accordingly
            print("path has collision")
            return None

    def stop(self):

        # Publish the drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.pose_msg.header.stamp
        drive_msg.header.frame_id = self.pose_msg.header.frame_id
        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.speed = 0.0
        self.drive_pub.publish(drive_msg)

    def steer(self):

        self.done_steering = False


        self.get_next_steer_point(self.current_position)
        # print("selected index:", self.steer_goal_index, "-> ", self.steer_goal_point_map)
        if LA.norm([self.local_waypoints[-1, 0] - self.current_position[0], self.local_waypoints[-1, 1] - self.current_position[1]]) < self.goal_dist_threshold * 2:
            self.done_steering = True
            print("steering complete!")
            return
        

        # Transform the goal point to the vehicle frame of reference
        transformation = self.transform_goal_point()

        goal_point_base_link = do_transform_point(
            PointStamped(point=self.steer_goal_point_map), transformation)

        goal_point_base_link = goal_point_base_link.point

        self.my_viz.publish_pp_marker(goal_point_base_link,
            frame=self.frame_base_link, color=(0.2, 1.0, 0.0, 1.0), size=0.4)

        # Calculate curvature/steering angle
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
        if dist <= self.goal_dist_threshold:
            return True
        return False

    def find_path(self, current_point):
        path = []

        # basic code to find the path
        while current_point and (current_point.parent is not None):
            path.append(Point(x=current_point.x, y=current_point.y, z=0.0))
            current_point = self.tree.vertices[current_point.parent]

        path.append(Point(x = 0.0, y = 0.0, z = 0.0))

        # find shortest path 
        # parent = current_point.parent
        # dist = 0
        # while current_point and (current_point.parent is not None):
        #     dist += LA.norm([current_point.x - self.tree.vertices[parent].x, current_point.y - self.tree.vertices[parent].y])
        #     current_point = self.tree.vertices[current_point.parent]
        #     parent = current_point.parent


        #     path.append(Point(x=current_point.x, y=current_point.y, z=0.0))
        #     current_point = self.tree.vertices[current_point.parent]

        # path.append(Point(x=0.0, y=0.0, z=0.0))

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
