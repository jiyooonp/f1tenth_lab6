#!/usr/bin/env python3

import numpy as np
from numpy import linalg as LA

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point

from scipy.ndimage import binary_dilation

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
        
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tf_buffer1 = Buffer()
        self.tf_listener1 = TransformListener(self.tf_buffer1, self)

        # Occupancy grid variables
        self.grid_resolution = 0.1  # meters per cell
        self.grid_size = 50
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8) 

        self.my_viz = MyViz()
        self.my_viz.grid_resolution = self.grid_resolution
        self.my_viz.grid_size = self.grid_size

        self.local_goal = [1., 0.]
        
        # RRT variables
        self.move_percentage = 0.3
        self.goal_dist_threshold = 0.25
        self.steer_goal_dist_threshold = 0.5

        self.clean_grid()

        # Driving variables
        self.frame_base_link = 'ego_racecar/base_link'
        self.frame_map = 'map'

        self.L = 3.0

        self.steer_L = 0.4
        self.speed = 1.0

        self.max_steering_angle = np.pi / 4

        # load all the waypoints from the csv file
        self.waypoints = np.genfromtxt(
            '/sim_ws/src/f1tenth_lab5/logs/waypoints.csv', delimiter=',')[:, :2]

        # only take every 50th waypoint
        self.waypoints = self.waypoints[::50]
        
        self.my_viz.publish_waypoints(self.waypoints)

        self.local_waypoints = None

        self.done_steering = False

        self.collision_check_step = 50
        self.dilate_size = 3

        self.first_grid = True

        self.steer_goal_index = 0
        self.goal_index = 0

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
        self.first_grid = False

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Convert ranges to x, y coordinates
        self.x_coords = ranges * np.cos(angles)
        self.y_coords = ranges * -np.sin(angles)

        # Convert coordinates to grid indices
        grid_x = np.clip(np.floor((self.x_coords / self.grid_resolution)).astype(int), 0, self.grid_size - 1)
        grid_y = np.clip(np.floor((self.y_coords / self.grid_resolution) + (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)

        # delete the inf ranges
        valid_indices_x = grid_x != (self.grid_size - 1 or 0)
        valid_indices_y = grid_y != (self.grid_size - 1 or 0)

        valid_indices_xy = valid_indices_x & valid_indices_y

        grid_x = grid_x[valid_indices_xy]
        grid_y = grid_y[valid_indices_xy]

        # Update occupancy grid
        self.grid.fill(0)
        self.grid[grid_x, grid_y] = 1  

        # Perform erosion on the occupancy grid
        self.grid = binary_dilation(
            self.grid, structure=np.ones((self.dilate_size, self.dilate_size))).astype(np.int8) 
        
        self.grid *= 100

        self.grid_viz = np.rot90(self.grid, k=1)  # Rotate 180 degrees

        self.my_viz.grid = self.grid_viz

        # Publish occupancy grid
        self.my_viz.publish_grid(self.get_clock().now().to_msg())

    def get_next_point(self, current_position):

        # Find the closest waypoint to the current position
        closest_waypoint_index = self.goal_index -1 

        # Find the next waypoint to track
        while True:
            next_waypoint_index = min(
                closest_waypoint_index + 1, len(self.waypoints)-1)
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
        # print(f"Closest waypoint index is {closest_waypoint_index}, steer goal index is {self.steer_goal_index}")

        closest_waypoint_index = max(closest_waypoint_index, self.steer_goal_index -1) # np.argmin(distances)

        # Find the next waypoint to track
        while True and (self.local_waypoints is not None):
            
            next_waypoint_index = min(
                closest_waypoint_index + 1, len(self.local_waypoints)-1)
            
            distance_to_next_waypoint = np.linalg.norm(
                self.local_waypoints[next_waypoint_index] - current_position)
            if distance_to_next_waypoint >= self.steer_L:
                break
            closest_waypoint_index = next_waypoint_index

        # Set the goal index to the next waypoint to track
        self.steer_goal_index = next_waypoint_index

        self.steer_goal_point_map = Point()
        self.steer_goal_point_map.x = self.local_waypoints[self.steer_goal_index, 0]
        self.steer_goal_point_map.y = self.local_waypoints[self.steer_goal_index, 1]
        self.steer_goal_point_map.z = 0.0


    def pose_callback(self, pose_msg):
        
        self.pose_msg = pose_msg

        self.current_position = np.array(
            [pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        
        self.get_next_point([self.current_position[0], self.current_position[1]])

        if self.local_waypoints is not None:
            if not self.done_steering:
                self.steer()
            else:
                self.stop()
                self.rrt()
        else:
            self.stop()
            self.rrt()
        
    def rrt(self):
        
        if self.first_grid:
            return

        new_nodes = None
        while new_nodes is None:
            self.sample_free_space() # this will set self.chosen_point
            self.get_nearest_node() # this will set self.nearest_node_id

        #   go move_percentage of the way from nearest_node to chosen_point, add to tree
            # new_node = self.update_step_collision()  # this adds to the tree
            new_nodes = self.update_step_collision()  # this adds to the tree

        for i, new_node in enumerate(new_nodes):
            path = self.find_path(new_node)
            self.my_viz.publish_line_strip(path, i)

        new_node = new_nodes[-1]

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

            # reverse the path
            interpolated_path = interpolated_path[::-1]

            # Transform the goal point to the vehicle frame of reference
            transformation = self.transform_goal_point_base_to_map()

            path_in_map = []
            for point in interpolated_path:
                goal_point_base_link = do_transform_point(
                    PointStamped(point=Point(x=point[0], y=point[1], z = 0.0)), transformation)

                goal_point_base_link = goal_point_base_link.point
                path_in_map.append([goal_point_base_link.x, goal_point_base_link.y])
                # path_in_map.append([point[0], point[1]])

            self.local_waypoints = np.array(path_in_map).reshape(-1, 2)
            # self.my_viz.publish_waypoint_sphere(self.local_waypoints, frame='map', color=(0.0, 1.0, 1.0, 1.0), size=0.3)
            self.my_viz.publish_waypoint_strip(self.local_waypoints)

            # steer to path using pure pursuit
            print("found path")
            # time.sleep(1)
            self.done_steering = False
            self.steer_goal_index = 1

            self.my_viz.line_strip_history = MarkerArray()

            self.clean_grid()
            
    def sample_free_space(self):
        # choose a random point that is not in an occupied cell

        random_point = [np.random.rand(1)*self.grid_size*self.grid_resolution, np.random.rand(
            1) * self.grid_size*self.grid_resolution - self.grid_size*self.grid_resolution/2.0]
        self.chosen_point = Point()
        self.chosen_point.x = float(random_point[0])
        self.chosen_point.y = float(random_point[1])
        self.chosen_point.z = 0.0

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

        x1, y1 = nearest_node.x, nearest_node.y
        x2, y2 = new_node.x, new_node.y

        x_step = (x2 - x1) / self.collision_check_step
        y_step = (y2 - y1) / self.collision_check_step

        for i in range(self.collision_check_step + 1):
            x = x1 + i * x_step
            y = y1 + i * y_step
            
            grid_x = np.clip(np.floor((x / self.grid_resolution)).astype(int), 0, self.grid_size - 1)
            grid_y = np.clip(np.floor((y / self.grid_resolution) + (self.grid_size / 2)).astype(int), 0, self.grid_size - 1)

            rotated_x, rotated_y = grid_y,  grid_x

            if self.my_viz.grid[rotated_x, rotated_y] != 0:
                return True  # Path is not collision-free

        return False  # Path is collision-free
    
    def update_step_collision(self):

        # Get the nearest node from chosen node
        nearest_node = self.tree.vertices[self.nearest_node_id]
        new_node = MyPoint()
        new_node.x = nearest_node.x + self.move_percentage * (self.chosen_point.x - nearest_node.x)
        new_node.y = nearest_node.y + self.move_percentage * (self.chosen_point.y - nearest_node.y)
        new_node.id = len(self.tree.vertices)

        new_straight_node = MyPoint()
        new_straight_node.x = nearest_node.x + self.move_percentage * \
            (self.chosen_point.x - nearest_node.x)
        new_straight_node.y = nearest_node.y + self.move_percentage * \
            (self.chosen_point.y - nearest_node.y)
        new_straight_node.id = len(self.tree.vertices)+1
        

        # Check if the path to the new node is collision-free
        if not self.check_collision(nearest_node, new_node):
            # Path is collision-free, proceed with adding the new node

            # no optimization
            new_node.parent = nearest_node.id
            self.tree.vertices[new_node.id] = new_node
        
            # add optimization additionally 
            # check if parent should be nearest_node or one of it's ancestors
            new_straight_node.parent = nearest_node.id
            prev = new_straight_node
            dist = 0
            if nearest_node.parent:
                curr_point = self.tree.vertices[nearest_node.parent]

                while curr_point.parent is not None:
                    dist += LA.norm([prev.x - curr_point.x, prev.y - curr_point.y])
                    new_dist = LA.norm(
                        [new_straight_node.x - curr_point.x, new_straight_node.y - curr_point.y])
                    if dist >= new_dist and not self.check_collision(curr_point, new_straight_node):
                        dist = new_dist
                        new_straight_node.parent = curr_point.id
                    prev = curr_point
                    curr_point = self.tree.vertices[curr_point.parent]

                wanted_xy = self.path_contains_goal(new_straight_node)

                if wanted_xy is not None:
                    wanted_x, wanted_y = wanted_xy
                    new_straight_node.x = wanted_x
                    new_straight_node.y = wanted_y
            else:
                new_straight_node.parent = nearest_node.id
            self.tree.vertices[new_straight_node.id] = new_straight_node

            # check if parent should be nearest_node or it's parent 
            # if nearest_node.parent and LA.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y]) > LA.norm([nearest_node.x - self.tree.vertices[nearest_node.parent].x, nearest_node.y - self.tree.vertices[nearest_node.parent].y]):

            #     new_node.parent = nearest_node.parent
            # else:
            #     new_node.parent = nearest_node.id

            # return new_straight_node
            return_nodes = []
            for i in range(new_node.id, new_straight_node.id+1):
                return_nodes.append(self.tree.vertices[i])
            return return_nodes
        else:
            # Path is not collision-free, return None or handle accordingly
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

        distance_to_local_goal = LA.norm([self.local_waypoints[-1, 0] - self.current_position[0], self.local_waypoints[-1, 1] - self.current_position[1]])

        if distance_to_local_goal < self.steer_goal_dist_threshold:
            self.done_steering = True
            self.local_waypoints = None
            self.grid = np.zeros(
                (self.grid_size, self.grid_size), dtype=np.int8)
            self.stop()
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
    def transform_goal_point_base_to_map(self):
        try:
            t = self.tf_buffer1.lookup_transform(
                self.frame_map,
                self.frame_base_link,
                rclpy.time.Time())
            return t
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.frame_base_link} to {self.frame_map}: {ex}')
            return
    
    def path_contains_goal(self, some_node):
        parent_node = self.tree.vertices[some_node.parent]

        if parent_node is None or self.local_waypoints is None:
            return None

        x1, y1 = parent_node.x, parent_node.y
        x2, y2 = some_node.x, some_node.y

        x_step = (x2 - x1) / self.collision_check_step
        y_step = (y2 - y1) / self.collision_check_step

        for i in range(self.collision_check_step + 1):
            x = x1 + i * x_step
            y = y1 + i * y_step

            dist = LA.norm([x - self.local_waypoints[-1, 0],
                           y - self.local_waypoints[-1, 1]])
            if dist <= self.steer_goal_dist_threshold*2:
                return (x, y)
        
        return None
    
    def is_goal(self, current_point):

        point = Point(x=self.local_goal[0], y=self.local_goal[1], z=0.0)

        # Transform the goal point to the vehicle frame of reference
        transformation = self.transform_goal_point()

        goal_point_base_link = do_transform_point(
            PointStamped(point=point), transformation)

        goal_point_base_link = goal_point_base_link.point

        self.my_viz.what_are_you(current_point, frame=self.frame_base_link, color=(0.0, 0.0, 1.0, 1.0), size=0.3)

        dist = LA.norm(
            [current_point.x - goal_point_base_link.x, current_point.y - goal_point_base_link.y])

        # input("press enter")

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
