# Lab 1: Automatic Emergency Braking

## Video Link
(FILL ME IN)



IDEA:

1. Make occupancy grid based on LiDAR sensor (lookahead_dist)

    ref: http://wiki.ros.org/local_map

    ref: https://atsushisakai.github.io/PythonRobotics/modules/mapping/lidar_to_grid_map_tutorial/lidar_to_grid_map_tutorial.html

    ref: https://answers.ros.org/question/399186/creating-a-map-from-laser-scan/
    

2. Have a struct that stores
    points (V)
        point parent
        pair<int, int> x, y

    edges (E)
        start_point
        end_point
        
    def nearest()
    def shorten()
