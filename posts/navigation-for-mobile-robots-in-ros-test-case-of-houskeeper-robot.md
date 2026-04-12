<!--
.. title: Up and Running with Autonomous Mobile Robotics (AMR) in ROS 2 and C++
.. slug: navigation-for-mobile-robots-in-ros-test-case-of-housekeeper-robot
.. date: 2025-12-01 04:41:14 UTC+01:00
.. tags: ROS2, C++, Robotics, Path Planning, tutorial
.. category: 
.. link: 
.. description: 
.. type: text
.. has_math: true
.. status: draft
-->

*Reinforcement Learning (RL) represents a powerful framework for solving sequential decision-making problems in dynamic environments across diverse domains, such as control of robots or optimization of profit. However, its practical implementation requires navigating a variety of software packages, encompassing deep learning libraries (e.g., TensorFlow, PyTorch, JAX/Flax), environment frameworks (e.g., Gymnasium, Numpy), and hyperparameter optimization techniques and libraries. This post critically evaluates the common PyTorch, Gymnasium, and NumPy RL stack by comparing it to a faster alternative: JAX/Flax for both of the model training and simulation of environments. A Gridworld example evaluating both training speed and accuracy is utilized to test these packages. Additionally, we complement this example by a comprehensive tracking and monitoring of the training process using MLflow along with a thorough hyperparameters optimization via Optuna. The post concludes with a discussion of the results and final recommendations for optimal use cases of each of these packages.* 

<!--END_TEASER -->

<!-- Alternative Title: -->

<center>
<br>
<img width="100%"src='/images/ros2starter/example.png'>
<br>
Figure 1: The popular eco-system for modular and scalable training of RL agents. 
</center>

The Gif 

----------

**Table of Content**

----------

[TOC]

----------

<!-- Gif of learning across time for all envs -->

# Introduction:

What is AMR and ROS2



## Application Areas of Mobile Robotics

Logitic 
Farming
Rescue
House chores


## Main Tasks in Mobile Robotic

images from your show.

Localizing

Mapping

Path Planning


## Why C++

> Note about difference from ROS1

## The navigation task in focus


### The simulation idea


### The bicycle model


### Primer on Path Planning (more to come)




# The Practical Part


<div style="border-width: 2px;padding:4px;border-style: solid;border-color:black; background:lightgreen;">
<b>Note</b>
<br>

Sometimes it is hard to note

</div>


## Installing ROS

Here we're adapting ROS2 Jazzy release, which still maintained in the time of writing withing Ubuntu 24 operating system.

```bash
sudo apt install ros-<distro>-desktop
sudo apt install ros-dev-tools
```


test it.


```bash
source /opt/ros/jazzy/setup.bash
ros2 run turtlesim turtlesim_node
```

You can put the source command in `.bashrc`

## Core concepts of ROS

ROS is a middelware for:

- Better communication between robotic functionalites
- Better collaboration (unified popular framework)
- Huge library of popular algorthems


Main elements in ROS:

- Nodes
- Topics
- Services
- Actions


## Creating the package in ROS


The hierarchy of any ROS2 project contain the following levels:

- Workspace level (the full robot platform of hardware)
    - Package level (the full intended application you're doing)
        - Node level (the exact functionalities undertaken to achieve the application)



### Creating the project directories

The following commands show how to initialize the project directories:

```
$ mkdir amr_ws
$ cd src/
$ cd ..
$ colcon build // always at ws level
$ source install/setup.bash
$ source opt/ros/jazzy/setup.bash

```

Creating the package:

```
$ cd src/
$ ros2 pkg create  planners_pkg --build-type ament_cmake --dependencies rclcpp
```

Creating a node:

```
$ cd amr_ws/src/planners_pkg/src
$ touch planners.cpp
```

After that you can edit your node code in planners.cpp. For the change to take effect you need to build the whole package again (with colcon) and source the project once more.

To run the node afterwards use (possibly under alternative name):

```
$ros2 run planners_pkg  planners --ros-args -r __node:=alternative_name

```

To inspect the node under running, like viewing the topics, commands like the following are useful:

```
$ros2 node list
$ros2 node info /topic_name
```

Other command that helps debugging and viewing topics info are:

```
ros2 topic echo /topic_name
ros2 interface show example/interface 
ros2 topic pub -r 2.0 /topic_name interface "{key: value}"
```

The first command shows the content of a topic messages
The second show the structure of topic messages (called interfaces)
The last command shows how to publish new topic from command line just for debugging purposes


## Used interfaces

We will need mainly to send:

- Control commands: for that we will use here Twist messages

- Robot pose: for that we will use vec3 messages

- Images of the environment: for that we will use image messages

```
yasin@yasin-B650M-D3HP:~/Dokumente/SideProjects/housekeeper_ws/src$ ros2 interface show geometry_msgs/msg/Pose2D
# Deprecated as of Foxy and will potentially be removed in any following release.
# Please use the full 3D pose.

# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.

# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.# This expresses a position and orientation on a 2D manifold.

float64 x
float64 y
float64 theta
yasin@yasin-B650M-D3HP:~/Dokumente/SideProjects/housekeeper_ws/src$ ros2 interface show geometry_msgs/msg/Vector3
# This represents a vector in free space.

# This is semantically different than a point.
# A vector is always anchored at the origin.
# When a transform is applied to a vector, only the rotational component is applied.

float64 x
float64 y
float64 z
yasin@yasin-B650M-D3HP:~/Dokumente/SideProjects/housekeeper_ws/src$ ros2 interface show sensor_msgs/msg/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image

std_msgs/Header header # Header timestamp should be acquisition time of image
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
                             # Header frame_id should be optical frame of camera
                             # origin of frame should be optical center of cameara
                             # +x should point to the right in the image
                             # +y should point down in the image
                             # +z should point into to plane of the image
                             # If the frame_id here and the frame_id of the CameraInfo
                             # message associated with the image conflict
                             # the behavior is undefined

uint32 height                # image height, that is, number of rows
uint32 width                 # image width, that is, number of columns

# The legal values for encoding are in file include/sensor_msgs/image_encodings.hpp
# If you want to standardize a new string format, join
# ros-users@lists.ros.org and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.hpp

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)
yasin@yasin-B650M-D3HP:~/Dokumente/SideProjects/housekeeper_ws/src$ 


```

There are many other built-in interfaces. To view them run:

```
$ ros2 interface list
```

This command will show also interfaces for the services and actions as well.

## Sketching the node network

<center>
<br>
<img width="100%"src='/images/ros2starter/rosnodesplan.png'>
<br>
Figure 1: The popular eco-system for modular and scalable training of RL agents. 

</center>



After running all the nodes, we can view the graph showing the nodes and their topics with the command: `$rqt_graph`, which will give figure like the following

<center>
<br>
<img width="100%"src='/images/ros2starter/rosgraph.png'>
<br>
Figure 1: The popular eco-system for modular and scalable training of RL agents. 
</center>




## Bug 2 Algorithm


# Conclusion and Next Steps 



Write your post here.


