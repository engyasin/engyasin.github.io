<!--
.. title: Navigation for Mobile Robots in ROS - Test Case of Housekeeper Robot
.. slug: navigation-for-mobile-robots-in-ros-test-case-of-housekeeper-robot
.. date: 2025-12-01 04:41:14 UTC+01:00
.. tags:
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
<img width="100%"src='/images/rlwithbells/rlwithbells.png'>
<br>
Figure 1: The popular eco-system for modular and scalable training of RL agents. 
</center>


----------

**Table of Content**

----------

[TOC]

----------

<!-- Gif of learning across time for all envs -->



<div style="border-width: 2px;padding:4px;border-style: solid;border-color:black; background:lightgreen;">
<b>Note</b>
<br>

Sometimes it is hard to note

</div>

# Introduction (Motivation)

# Installing ROS

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


# The System: Housekeeper

# Control logic with FSM

# Setting up the system with ROS (Dummy controller)

# Training RL models with A*/D*

# Results (with GIFs)


# Open Source code and Contribution

# Results and Conclusion

Write your post here.


