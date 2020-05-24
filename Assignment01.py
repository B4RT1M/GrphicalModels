# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:12:30 2020

@author: barti
"""

import numpy as np
import matplotlib.pyplot as plt

# Mass Spring Damper System
# =========================
#
# --------------------------------
#       |       |
#       /     -----
#       \     |   |
#       / k   |---| d
#       \     | | |
#       |       |
#    ---------------- ---
#    |      m       |  |
#    ----------------  | p
#           |          |
#           | u        v
#           v
#

# Simulation of a first order euler approximation of a mass spring damper system

# System parameters
m = 10  # kg     mass
k = 1   # N/m    characteristic spring constant
d = 1   # N*s/m  characteristic damper constant


# Initial State
# The state of the mass spring damper system is composed of the position and speed of the mass m
# Create an initial state numpy array, at the position 0m and speed 0m/s. Feel free to test any other values later on

x0 = np.zeros(2,1)


def input_u(t):
    """
    System input function
    The system input is the force that acts on the mass m
    Implement the function to return the gravitational force of the mass on the earth. To make the input a little bit
    more interesting invert the force every 100s. Feel free to play around with any other input.
    :param t: time in s
    :return: system input as numpy array
    """
    u = np.zeros((2,1))
    # TODO: calculation of u 
    return u 


def system_function(x_last, u, delta_t):
    """
    System function
    The system function computes the next state of the spring mass damper system given the last state, the current
    input and the time step size.
    :param x_last: System state x at previous time step
    :param u: System Input u at current time step
    :param delta_t: time step size
    :return: return next state
    """
    next_x = np.zeros((2,1))
    # TODO: calculation of next_x
    return next_x


# Create time vector from 0s to 300s in 0.01s steps, to simulate the system
# to get a 0.01 step size we need to generate 30.000 samples -> 300/30.000 = 0.01 
t_vec = np.linspace(0,300,num = 30000)

# Simulate system and create output vector x with all simulated states corresponding to the time vector t_vec
x = None


# Plot result
# Use two subplots to plot the position and the speed of the mass
# TODO
