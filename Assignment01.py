# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:12:30 2020

@author: barti
"""

# https://ifcuriousthenlearn.com/blog/2015/06/09/mechanical-vibrations-with-python/

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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
omega_n = np.sqrt(k/m) # natural frequency 
zeta = d / (2*m*omega_n) # damping ration
g = 9.81 # gravity 
dt = 0.1 # time step (delta t)
t_start = 0 # start time in s
t_end = 300 # end time in s

# matrices for state space
# x_d(t) = Ax(t) + Bu(t) 
A = np.matrix([[0, 1], [-k/m, -d/m]])
B = np.matrix([[0], [1/m]])

# we discretize the system wth the Euler method: 
A_d = dt*A + np.eye(2, dtype = float)
B_d = dt*B

# Initial State
# The state of the mass spring damper system is composed of the position and speed of the mass m
# Create an initial state numpy array, at the position 0m and speed 0m/s. Feel free to test any other values later on

x0 = np.zeros((2,1))
x = x0

def input_u(t):
    """
    System input function
    The system input is the force that acts on the mass m
    Implement the function to return the gravitational force of the mass on the earth. To make the input a little bit
    more interesting invert the force every 100s. Feel free to play around with any other input.
    :param t: time in s
    :return: system input as numpy array
    """
    # initialize u 
    u = np.zeros((2,1))
    
    # if t/100s is even u = g*m
    if np.mod(t,100):
        u = g * m
    else: # u = -g*m
        u = -g * m
     
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
    # calculation of next_x with euler method
    next_x = A_d * x_last + B_d * u
    return next_x


# Create time vector from 0s to 300s in 0.01s steps, to simulate the system in the exercise sheet is delta = 0.1s mentioned
# to get a 0.01 step size we need to generate 30.000 samples -> 300/30.000 = 0.01 
t_vec = np.arange(t_start, t_end, dt) #np.linspace(0,300,num = 30000)

states = []
inputs = []

p = np.zeros((len(t_vec),1))
p_dot = np.zeros((len(t_vec),1))
# Simulate system and create output vector x with all simulated states corresponding to the time vector t_vec

# Need to use an ode function
# calculation of x
for i in range(len(t_vec)):
    u = input_u(t_vec[i])
    x = system_function(x, u, dt)
    
    p[i] = x[0]
    p_dot[i] = x[1]
    states.append(x)
    inputs.append(u)

# Plot result
# Use two subplots to plot the position and the speed of the mass
plt.subplot(2,2,1), plt.plot(t_vec,p,'b',label = r'$x (mm)$', linewidth=2.0)
plt.title('p')
plt.subplot(2,2,2),plt.plot(t_vec,p_dot,'g--',label = r'$\dot{x} (m/sec)$', linewidth=2.0)
plt.title('p_dot')
