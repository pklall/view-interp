#!/usr/bin/python

import numpy as n
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import os

points =

points = n.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:,0], points[:,1], points[:,2], c="b") 

# Plot median point in blue
ax.scatter(n.median(points[:,0]), n.median(points[:,1]), n.median(points[:,2]), c="b") 

plt.show()
