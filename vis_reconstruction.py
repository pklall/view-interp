#!/usr/bin/python

import numpy as n
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.realpath("./results/"))
from reconstruction import points, cameras

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

points = n.array(points)
cameras = n.array(cameras)

ax.scatter(points[:,0], points[:,1], points[:,2], c="b") 

ax.scatter(cameras[:,0], cameras[:,1], cameras[:,2], c="g")

plt.show();
