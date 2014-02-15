#!/usr/bin/python

import numpy as n
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import os

# Vector drawing code
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    """ Usage:
     > a = Arrow3D([0,1],[0,1],[0,1], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
     > ax.add_artist(a)
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

sys.path.insert(0, os.path.realpath("./results/"))

import reconstruction

points = n.array(reconstruction.points)
camerasT = n.array(reconstruction.cameras_trans)
camerasR = n.array(reconstruction.cameras_rot)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# ax.scatter(points[:,0], points[:,1], points[:,2], c="b") 

# Plot median point in blue
ax.scatter(n.median(points[:,0]), n.median(points[:,1]), n.median(points[:,2]), c="b") 

ax.scatter(camerasT[:,0], camerasT[:,1], camerasT[:,2], c="g")

for (x, y, z), i in zip(camerasT, range(len(camerasT))):
    ax.text(x, y, z, str(i))

    xd = x - 1.5 * camerasR[i, 0]
    yd = y - 1.5 * camerasR[i, 3]
    zd = z - 1.5 * camerasR[i, 6]
    a = Arrow3D([x, xd], [y, yd], [z, zd], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
    ax.add_artist(a)

plt.show();
