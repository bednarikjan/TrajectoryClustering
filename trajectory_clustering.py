'''
Created on 4. 5. 2015

@author: janbednarik
'''

from Tkinter import *
from trajectory import Trajectory
from clustering import Clustering
import sys

canvas_width = 600
canvas_height = 600

MAX_CLUSTERS = 3
MAX_CLUSTERS_USER_DEFINED = False
COLORS = ["#FF0000", # red 
          "#00FF00", # lime
          "#0000FF", # blue
          "#FFFFFF", # white
          "#FFFF00", # yellow
          "#00FFFF", # aqua
          "#FF00FF", # fuchsia
          "#800000", # maroon
          "#808000", # olive
          "#008000", # green
          "#008080", # teal
          "#000080", # navy
          "#800080", # purple
          "#808080", # gray
          "#C0C0C0"] # silver 
 
COLOR_BLACK = "#000000"

# line coordinates
xold, yold = None, None

ci = 0      # cluster index (for painting)
newT = True # Flag - a new trajectory being created

# list of the clusters of trajectories
trajectories = []
clust = Clustering()

# While mouse button 1 is pressed, the trajectory is being painted and new points are saved
def buttonMotion(event):
    global newT, xold, yold

    ## save point
    # a first point for a new trajectory
    if(newT):
        trajectories.append(Trajectory(ci))
        newT = False

    trajectories[len(trajectories) - 1].addPoint((event.x, event.y))

    ## paint one point
#     c = COLORS[ci]
    c = COLOR_BLACK
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)
    w.create_oval(x1, y1, x2, y2, fill = c)

    ## paint a line
    if xold is not None and yold is not None:
        w.create_line(xold, yold, event.x, event.y, smooth=True)

    xold = event.x
    yold = event.y

# Switch to next cluster
def nextCluster(event):
    global ci, ti, COLORS

    # switch to next cluster
    ci = (ci + 1) % MAX_CLUSTERS
    ti = 0
    COLOR_IDX = (ci + 1) % MAX_CLUSTERS

# Switch to next trajectory
def buttonUp(event):
    global newT, xold, yold
    newT = True
    xold = None
    yold = None

    # Check if last trajectory has 0 length
    if trajectories[len(trajectories) - 1].length() == 0.0:
        trajectories.pop()
        Trajectory.decGlobID()

# debug print trajectories
def printTrajectories(event):
    for t in trajectories:
        print(t)

def clusterTrajectoriesAgglomerative(event):
    # perform clustering
    clust.clusterAgglomerartive(trajectories, MAX_CLUSTERS)
     
    # clear canvas
    w.delete('all')

    #redraw background
    w.create_image(300, 300, image=bg)

    # draw colored trajectories
    for t in trajectories:
        t.draw(w, COLORS[t.getClusterIdx()])

def clusterTrajectoriesSpectral(event):
    # perform clustering
    #if MAX_CLUSTERS_USER_DEFINED:
    #    clust.clusterSpectral(trajectories, MAX_CLUSTERS)
    #else:
    clust.clusterSpectral(trajectories)
     
    # clear canvas
    w.delete('all')

    #redraw background
    w.create_image(300, 300, image=bg)

    # draw colored trajectories
    for t in trajectories:
        t.draw(w, COLORS[t.getClusterIdx()])

def reset(event):
    trajectories[:] = []
    Trajectory.resetGlobID()
    
    # clear canvas
    w.delete('all')

    #redraw background
    w.create_image(300, 300, image=bg)

# Command line parsing
if(len(sys.argv) == 2):
    MAX_CLUSTERS = int(sys.argv[1])
    MAX_CLUSTERS_USER_DEFINED = True
    

#print("Number of clusters: %d" % MAX_CLUSTERS)

master = Tk()
master.title( "Trajectory clustering" )

w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack(expand = YES, fill = BOTH)
bg = PhotoImage(file='roundabout.gif')
w.create_image(300, 300, image=bg)
w.focus_set()

w.bind('<B1-Motion>', buttonMotion)
w.bind('<ButtonRelease-1>', buttonUp)
w.bind('a', clusterTrajectoriesAgglomerative)
w.bind('s', clusterTrajectoriesSpectral)
w.bind('r', reset)


mainloop()
