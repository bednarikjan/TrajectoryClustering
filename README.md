# TrajectoryClustering
This application is capable of fully automatic clustering of 2D trajectory data. Why is this useful? Many systems today rely on capturing the data with one main property - a time-varying location which can be thought of as a trajectory. Whether it is ecologists who track wild animals, meteorologists who track hurricane flows or traffic researchers who analyse the common patterns in traffic, they all work with huge amount of data wchich must be (auomatically) analyzed in order to derive meaningful information.

## How it works?
The application performs either agglomerative or spectral clustering. The former requires the user to specify expected number of cluters wheras the latter is capable of finding the most suitable number of clusters automatically. It is based on the modified Hausdorff distance used as a semi-metric to define an affinity among different trajectories. For more detail see this paper: [Clustering of vehicle trajectories][1]

## Dependencies
* Python 2.7
* NumPy
* SciPy

## Install and run
```
$ chmod +x trajectory_clustering.py
$ python trajectory_clustering.py
```

## Controls
```
mouse <left>  draw a trajectory
key <a>		    agglomerative clustering (number of clusters must be specified)
key <s>		    spectral clustering (number of clusters is found automatically)
key <r>		    reset
```

## Synopsis
```
trajectory_clustering num
		num		expected number of clusters (only required for agglomerative clustering)
```

[1]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5462900
