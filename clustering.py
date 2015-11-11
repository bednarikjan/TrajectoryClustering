'''
Created on 24. 4. 2015

@author: janbednarik
'''

from trajectory import Trajectory
import numpy as np
import math
from common import *
from scipy.linalg import sqrtm
from scipy.io.matlab.mio5_utils import scipy
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import kmeans2
import random
from scipy.spatial.distance import euclidean
from scipy import spatial

class Clustering:
    """A class implementing trajectory clustering."""
    def __init__(self, alpha=0.88, w=2.0, stdNN=2, stdMin=0.4, stdMax=20.0):
        """ Constructor

        Arguments:
        alpha -- robustness against outliers (see [1])
        w -- neighborhood (see [1])
        stdNN -- number of nearest neighbors to compute standard deviation used in similarity measure (see [1])
        stdMin -- minimum value for clipping (see [1])
        stdMax -- maximum value for clipping (see [1])

        [1] Clustering of Vehicle Trajectories (Stefan Atev)
        """
        self.trajectories = []
        self.distMat = np.zeros((0,0))
        self.stdDevs = np.zeros((0,0))
        self.alpha = alpha
        self.w = w
        self.stdNN = stdNN
        self.stdMin = stdMin
        self.stdMax = stdMax

    def std(self, tidx):
        return self.stdDevs[tidx]

    def modHausDist(self, t1idx, t2idx):
        """Computes modified Hausdorf distance."""
        t1 = self.trajectories[t1idx]
        t2 = self.trajectories[t2idx]

        distances = np.zeros(len(t1))
        t1pointsRelPos = [t1.getPrefixSum()[i] / t1.length() for i in range(len(t1))]
        t2pointsRelPos = [t2.getPrefixSum()[i] / t2.length() for i in range(len(t2))]

        for i in range(len(t1)):
            pt1 = t1.getPoints()[i]

            # Find corresponding point pt2 in t2 for point pt1 = t1[i]
            pt2idx = np.argmin(np.array([abs(t1pointsRelPos[i] - t2pointsRelPos[j]) for j in range(len(t2pointsRelPos))]))
            pt2 = t2.getPoints()[pt2idx]

            # Get set of points sp2 of t2 within neighborhood of point pt2
            ps = t2.getPrefixSum()
            tmp = [abs(ps[j] - ps[pt2idx]) - (self.w / 2.0) for j in range(len(ps))]
            neighborhoodIdxs = [j for j in range(len(tmp)) if tmp[j] <= 0]

            # Find minimum Euclidean distance between point pt1 and set of points sp2
            dist = float("inf")
            for idx in neighborhoodIdxs:
                newdist = euclidDist(pt1, t2.getPoints()[idx])
                if newdist < dist:
                    dist = newdist

            distances[i] = dist

        # Find distance worse then self.alpha percent of the other distance
        distances = np.sort(distances)

#         return distances[int(round((len(distances) - 1) * self.alpha))]
        return distances[min(int(len(distances) * self.alpha), len(distances) - 1)]

    def createDistanceMatrix(self):
        size = len(self.trajectories)
        self.distMat = np.ones((size, size))

        for r in range(size):
            for c in range(size):
                dist = self.modHausDist(r, c)
                self.distMat[r, c] *= dist
#                 self.distMat[c, r] *= dist

    def createStdDevs(self):
        rowSortedDistMat = np.copy(self.distMat)
        rowSortedDistMat.sort(axis = 1)

        self.stdDevs = rowSortedDistMat[:, min(self.stdNN, rowSortedDistMat.shape[1] - 1)]
        for i in range(len(self.stdDevs)):
            self.stdDevs[i] = max(self.stdMin, min(self.stdMax, self.stdDevs[i]))

    def similarity(self, t1idx, t2idx):
        """A function computes the similarity measure of trajectories t1 and t2
        according to paper 'Clustering of Vehicle Trajectories (Stefan Atev)'
        """
        return math.exp( -(self.distMat[t1idx, t2idx] * self.distMat[t2idx, t1idx]) / (2 * self.std(t1idx) * self.std(t2idx)) )

    def similarityDummy(self, t1idx, t2idx):
        """DEBUG VERSION
        A function computes the similarity measure of trajectories t1 and t2 as
        a simple average Euclidian distance of corresponding point pairs"""
        t1 = self.trajectories[t1idx]
        t2 = self.trajectories[t2idx]

        tlen = min([len(t1), len(t2)])

        dist = 0
        for i in range(tlen):
            dist += math.sqrt((t1.getPoints()[i][0] - t2.getPoints()[i][0])**2 + (t1.getPoints()[i][1] - t2.getPoints()[i][1])**2)

        return 1.0 / (dist / float(tlen) + 1e-6)

    def clusterAgglomerartive(self, trajectories, cn):
        """
        input: A list 'trajectories' of trajectories given as lists of
        objects of class Trajectory. The number of desired clusters 'nc'.
        output: /
        The function performs agglomerative clustering of trajectories
        and for each trajectory sets an index t.ci denoting estimated cluster.
        """
        self.trajectories = trajectories

        # Update a distance matrix and std deviations
        self.createDistanceMatrix()

        clusters = [[i] for i in range(len(trajectories))]

        while len(clusters) > cn:
            affMat = np.zeros((len(clusters), len(clusters)))
            for r in range(affMat.shape[0] - 1):
                for c in range(r + 1, affMat.shape[1]):
                    ## count inter-cluster average distance
                    dist = 0

                    for t1idx in clusters[r]:
                        for t2idx in clusters[c]:
                            # distance of trajectory t1 (t1 in tA) and trajectory t2 (t2 in tB)
                            dist += 1 / ((self.distMat[t1idx, t2idx] * self.distMat[t2idx, t1idx]) + 1e-6)

                    dist *= 1.0 / (len(clusters[r]) * len(clusters[c]))
                    affMat[r, c] = dist

            # Find two closest clusters and merge them
            # First trajectory is given by row index, second trajectory is given by column index of affinity matrix
            t1idx = np.argmax(affMat) / affMat.shape[1]
            t2idx = np.argmax(affMat) % affMat.shape[0]

            clusters[t1idx].extend(clusters[t2idx])
            clusters = [clusters[i] for i in range(len(clusters)) if i != t2idx]

        # Assign an estimated cluster index to each trajectory
        for i in range(len(clusters)):
            for j in clusters[i]:
                self.trajectories[j].setClusterIdx(i)


    def clusterSpectral(self, trajectories, clusters=-1):
        """
        input:
        trajectories - a list 'trajectories' of trajectories given as lists of
        points given as  tuples (x, y).
        clusters - A number of clusters. If the value is not specified, the
        algorithm estimates the best number itself
        output: /
        The function performs spectral clustering of trajectories
        and for each trajectory sets an index t.ci denoting estimated cluster.
        the function estimates the number of resulting clusters automatically.
        """
        # Need to be assigned as am object variable - other support functions use it (createStdDevs(), etc.)!
        self.trajectories = trajectories       

        # Update a distance matrix and std deviations
        self.createDistanceMatrix()

        self.createStdDevs()

        # Compute affinity matrix
        K = np.zeros((len(trajectories), len(trajectories)))
        for r in range(len(trajectories)):
            for c in range(len(trajectories)):
                K[r, c] = self.similarity(r, c)

        # Diagonal matrix W for normalization
        W = np.diag(1.0 / np.sqrt(np.sum(K, 1)))

        # Normalized affinity matrix
        L = np.dot(np.dot(W, K), W)

        # Eigendecomposition
        Eval, Evec = np.linalg.eig(L)

        gMin, gMax = 0, 0
        for val in Eval:
            if val > 0.8:
                gMax += 1
                if val > 0.99:
                    gMin += 1

        # Sort eigenvalues and eigenvectors according to descending eigenvalue
        Eval, Evec = zip(*sorted(zip(Eval, Evec.T), reverse=True))
        Evec = np.array(Evec).T

        g = clusters
        if g == -1:
            ## Estimate the number of clusters
            # Distortion scores for different number of clusters g
            rhog = np.zeros(gMax - gMin + 1)

            for g in range(gMin, gMax + 1):
                V = np.copy(Evec[:, 0:g])
                S = np.diag(1.0 / np.sqrt(np.sum(np.multiply(V, V), 1)))
                R = np.dot(S, V)

                # k-means clustering of the row vectors of R
                cb, wcScatt = kmeans(R, g, iter=20, thresh=1e-05) # cb = codebook (centroids = rows of cb)

                # compute distortion score rho_g (withit class scatter /  sum(within class scatter, total scatter))
                totScatt = np.sum([np.linalg.norm(r - c) for r in R for c in cb])
                rhog[g - gMin] = wcScatt / (totScatt - wcScatt)

            # Best number of centroids.
            g = gMin + np.argmin(rhog)

        print("Number of centroids = %d" % g)

        # Prerfofm classification of trajectories using k-means clustering
        V = np.copy(Evec[:, 0:g])
        S = np.diag(1.0 / np.sqrt(np.sum(np.multiply(V, V), 1)))
        R = np.dot(S, V)

        ## Find g initial centroids (rows)
        initCentroids = np.zeros((g, R.shape[1]))
        # Matrix of distance of each observation (rows) to each initial centroid (columns)
        initCentroidsDist = np.zeros((R.shape[0], g))

        initCentroids[0] = R[random.randint(0, R.shape[0] - 1)]
        for i in range(g - 1):
            # get each observation's distance to the new centroid
            initCentroidsDist[:, i] = [spatial.distance.euclidean(obs, initCentroids[i]) for obs in R]

            # get the observation which has the worst minimal distance to some already existing centroid
            newidx = np.argmax(np.min(initCentroidsDist[:,:(i + 1)], 1))
            initCentroids[i + 1] = R[newidx]

        controids, labels = kmeans2(R, initCentroids, iter=10, thresh=1e-05, minit='matrix', missing='warn')

        assert(len(trajectories) == len(labels))

        for trajLab in zip(trajectories, labels):
            trajLab[0].setClusterIdx(trajLab[1])


# Testing the module
if __name__ == "__main__":
    trajs = [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
             [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5), (3.5, 3.5)],
             [(12.0, -5.0), (10.0, -2.5), (8.0, 0.0), (6.0, 2.5)],
             [(14.0, -7.0), (12.0, -4.5), (10.0, -2.0), (8.0, 0.5)]]

    clust = Clustering()
    res = clust.clusterAgglomerartive(trajs, 2)

    print(res)
