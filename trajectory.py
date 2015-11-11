'''
Created on 24. 4. 2015

@author: janbednarik
'''

from math import *
from common import *

class Trajectory:
    """A class implementing one trajectory"""

    globID = 0

    def __init__(self, gti):
        self.id = Trajectory.globID
        Trajectory.globID += 1

        self.points = []
        self.gti = gti
        self.ci = -1
        self.prefixSum = [0.0]

    def addPoint(self, p):
        # compute prefix sum
        if len(self.points) > 0:
            self.prefixSum.append(self.prefixSum[len(self.prefixSum) - 1] +
                                  euclidDist(p, self.points[len(self.points) - 1]))

        # add point
        self.points.append(p)

    def getPoints(self):
        return self.points

    def getPrefixSum(self):
        return self.prefixSum

    def groundTruth(self):
        return self.gti

    def getClusterIdx(self):
        return self.ci

    def setClusterIdx(self, ci):
        self.ci = ci

    def length(self):
        return self.prefixSum[len(self.prefixSum) - 1]

    def draw(self, widget, color):
        xlast, ylast = None, None
        for p in self.points:
            # paint a point
            widget.create_oval(p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2, fill = color)

            # paint a line
            if xlast is not None and ylast is not None:
                widget.create_line(xlast, ylast, p[0], p[1], smooth=True)
            xlast = p[0]
            ylast = p[1]

    @staticmethod
    def decGlobID():
        Trajectory.globID -= 1

    @staticmethod
    def resetGlobID():
        Trajectory.globID = 0

    def __str__(self):
        str  = "=== Trajectory ===\n"
        str += "ground truth: %d\n" % self.gti
        str += "cluster: %d\n" % self.ci
        for p in self.points:
            str += repr(p) + ", "
        str += "\n"
        return str

    def __len__(self):
        return len(self.points)

if __name__ == "__main__":
    # Test prefix sum
    t1 = Trajectory(0)

    t1.addPoint((0, 0))
    t1.addPoint((1, 1))
    t1.addPoint((2, 2))
    t1.addPoint((3, 3))

    ps = [i * sqrt(2) for i in range(4)]
    assert(ps == t1.prefixSum)

    l = t1.length()
    assert(l == 3 * sqrt(2))

    print("TEST END")
