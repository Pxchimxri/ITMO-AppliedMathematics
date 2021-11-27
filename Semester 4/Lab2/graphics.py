from helpers import D
import pylab
import numpy as np


def T(f, n, k):
    a = np.array([5])
    for i in range(n):
        a = np.append(a, 5)
    return newton(f, a, 0.01)


def getData(f):
    x = np.arange(-40, 40, 0.5)
    y = np.arange(-40, 40, 0.5)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = f([xgrid, ygrid])
    return xgrid, ygrid, zgrid


def init(f):
    x, y, z = getData(f)
    pylab.ion()
    fig, ax = pylab.subplots()
    ax.contourf(x, y, z, levels=10)
    ax.axis([-40, 40, -40, 40])
    ax.set_ylabel('y', fontsize=15)
    ax.set_xlabel('x', fontsize=15)
    return (ax, fig)


def buildGraph(f, builder):
    ax, fig = init(f)
    start = np.array([D(-38), D(20)])
    arr = builder(f, start, D(0.001))
    ax.plot(arr[0], arr[1], c='orange')
    pylab.ioff()
    pylab.show()
