from graphBuilders import *
from graphics import buildGraph
from functions import *
import numpy as np


def f(a): return (a[0]**2)/15 + a[1]**2


builders = [defGradientBuilder, steepestDespectBuilder,
            conjugateGradientBuilder, conjugateDirectionBuilder, newtonBuilder]

start = np.array([D(-38), D(20)])

print("steepestDespect")
print(steepestDespect(f, start, D(0.001)))
print("defGradient")
print(defGradient(f, start, D(0.001)))
print("conjugateDirection")
print(conjugateDirection(f, start, D(0.001)))
print("conjugateGradient")
print(conjugateGradient(f, start, D(0.001)))
print("newton")
print(newton(f, start, D(0.001)))

while True:
    print("Input:")
    buildGraph(f, builders[int(input()) - 1])
