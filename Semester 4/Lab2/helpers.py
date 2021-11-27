import decimal
from sympy import *
import math
import numpy


def getDelta(a, b):
    delta = 0
    for (lh, rh) in zip(a, b):
        delta += (lh - rh)**2
    return delta


def FirstOrderEquasion(f, a, i):
    d = D(0.00000001)
    c = a.copy()
    c[i] += d
    return (f(c) - f(a))/d


def SecondOrderEquasion(f, a, i):
    d = D(0.000000001)
    c = a.copy()
    c[i] += d
    x1 = D(f(c))
    c[i] -= 2*d
    x2 = D(f(c))
    c[i] += d
    x3 = D(f(c))
    return (x1 - 2*x3 + x2)/(d**2)


def deltaArgs(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])**2
    return sum


def inner(a, b):
    sum = D(0)
    for (x, y) in zip(a, b):
        sum += x * y
    return sum


def sqGrad(f, a):
    sum = D(0)
    for i in range(len(a)):
        sum += FirstOrderEquasion(f, a, i)**2
    return sum


def getGrad(f, a):
    argsPD = a.copy()
    for i in range(len(argsPD)):
        argsPD[i] = FirstOrderEquasion(f, argsPD, i)
    return argsPD


def goldenSectionForSteepestDespect(f, a, b, e):
    x1 = a + (b - a) * decimal.Decimal(0.381966)
    x2 = a + (b - a) * decimal.Decimal(0.618034)
    f1 = f(x1)
    f2 = f(x2)
    delta = getDelta(a, b)
    while sqrt(delta) > e:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) * decimal.Decimal(0.381966)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * decimal.Decimal(0.618034)
            f2 = f(x2)
        delta = getDelta(a, b)
    return (a + b) / D(2)


def goldenSectionForConjugateGradient(f, a, b, eps):
    a1 = D(0)
    b1 = D(1e5)
    x0 = a1 + D(0.5) * (D(3) - D(math.sqrt(5))) * (b1 - a1)
    x1 = b1 - x0 + a1
    while math.fabs(b1 - a1) > D(eps):
        l = a + x0 * b
        r = a + x1 * b
        if f(l) < f(r):
            b1 = x1
        else:
            a1 = x0
        x1 = x0
        x0 = b1 + a1 - x1
    return (a1 + b1)/D(2)


def findM(f, a1, b1, e):
    a = D(-1e3)
    b = D(1e3)
    while math.fabs(b - a) > e:
        y1 = f(a1 + a * b1)
        y2 = f(a1 + b * b1)
        c = (a + b) / 2
        if y1 < y2:
            b = c
        else:
            a = c
    return (a + b) / 2


def updateVertex(vertex, a, b):
    vertex[0].append(b[0])
    vertex[0].append(a[0])
    vertex[1].append(b[1])
    vertex[1].append(a[1])


def D(x):
    return decimal.Decimal(x)
