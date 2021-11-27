from helpers import *
from sympy import *
import math
import numpy


def steepestDespectBuilder(f, a, e):
    isWorking = True
    b = a.copy()
    vertex = [[], []]
    while isWorking:
        b = goldenSectionForSteepestDespect(f, a, -getGrad(f, a), e)
        delta = getDelta(b, a)
        updateVertex(vertex, b, a)
        if sqrt(delta) < e and math.fabs(f(b) - f(a)) < e:
            isWorking = False
        a = b.copy()
    return vertex


def defGradientBuilder(f, a, e):
    isWorking = True
    b = a.copy()
    vertex = [[], []]
    while isWorking:
        for i in range(len(b)):
            b[i] = a[i] - FirstOrderEquasion(f, a, i) * D(0.01)
        delta = getDelta(a, b)
        updateVertex(vertex, b, a)
        if sqrt(delta) < e and math.fabs(f(a) - f(b)) < e:
            isWorking = False
        a = b.copy()
    return vertex


def conjugateDirectionBuilder(f, a, e):
    s = numpy.zeros((len(a), len(a)), dtype=numpy.dtype(decimal.Decimal))
    for i in range(len(a)):
        for j in range(len(a)):
            if i == j:
                s[i][j] = 1
            else:
                s[i][j] = 0
    vertex = [[], []]
    while sqGrad(f, a) > D(e)**D(2):
        for i in range(2, len(a)):
            lbm = lbm = findM(f, a, s[i], e)
            a = a + lbm*s[i]
        lbm = findM(f, a, s[0], e)
        args1 = a + lbm*s[0]
        lbm = findM(f, args1, s[1], e)
        args2 = args1 + lbm*s[1]
        lbm = findM(f, args2, s[0], e)
        args3 = args2 + lbm*s[0]
        s[0] = args3 - args1
        updateVertex(vertex, args3, a)
        a = args3
    return vertex


def conjugateGradientBuilder(f, a, e):
    isWorking = True
    counter = 0
    b = a.copy()
    p = -getGrad(f, a)
    grad = p
    vertex = [[], []]

    while isWorking:
        counter += 1
        alpha = goldenSectionForConjugateGradient(f, a, p, e)
        a = a + alpha * p
        updateVertex(vertex, a, b)
        b = a.copy()
        grad1 = -getGrad(f, a)
        if counter % 2 == 0:
            beta = 0
        else:
            beta = inner(grad1, grad1) / inner(grad, grad)
        p = grad1 + beta * p
        grad = grad1.copy()
        if inner(grad, grad) <= e:
            isWorking = False
    return vertex


def newtonBuilder(f, a, e):
    isWorking = True
    b = a.copy()
    vertex = [[], []]
    while isWorking:
        for i in range(len(a)):
            a[i] = b[i] - \
                FirstOrderEquasion(f, b, i)/SecondOrderEquasion(f, b, i)
        if deltaArgs(b, a) < D(e**2):
            isWorking = False
        updateVertex(vertex, a, b)
        b = a.copy()
    return vertex
