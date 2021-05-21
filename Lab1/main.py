import math
import numpy

e = 0.0000001
phi = 1 - (math.sqrt(5) - 1)*0.5


def f(x):
    return math.sin(x)*x*x


# def f(x):
#     return math.sin(x)*math.log(x)


def dihotomie(a, b):
    delta = e * 0.1
    iterationCounter = 0
    callCounter = 0
    while (b-a >= e):
        middle = (a+b)*0.5
        iterationCounter += 1
        callCounter += 2
        x1 = middle - delta
        x2 = middle + delta
        if (f(x1) < f(x2)):
            b = x2
        elif (f(x1) > f(x2)):
            a = x1
        elif (f(x1) == f(x2)):
            a = x1
            b = x2
        # print(f"Итерация: {iterationCounter}\t a = {a} b = {b}")
    middle = (a+b)*0.5
    print(
        f"\nДихотомия: f({middle}) = {f(middle)}\n итераций: {iterationCounter}, вызовов f(x): {callCounter}\n")


def goldenRatio(a, b):
    iterationCounter = 0
    callCounter = 2
    x1 = a+phi*(b-a)
    x2 = a+(1-phi)*(b-a)
    fx1 = f(x1)
    fx2 = f(x2)
    while (b-a >= e):
        iterationCounter += 1
        callCounter += 1
        if (fx1 >= fx2):
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a+(1-phi)*(b-a)
            fx2 = f(x2)
        else:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a+phi*(b-a)
            fx1 = f(x1)
        # print(f"Итерация: {iterationCounter}\ta = {a} b = {b}")
        result = 0.5 * (a + b)
    print(
        f"\nЗолотое сечение: f({result}) = {f(result)}\n итераций: {iterationCounter}, вызовов f(x): {callCounter}\n")


def fibonacci(a, b):

    def getFibonacciNumber(number):
        return round(math.pow((1 + math.sqrt(5)) / 2, number) - pow((1 - math.sqrt(5)) / 2, number) / math.sqrt(5))

    n = 1
    while getFibonacciNumber(n + 2) <= (b - a) / e:
        n += 1

    startPoint = a
    endPoint = b
    x1 = startPoint + \
        getFibonacciNumber(n) / getFibonacciNumber(n + 2) * \
        (endPoint - startPoint)
    x2 = startPoint + \
        getFibonacciNumber(n + 1) / getFibonacciNumber(n +
                                                       2) * (endPoint - startPoint)
    y1 = f(x1)
    y2 = f(x2)
    callCounter = 2
    iterationCounter = 0
    while (endPoint - startPoint > e and startPoint < endPoint):
        iterationCounter += 1
        if (y1 > y2):
            n -= 1
            startPoint = x1
            x1 = x2
            x2 = startPoint + \
                getFibonacciNumber(
                    n + 1) / getFibonacciNumber(n + 2) * (endPoint - startPoint)
            y1 = y2
            y2 = f(x2)
            callCounter += 1
        else:
            n -= 1
            endPoint = x2
            x2 = x1
            x1 = startPoint + \
                getFibonacciNumber(n) / getFibonacciNumber(n +
                                                           2) * (endPoint - startPoint)
            y2 = y1
            y1 = f(x1)
            callCounter += 1
        # print(
        #     f"Итерация: {iterationCounter}\ta = {startPoint} b = {endPoint}")
    result = (startPoint + endPoint) / 2

    print(
        f"\nФибоначчи: f({result}) = {f(result)}\n итераций: {iterationCounter}, вызовов f(x): {callCounter}\n")


def parabola(a, b):
    x1 = a
    x2 = -4.0
    x3 = b
    y1 = f(x1)
    y2 = f(x2)
    y3 = f(x3)
    xm = b + e + 1
    callCounter = 3
    iterationCounter = 0
    while (True):
        iterationCounter += 1
        a1 = (y2 - y1) / (x2 - x1)
        a2 = 1 / (x3 - x2) * ((y3 - y1) / (x3 - x1) - a1)
        xm_prev = xm
        xm = 1 / 2 * (x1 + x2 - a1 / a2)
        ym = f(xm)
        callCounter += 1
        if (x1 < xm and xm < x2 and x2 < x3):
            if (ym >= y2):
                x1 = xm
                y1 = ym
            else:
                x3 = x2
                y3 = y2
                x2 = xm
                y2 = ym
        else:
            if (y2 >= ym):
                x1 = x2
                y1 = y2
                x2 = xm
                y2 = ym
            else:
                x3 = xm
                y3 = ym
        # print(
        #     f"Итерация: {iterationCounter}\tx1 = {x1} x2 = {x2} x3 = {x3}")
        if (abs(xm - xm_prev) <= e):
            break
    print(
        f"\nПарабола: f({xm}) = {f(xm)}\n итераций: {iterationCounter}, вызовов f(x): {callCounter}\n")


def brent(a, b):
    startPoint = a
    endPoint = b
    r = (3 - math.sqrt(5)) / 2
    currentD = endPoint - startPoint
    previousD = currentD
    x = startPoint + r * (endPoint - startPoint)
    w = x
    v = x
    yx = f(x)
    yw = yx
    yv = yw
    callCounter = 1
    iterationCounter = 0
    while (True):
        iterationCounter += 1
        if (max(x - startPoint, endPoint - x) < e):
            # print(
            #     f"Итерация: {iterationCounter}\nX = {x}\tW = {w}\tV = {v}\t yx = {yx}\tyw = {yw}\tyv = {yv}")
            print(
                f"\nБрент: f({x}) = {f(x)}\n итераций: {iterationCounter}, вызовов f(x): {callCounter}\n")
            break
        g = previousD / 2
        previousD = currentD
        if (w == x):
            a1 = numpy.NaN
        else:
            a1 = (yw - yx) / (w - x)
        if (v == x or v == w):
            a2 = numpy.NaN
        else:
            a2 = 1 / (v - w) * ((yv - yx) / (v - x) - a1)
        u = 1/2 * (x + w - a1 / a2)
        yu = f(u)
        callCounter += 1

        if (numpy.isnan(u) or (u < startPoint or u > endPoint) or abs(u - x) > g):
            if (x < (startPoint + endPoint) / 2):
                u = x + r * (endPoint - x)
                previousD = endPoint - x
            else:
                u = x - r * (x - startPoint)
                previousD = x - startPoint
            yu = f(u)
            callCounter += 1
        currentD = abs(u - x)
        if (yu > yx):
            if (u < x):
                startPoint = u
            else:
                endPoint = u
            if (yu <= yw or w == x):
                v = w
                yv = yw
                w = u
                yw = yu
            else:
                if (yu < yv or v == x or v == w):
                    v = u
                    yv = yu

        else:
            if (u < x):
                endPoint = x
            else:
                startPoint = x
            v = w
            yv = yw
            w = x
            yw = yx
            x = u
            yx = yu
        # print(
        #     f"Итерация: {iterationCounter}\nX = {x}\tW = {w}\tV = {v}\t yx = {yx}\tyw = {yw}\tyv = {yv}")


def calc(a, b):
    dihotomie(a, b)
    goldenRatio(a, b)
    fibonacci(a, b)
    parabola(a, b)
    brent(a, b)


calc(-3, 0)
# calc(0, 16)
