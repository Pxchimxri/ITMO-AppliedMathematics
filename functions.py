# import sparse
# import stats
# import numpy
import scipy.sparse as sps
import scipy.stats as stats
import numpy as np
from numpy import array, zeros, diag, diagflat, dot


def getMatrix(n):
    matrix = sps.rand(n, n, density=0.1, format='csr', dtype=np.int16)
    matrix.setdiag(1)
    return matrix


def LU(A):
    n = len(A.indptr) - 1

    U = sps.csr_matrix(sps.rand(n, n, density=0.0))
    L = sps.csr_matrix(sps.rand(n, n, density=0.0))

    for i in range(n):
        for j in range(n):
            U[0, i] = A[0, i]
            L[i, 0] = A[i, 0] / U[0, 0]

            s = 0.0

            for k in range(i):
                s += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - s

            if i > j:
                L[j, i] = 0
            else:
                s = 0.0
                for k in range(i):
                    s += L[j, k] * U[k, i]
                L[j, i] = (A[j, i] - s) / U[i, i]

    return [L, U]


def getEMatrix(n):
    eMatrix = sps.csr_matrix(sps.rand(n, n, density=0.0))
    eMatrix.setdiag(1)
    return eMatrix


def getXMatrix(n):
    matrix = sps.csr_matrix(sps.rand(n, n, density=0.0))
    return matrix


def getColOfMatrix(xyMAtrix, i):
    return xyMAtrix.getrow(i).toarray()[0]


def getRandomBVector(n):
    return np.random.rand(n)


def inverseMatrix(L, U):
    n = len(L.indptr) - 1
    eMatrix = getEMatrix(n)

    y = np.array([])
    resultMatrix = [np.array([])]

    iter_ = 0

    for k in range(0, n):
        e = getColOfMatrix(eMatrix, k)
        temp = np.array([])
        for i in range(0, n):
            sum = 0
            for p in range(0, i):
                sum += L[i, p] * temp[p]
            yi = e[i] - sum
            temp = np.append(temp, yi)
            iter_ += 1
        y = np.append(y, temp)
        iter_ += 1

    y = y.reshape(n, n)

    for k in range(0, n):
        yi = y[k]
        x = np.zeros(n)
        for i in range(0, n):
            sum = 0
            for k in range(0, i):
                sum += U[n - i - 1, n - k - 1] * x[n - k - 1]
            x[n - i - 1] = 1/U[n - i - 1, n - i - 1] * (yi[n - i - 1] - sum)
            iter_ += 1
        resultMatrix = np.append(resultMatrix, x)
        iter_ += 1

    resultMatrix = resultMatrix.reshape(n, n)
    resultMatrix = resultMatrix.transpose()

    return (resultMatrix, iter_)


def Jakobi(A, b, eps):
    n = len(A.indptr) - 1
    x = b.copy()

    norm = 10

    X = np.array([])

    ab = A.copy()
    ab = ab.multiply(sps.csr_matrix(b))

    iter_ = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                X = np.append(X, ab[i, j])

    while abs(norm) > eps:
        for i in range(n):
            x[i] = b[i]

            for j in range(n):
                if i != j:
                    x[i] = x[i] - (A[i, j] * X[j])

            x[i] = x[i]/A[i, i]

        norm = X[0] - x[0]

        for i in range(n):
            if (abs(X[i] - x[i]) > norm):
                norm = abs(X[i] - x[i])
            X[i] = x[i]
        iter_ += 1

    return (X, iter_)


def jacobiBoost(A1, b, N=1000, x=None, tol=1e-15):
    A = A1.toarray()

    if x is None:
        x = zeros(len(A[0]))
    D = diag(A)
    R = A - diagflat(D)

    for i in range(N):
        x2 = (b - dot(R, x)) / D
        delta = np.linalg.norm(x - x2)
        if delta < tol:
            return x2
        x = x2
    return x


def getBcof(A):
    n = len(A.indptr) - 1
    y = np.array([])
    for i in range(1, n + 1):
        y = np.append(y, i)

    return np.array(A.dot(y))


def solution(L, U, b):
    n = len(U.indptr) - 1

    y = np.array([])

    iter_ = 0

    for i in range(0, n):
        s = 0
        for k in range(0, i):
            s += y[k] * L[i, k]
        y = np.append(y, b[i] - s)
        iter_ += 1

    x = np.zeros(n)

    for i in range(0, n):
        s = 0

        for k in range(0, i):
            s += U[n - i - 1, n - k - 1] * x[n - k - 1]

        x[n - i - 1] = 1/U[n - i - 1, n - i - 1] * (y[n - i - 1] - s)
        iter_ += 1
    return (x, iter_)
