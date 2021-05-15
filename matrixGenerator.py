import scipy.sparse as sps
import numpy as np
from random import choice
def matrixGenerator(n, getItem):
    matrix = sps.csr_matrix(sps.rand(n, n, density=0.0))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 1.0/ (i + j + 1.0)
    return matrix

def getGilbert(n):
    f = lambda i, j: np.float16(1)/ (np.float16(i) +np.float16(j) - np.float16(1.0))

    return matrixGenerator(n, f)
    
def setItemAKMatrix(matrix, i, j, k):
    len_ = len(matrix.indptr) - 1
    sum_ = np.float64(10 ** (-k))
    for i_ in range(len_):
        if i != i_:
            sum_ += matrix[i, i_]
    matrix[i, j] = -sum_

# n -- size, k -- k
def getAKMatrix(n, k):     
    matrix = sps.rand(n, n, density=0.0, format='csr', dtype=np.float64)
    nums = [ -1, -2, -3, -4 ]  
   #nums = [ 0, -1, -2, -3, -4 ]   #раскоментить чтобы включить 0  
    for i in range(n):
        for j in range(n):
            if i == j : continue
            ranadNum = choice(nums)
            #if ranadNum == 0 : continue #раскоментить чтобы включить 0
            matrix[i, j] += ranadNum + 10 ** (-k)
    for i in range(n):
        sum_ = 0
        for j in range(n):
            if i == j: continue
            sum_ += matrix[i, j]
        matrix[i, i] = sum_
    
     #for i in range(n): #раскоментить чтобы были только положительные
     #   for j in range(n):
     #       matrix[i, j] = -matrix[i, j] 
    return matrix