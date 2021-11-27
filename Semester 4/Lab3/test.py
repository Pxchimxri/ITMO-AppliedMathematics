import numpy as np
import functions as mathStuff
import matrixGenerator as generate
import time

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def task6Graphics(maxSize):
    def getIterationsGraph(matrixSize, JakobiMethodIterations, LUMethodIterations):
        plt.plot(matrixSize, JakobiMethodIterations, label="Jakobi Iterations")
        plt.plot(matrixSize, LUMethodIterations, label="LU Iterations")
        plt.xlabel('Matrix size')
        plt.ylabel('Iterations')
        plt.title('Matrix size - Iterations')
        plt.legend()
        plt.savefig('Graphs/f_Iterations.jpg')

    def getTimeGraph(matrixSize, JakobiMethodTime, LUMethodTime):
        plt.plot(matrixSize, JakobiMethodTime, label="Jakobi time ")
        plt.plot(matrixSize, LUMethodTime, label="LU time")
        plt.xlabel('Matrix size')
        plt.ylabel('Time, sec')
        plt.title('Matrix Size - Time')
        plt.legend()
        plt.savefig('Graphs/_Time.jpg')

    def getLUTimeGraph(matrixSize, LUBuildTime):
        plt.plot(matrixSize, LUBuildTime, label="LU time")
        plt.xlabel('Matrix size')
        plt.ylabel('Time, sec')
        plt.title('LU decomposition')
        plt.legend()
        plt.savefig('Graphs/_TimeDecomposition.jpg')
        matrixSize = []

    matrixSize = []

    for i in range(1, maxSize):
        matrixSize.append(i)

    eps = 0.1

    LUBuildTime = np.array([])
    LUMethodTime = np.array([])
    JakobiMethodTime = np.array([])

    LUMethodIterations = np.array([])
    JakobiMethodIterations = np.array([])

    for n in matrixSize:

        matrix = mathStuff.getMatrix(n)

        buildLUTime = time.time()
        luMatrix = mathStuff.LU(matrix)
        buildLUTime = time.time() - buildLUTime

        LUBuildTime = np.append(LUBuildTime, buildLUTime)

        lMatrix = luMatrix[0]
        uMatrix = luMatrix[1]

        bVector = mathStuff.getRandomBVector(n)

        luResolveTime = time.time()
        LUResolve = mathStuff.solution(lMatrix, uMatrix, bVector)
        luResolveTime = time.time() - luResolveTime

        LUMethodTime = np.append(LUMethodTime, luResolveTime)
        LUMethodIterations = np.append(LUMethodIterations, LUResolve[1])

        jTime = time.time()
        JakobiResolve = mathStuff.Jakobi(matrix, bVector, eps)
        jTime = time.time() - jTime

        JakobiMethodTime = np.append(JakobiMethodTime, jTime)
        JakobiMethodIterations = np.append(
            JakobiMethodIterations, JakobiResolve[1])

        print(f"end {n}")

    getIterationsGraph(matrixSize, JakobiMethodIterations, LUMethodIterations)
    plt.clf()
    getTimeGraph(matrixSize, JakobiMethodTime, LUMethodTime)
    plt.clf()
    getLUTimeGraph(matrixSize, LUBuildTime)


def task4Test(n):
    open('Tests/task4.txt', 'w').close()
    file = open('Tests/task4.txt', 'a')

    file.write("diagonal dominance \n\n")
    for k in range(1, n):

        mat = generate.getAKMatrix(k, k)
        b = mathStuff.getBcof(mat)

        file.write("---------------[A]---------------\n")
        file.write(f"{mat.toarray()} \n")

        file.write("---------------[B]---------------\n")
        file.write(f"{b} \n")

        file.write("---------------[SOL Jakobi]---------------\n")
        file.write(f"{mathStuff.jacobiBoost(mat, b)} \n\n")

    file.close()


def task5Test():
    open('Tests/task5.txt', 'w').close()
    file = open('Tests/task5.txt', 'a')

    file.write("Gilbert \n \n")
    for size in range(1, 4):
        mat = generate.getGilbert(size)
        b = mathStuff.getBcof(mat)

        file.write("---------------[A]---------------\n")
        file.write(f"{mat.toarray()} \n")

        file.write("---------------[B]---------------\n")
        file.write(f"{b} \n")

        file.write("---------------[SOL Jakobi]---------------\n")
        file.write(f"{mathStuff.jacobiBoost(mat, b)} \n\n")

    file.close()


def main():
    n = 10
    task6Graphics(10)
    task5Test()
    task4Test(n)


if __name__ == "__main__":
    main()
