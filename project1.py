import argparse
import numpy as np
import math
import matplotlib.pyplot as pyplot
import string

iteration: int = 1000


def addRowToArrayA(A: np.array, x: float, n: int):
    newRow: np.array = ([])

    for i in range(n):
        newElement: float = math.pow(x, n - i - 1)
        newRow = np.append(newRow, newElement)
    A = np.append(A, [newRow], axis=0)

    return A


def addElementToArray(array: np.array, x: float):
    return np.append(array, x)


# The return value xInArray is the array of all points x.
def getDataInArrayFormat(filename: string, n: int):
    A = np.empty((0, n), float)
    xInArray = np.array([])
    b = np.array([])

    with open(filename) as file:
        for line in file:
            splitLine = line.split(",")

            x = float(splitLine[0])
            y = float(splitLine[1])

            A = addRowToArrayA(A, x, n)
            xInArray = addElementToArray(xInArray, x)
            b = addElementToArray(b, y)

    return A, xInArray, b


def transpose(array: np.array):
    numOfRow = array.shape[0]
    numOfColumn = array.shape[1]
    transposedArray = np.zeros((numOfColumn, numOfRow))

    for rowNum in range(numOfRow):
        for colNum in range(numOfColumn):
            transposedArray[colNum][rowNum] = array[rowNum][colNum]

    return transposedArray


def mul(array1: np.array, array2: np.array):
    numOfRow = array1.shape[0]
    numOfColumn = array2.shape[1]
    numOfMulElement = array1.shape[1]
    mulResult = np.zeros((numOfRow, numOfColumn))

    for rowNum in range(numOfRow):
        for colNum in range(numOfColumn):
            element: float = 0

            for elementNum in range(numOfMulElement):
                array1Element = array1[rowNum][elementNum]
                array2Element = array2[elementNum][colNum]

                element = element + array1Element * array2Element
            mulResult[rowNum][colNum] = element

    return mulResult


def addDiagonalNum(array: np.array, number: float):
    dupArray = np.copy(array)

    numOfRow = dupArray.shape[0]
    numOfColumn = dupArray.shape[1]

    assert numOfRow == numOfColumn

    for rowNum in range(numOfRow):
        originalVal = dupArray[rowNum][rowNum]

        dupArray[rowNum][rowNum] = originalVal + number

    return dupArray


def LUDecomposition(array: np.array):
    numOfRow = array.shape[0]
    numOfColumn = array.shape[1]

    L = np.zeros((numOfRow, numOfColumn))
    L = addDiagonalNum(L, 1)
    U = np.copy(array)

    for rowNum in range(numOfRow):
        for appendedRowNum in range(rowNum + 1, numOfRow):
            if rowNum >= appendedRowNum:
                break

            multiple = U[appendedRowNum][rowNum] / U[rowNum][rowNum]

            # Update L matrix.
            L[appendedRowNum][rowNum] = multiple

            # Update U matrix.
            negativeMultiple = multiple * -1
            for colNum in range(numOfColumn):
                originalVal = U[appendedRowNum][colNum]
                appendVal = negativeMultiple * U[rowNum][colNum]

                U[appendedRowNum][colNum] = originalVal + appendVal
    return L, U


# Calculate x (Ax = y)
def calculateX(A: np.array, y: np.array):
    dupA = np.copy(A)

    numOfRow = dupA.shape[0]
    numOfColumn = dupA.shape[1]

    assert y.shape[0] == numOfRow

    for rowNum in range(numOfRow):
        actualRowNum = numOfRow - rowNum - 1

        # Make the leading element = 1
        multiple = 1 / dupA[actualRowNum][actualRowNum]
        # Update dupA
        for colNum in range(numOfColumn):
            originalVal = dupA[actualRowNum][colNum]

            dupA[actualRowNum][colNum] = multiple * originalVal
        # Update y
        y[actualRowNum] = multiple * y[actualRowNum]

        # Start to append one row to another row
        for appendedRowNum in range(rowNum + 1, numOfRow):
            if rowNum >= appendedRowNum:
                break

            actualAppendedRowNum = numOfRow - appendedRowNum - 1
            multiple = dupA[actualAppendedRowNum][actualRowNum] / dupA[actualRowNum][actualRowNum]
            negativeMultiple = multiple * -1

            # Update dupA
            originalVal = dupA[actualAppendedRowNum][actualRowNum]
            appendVal = negativeMultiple * dupA[actualRowNum][actualRowNum]

            dupA[actualAppendedRowNum][actualRowNum] = originalVal + appendVal
            # Update y
            originalVal = y[actualAppendedRowNum]
            appendVal = negativeMultiple * y[actualRowNum]

            y[actualAppendedRowNum] = originalVal + appendVal

    return y


def getInverseByLU(L: np.array, U: np.array):
    numOfRow = L.shape[0]
    numOfColumn = L.shape[1]

    # Init y as identity matrix.
    y = np.zeros((numOfRow, numOfColumn))
    y = addDiagonalNum(y, 1)

    # Calculate y (Ly=b).
    for rowNum in range(numOfRow):
        for appendedRowNum in range(rowNum + 1, numOfRow):
            if rowNum >= appendedRowNum:
                break

            multiple = L[appendedRowNum][rowNum] / L[rowNum][rowNum]
            negativeMultiple = multiple * -1
            # Update L matrix
            for colNum in range(numOfColumn):
                originalVal = L[appendedRowNum][colNum]
                appendVal = negativeMultiple * L[rowNum][colNum]

                L[appendedRowNum][colNum] = originalVal + appendVal
            # Update y matrix.
            for colNum in range(numOfColumn):
                originalVal = y[appendedRowNum][colNum]
                appendVal = negativeMultiple * y[rowNum][colNum]

                y[appendedRowNum][colNum] = originalVal + appendVal

    y = transpose(y)

    # Calculate x (Ux=y)
    for colNum in range(numOfColumn):
        y[colNum] = calculateX(U, y[colNum])

    return y


def calculateError(A: np.array, x: np.array, b: np.array):
    numOfRow = A.shape[0]
    numOfColumn = A.shape[1]

    result = mul(A, x)
    error = 0

    for rowNum in range(numOfRow):
        error = error + pow((result[rowNum] - b[rowNum]), 2)

    return error


def printResult(x, error):
    # Print fitting line.
    numOfRow = x.shape[0]

    print("Fitting Line:", end=" ")
    for i in range(numOfRow):
        val = x[i][0]
        power = numOfRow - i - 1

        if val != 0:
            if val > 0:
                print("+", val, 'x^', power, end="")
            else:
                print(val, 'x^', power, end="")

    print("")
    # Print error
    print("Total error:", error[0])


def LSE(A: np.array, b: np.array, lamb: float):
    # Calculate AtALambIInverse.
    At = transpose(A)
    AtA = mul(At, A)
    AtALambI = addDiagonalNum(AtA, lamb)
    L, U = LUDecomposition(AtALambI)
    AtALambIInverse = getInverseByLU(L, U)

    # Calculate Atb.
    b = transpose(np.array([b]))
    Atb = mul(At, b)

    # Calculate x (according to lecture).
    x = mul(AtALambIInverse, Atb)
    error = calculateError(A, x, b)

    # Print result.
    print("LSE:")
    printResult(x, error)

    return x


def Newton(A: np.array, b: np.array, n: int):
    # Calculate inverse of Hessian
    Hessian = 2 * mul(transpose(A), A)
    L, U = LUDecomposition(Hessian)
    HessianInverse = getInverseByLU(L, U)

    # Calculate right part of Gradient
    b = transpose(np.array([b]))
    rightPartOfGradient = 2 * mul(transpose(A), b)

    # Init x
    x = np.zeros((n, 1))

    # Compute x
    for _ in range(iteration):
        leftPartOfGradient = 2 * mul(mul(transpose(A), A), x)
        gradient = leftPartOfGradient - rightPartOfGradient
        x = x - mul(HessianInverse, gradient)

    # Calculate x (according to lecture).
    error = calculateError(A, x, b)

    # Print result.
    print("\nNewton's Method:")
    printResult(x, error)

    return x


def visualize(xInArray: np.array, b: np.array, xOfLSE: np.array, xOfNewton: np.array, n: int):
    # Init interval
    minOfX = min(xInArray)
    maxOfX = max(xInArray)
    xAxis = np.arange(minOfX - 3, maxOfX + 3)

    # Init yPoints
    yPointsLSE = 0
    yPointsNewton = 0

    for i in range(n):
        yPointsLSE = yPointsLSE + xOfLSE[i] * (xAxis ** (n - i - 1))
        yPointsNewton = yPointsNewton + xOfNewton[i] * (xAxis ** (n - i - 1))

    pyplot.figure(1)
    # Output LSE Figure
    pyplot.subplot(2, 1, 1)
    pyplot.scatter(xInArray, b, c="red")
    pyplot.plot(xAxis, yPointsLSE, c="black")

    # Output Newton Figure
    pyplot.subplot(2, 1, 2)
    pyplot.scatter(xInArray, b, c="red")
    pyplot.plot(xAxis, yPointsNewton, c="black")

    pyplot.show()


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--n')
    parse.add_argument('--lamb')
    parse.add_argument('--filename')

    args = parse.parse_args()

    n: int = int(args.n)
    lamb: float = float(args.lamb)
    filename: string = args.filename

    A, xInArray, b = getDataInArrayFormat(filename, n)
    xOfLSE: np.array = LSE(A, b, lamb)
    xOfNewton: np.array = Newton(A, b, n)
    xOfLSEFlatten = xOfLSE.flatten()
    xOfNewtonFlatten = xOfNewton.flatten()

    visualize(xInArray, b, xOfLSEFlatten, xOfNewtonFlatten, n)


if __name__ == "__main__":
    main()
