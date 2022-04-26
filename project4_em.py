import argparse
from numba import jit
import numpy as np
import os
import string
from typing import Dict, Union

fileNameOfTrainImages: string = 'train-images.idx3-ubyte'
fileNameOfTrainLabels: string = 'train-labels.idx1-ubyte'


def getFilePath():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir')

    args = parse.parse_args()
    dataDirectory: string = args.data_dir

    trainImagePath = os.path.join(dataDirectory, fileNameOfTrainImages)
    trainLabelPath = os.path.join(dataDirectory, fileNameOfTrainLabels)

    return trainImagePath, trainLabelPath


def getImageTrainingSet(trainImagePath):
    _, numOfTrainImages, rows, cols = np.fromfile(file=trainImagePath, dtype=np.dtype('>i4'), count=4)
    # Read the training images.
    trainingImages = np.fromfile(file=trainImagePath, dtype=np.dtype('>B'), offset=16)
    trainingImages = np.reshape(trainingImages, (numOfTrainImages, rows * cols))

    return numOfTrainImages, rows, cols, trainingImages


def getLabelTrainingSet(trainLabelPath):
    _, numOfTrainLabels = np.fromfile(file=trainLabelPath, dtype=np.dtype('>i4'), count=2)
    trainingLabels = np.fromfile(file=trainLabelPath, dtype=np.dtype('>B'), offset=8)

    return numOfTrainLabels, trainingLabels


def getBinImages(trainImages: np.ndarray):
    binImages = trainImages.copy()
    binImages[binImages < 128] = 0
    binImages[binImages >= 128] = 1
    binImages = binImages.astype(int)

    return binImages


@jit
def eStep(binImages, lamb, probability, numOfTrainImages, pixelOfTrainImages):
    # Init new responsibility.
    newResponsibility = np.zeros((numOfTrainImages, 10))

    # Compute the responsibility for each class of each image, based on the
    # lambda and probability.
    for imageNum in range(numOfTrainImages):
        for classNum in range(10):
            # w = λ * p^xi * (1-p)^(1-xi)
            newResponsibility[imageNum, classNum] = lamb[classNum]
            # For each pixel, compute the p^xi * (1-p)^(1-xi).
            for pixelNum in range(pixelOfTrainImages):
                if binImages[imageNum, pixelNum] == 1:
                    newResponsibility[imageNum, classNum] *= probability[classNum, pixelNum]
                else:
                    newResponsibility[imageNum, classNum] *= (1.0 - probability[classNum, pixelNum])

        # Marginalize the responsibility of this image.
        summation = np.sum(newResponsibility[imageNum, :])
        if summation:
            newResponsibility[imageNum, :] /= summation

    return newResponsibility


@jit
def mStep(binImages, responsibility, pixelOfTrainImages):
    # Compute the sum of responsibility of each class.
    sumOfResponsibility = np.zeros(10)
    for classNum in range(10):
        sumOfResponsibility[classNum] += np.sum(responsibility[:, classNum])

    # Init new lambda and probability.
    lamb = np.zeros(10)
    probability = np.zeros((10, pixelOfTrainImages))

    # Compute the probability of each class of each pixel & lambda.
    for classNum in range(10):
        for pixelNum in range(pixelOfTrainImages):
            # probability = Σ(responsibility * x) / Σ(responsibility)
            for imageNum in range(len(binImages)):
                probability[classNum, pixelNum] += responsibility[imageNum, classNum] * binImages[imageNum, pixelNum]

            if probability[classNum, pixelNum] == 0:
                probability[classNum, pixelNum] = 1 / pixelOfTrainImages
            else:
                probability[classNum, pixelNum] /= sumOfResponsibility[classNum]

        # Compute lambda.
        lamb[classNum] = sumOfResponsibility[classNum] / np.sum(sumOfResponsibility)
        if lamb[classNum] == 0:
            lamb[classNum] = 1 / 10

    return lamb, probability


def outputImaginations(probability, count, difference, rows, cols):
    imagination = (probability >= 0.5)

    print('')
    for classNum in range(10):
        print(f'class {classNum}:', )
        for rowNum in range(rows):
            for colNum in range(cols):
                if imagination[classNum, rowNum * cols + colNum]:
                    print(f'\033[93m1\033[00m', end=' ')
                else:
                    print(0, end=' ')
            print('')
        print('')

    print(f'No. of Iteration: {count}, Difference: {difference:.12f}')


@jit
def getCountingMatrix(lamb, probability, binImages, numOfTrainImages, pixelsOfTrainImages, trainingLabels):
    # The objective of counting matrix is to calculate number between the estimation class and real class.
    # The column is the estimation class, and row is the real class.
    countingMatrix = np.zeros((10, 10))

    # To record the probability of all class for each image.
    result = np.zeros(10)

    for imageNum in range(numOfTrainImages):
        for classNum in range(10):
            # p = λ * p^xi * (1-p)^(1-xi)
            result[classNum] = lamb[classNum]
            for pixelNum in range(pixelsOfTrainImages):
                if binImages[imageNum, pixelNum]:
                    result[classNum] *= probability[classNum, pixelNum]
                else:
                    result[classNum] *= (1.0 - probability[classNum, pixelNum])

        estimationClass = np.argmax(result)

        countingMatrix[estimationClass, trainingLabels[imageNum]] += 1

    return countingMatrix


# Return the matrix recording the relation between estimation classes to real classes.
def getMatchingMatrix(countingMatrix):
    matchingMatrix = np.full(10, -1)

    for _ in range(10):
        indexPair = np.unravel_index(np.argmax(countingMatrix), (10, 10))

        # Record the matching relation.
        matchingMatrix[indexPair[0]] = indexPair[1]

        # Init the number related to indexPair.
        for k in range(10):
            countingMatrix[indexPair[0]][k] = -1
            countingMatrix[k][indexPair[1]] = -1

    return matchingMatrix


@jit
def getPredictionMatrix(lamb, probability, binImages, numOfTrainImages, pixelsOfTrainImages, trainingLabels,
                        matchingMatrix):
    # The column is the estimation class (transformed to estimated real class) of each image,
    # and the row is the real label class of each image.
    predictionMatrix = np.zeros((10, 10))

    # To record the probability of all class for each image.
    result = np.zeros(10)

    for imageNum in range(numOfTrainImages):
        for classNum in range(10):
            # p = λ * p^xi * (1-p)^(1-xi)
            result[classNum] = lamb[classNum]
            for pixelNum in range(pixelsOfTrainImages):
                if binImages[imageNum, pixelNum]:
                    result[classNum] *= probability[classNum, pixelNum]
                else:
                    result[classNum] *= (1.0 - probability[classNum, pixelNum])

        estimationClass = np.argmax(result)

        predictionMatrix[matchingMatrix[estimationClass], trainingLabels[imageNum]] += 1

    return predictionMatrix


def outputResultImaginations(probability, matchingMatrix, rows, cols):
    imagination = (probability >= 0.5)

    matchingList = matchingMatrix.tolist()

    print('')
    for classNum in range(10):
        estimationClass = matchingList.index(classNum)

        print(f'class {classNum}:', )
        for rowNum in range(rows):
            for colNum in range(cols):
                if imagination[estimationClass, rowNum * cols + colNum]:
                    print(f'\033[93m1\033[00m', end=' ')
                else:
                    print(0, end=' ')
            print('')
        print('')


def getConfusionMatrices(classNum, predictionMatrix):
    tp, fp, tn, fn = 0, 0, 0, 0

    for prediction in range(10):
        for real in range(10):
            if prediction == classNum and real == classNum:
                tp += predictionMatrix[prediction, real]
            elif prediction == classNum:
                fp += predictionMatrix[prediction, real]
            elif real == classNum:
                fn += predictionMatrix[prediction, real]
            else:
                tn += predictionMatrix[prediction, real]

    return int(tp), int(fp), int(tn), int(fn)


def outputConfusionMatrices(predictionMatrix, count, numOfTrainImages):
    # Setup error counts
    error = numOfTrainImages

    # Print confusion matrix
    for classNum in range(10):
        tp, fp, tn, fn = getConfusionMatrices(classNum, predictionMatrix)
        error -= tp

        print('\n------------------------------------------------------------\n')
        print(f'Confusion Matrix {classNum}:')
        print(f'\t\tPredict number {classNum}\tPredict not number {classNum}')
        print(f'Is number {classNum}\t\t{tp}\t\t\t{fn}')
        print(f"Isn't number {classNum}\t\t{fp}\t\t\t{tn}")
        print(f'\nSensitivity (Successfully predict number {classNum}): {float(tp) / (tp + fn):.5f}')
        print(f'Specificity (Successfully predict not number {classNum}): {float(tn) / (fp + tn):.5f}')

    # Print total message
    print(f'\nTotal iteration to converge: {count}')
    print(f'Total error rate: {float(error) / numOfTrainImages:.16f}')


def emAlgorithm(trainImage: Dict[str, Union[int, np.ndarray]],
                trainLabel: Dict[str, Union[int, np.ndarray]]):
    binImages = getBinImages(trainImage['images'])

    # Init lambda, probability, and responsibility.
    lamb = np.full(10, 0.1)
    probability = np.random.uniform(0.0, 1.0, (10, trainImage['pixels']))
    # Marginalize the probability.
    for classNum in range(10):
        probability[classNum, :] /= np.sum(probability[classNum, :])
    responsibility = np.zeros((trainImage['num'], 10))

    # Start the EM algorithm.
    count = 0
    while True:
        count += 1
        oldProbability = probability

        # E step.
        responsibility = eStep(binImages, lamb, probability, trainImage['num'], trainImage['pixels'])

        # M step.
        lamb, probability = mStep(binImages, responsibility, trainImage['pixels'])

        # Output the imaginations.
        difference = np.linalg.norm(probability - oldProbability)
        outputImaginations(probability, count, difference, trainImage['rows'], trainImage['cols'])

        if np.linalg.norm(probability - oldProbability) < 0.15 or count > 50:
            break

    countingMatrix = getCountingMatrix(lamb, probability, binImages, trainImage['num'], trainImage['pixels'],
                                       trainLabel['labels'])

    matchingMatrix = getMatchingMatrix(countingMatrix)

    outputResultImaginations(probability, matchingMatrix, trainImage['rows'], trainImage['cols'])

    predictionMatrix = getPredictionMatrix(lamb, probability, binImages, trainImage['num'], trainImage['pixels'],
                                           trainLabel['labels'], matchingMatrix)

    outputConfusionMatrices(predictionMatrix, count, trainImage['num'])


def main():
    trainImagePath, trainLabelPath = getFilePath()

    numOfTrainImages, rows, cols, trainingImages = getImageTrainingSet(trainImagePath)
    numOfTrainLabels, trainingLabels = getLabelTrainingSet(trainLabelPath)

    emAlgorithm({'num': numOfTrainImages, 'rows': rows, 'cols': cols, 'pixels': rows * cols, 'images': trainingImages},
                {'num': numOfTrainLabels, 'labels': trainingLabels})


if __name__ == '__main__':
    main()
