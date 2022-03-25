import argparse
import math
import numpy as np
import os
import string

# Types of label (0~9).
numOfLabels = 10
# Bins of pixel values (0~31).
numOfBins = 32

fileNameOfTrainImages: string = 'train-images.idx3-ubyte'
fileNameOfTrainLabels: string = 'train-labels.idx1-ubyte'
fileNameOfTestImages: string = 't10k-images.idx3-ubyte'
fileNameOfTestLabels: string = 't10k-labels.idx1-ubyte'


# Get the number of images, number of rows & columns,
# and image stream of train files.
def getTrainData(dataDirectory: string):
    trainImagePath = os.path.join(dataDirectory, fileNameOfTrainImages)
    trainLabelPath = os.path.join(dataDirectory, fileNameOfTrainLabels)

    trainImageStream = open(trainImagePath, 'rb')
    trainLabelStream = open(trainLabelPath, 'rb')

    # Read train image stream first.
    magicOfTrainImages = int.from_bytes(trainImageStream.read(4), byteorder='big')
    numOfTrainImages = int.from_bytes(trainImageStream.read(4), byteorder='big')
    numOfTrainRows = int.from_bytes(trainImageStream.read(4), byteorder='big')
    numOfTrainCols = int.from_bytes(trainImageStream.read(4), byteorder='big')

    # Read train label stream.
    magicOfTrainLabels = int.from_bytes(trainLabelStream.read(4), byteorder='big')
    numOfTrainLabels = int.from_bytes(trainLabelStream.read(4), byteorder='big')

    assert numOfTrainImages == numOfTrainLabels

    return numOfTrainImages, numOfTrainRows, numOfTrainCols, trainImageStream, trainLabelStream


# Return the prior and likelihhood analyzed result.
def analyzeDiscreteTrainData(numOfTrainImages, imageSize, trainImageStream, trainLabelStream, prior, likeliHood):
    for i in range(numOfTrainImages):
        label = int.from_bytes(trainLabelStream.read(1), byteorder='big')

        prior[label] += 1
        for j in range(imageSize):
            pixelValue = int.from_bytes(trainImageStream.read(1), byteorder='big')
            # The reason of //8 is that, the original pixel value is 0~255,
            # and we need to tally the pixel value to bin num (0~31).
            likeliHood[label][j][pixelValue // 8] += 1


def getDiscreteLikeliHoodSum(imageSize, likeliHood, likeliHoodSum):
    for i in range(numOfLabels):
        for j in range(imageSize):
            for k in range(numOfBins):
                likeliHoodSum[i][j] += likeliHood[i][j][k]


# Get the number of images, number of rows & columns,
# and image stream of train files.
def getTestData(dataDirectory: string):
    testImagePath = os.path.join(dataDirectory, fileNameOfTestImages)
    testLabelPath = os.path.join(dataDirectory, fileNameOfTestLabels)

    testImageStream = open(testImagePath, 'rb')
    testLabelStream = open(testLabelPath, 'rb')

    # Read test image stream first.
    magicOfTestImages = int.from_bytes(testImageStream.read(4), byteorder='big')
    numOfTestImages = int.from_bytes(testImageStream.read(4), byteorder='big')
    numOfTestRows = int.from_bytes(testImageStream.read(4), byteorder='big')
    numOfTestCols = int.from_bytes(testImageStream.read(4), byteorder='big')

    # Read test label stream.
    magicOfTestLabels = int.from_bytes(testLabelStream.read(4), byteorder='big')
    numOfTestLabels = int.from_bytes(testLabelStream.read(4), byteorder='big')

    assert numOfTestImages == numOfTestLabels

    return numOfTestImages, numOfTestRows, numOfTestCols, testImageStream, testLabelStream


def marginalize(probability):
    sumOfProbability = 0

    for element in probability:
        sumOfProbability += element
    for i in range(numOfLabels):
        probability[i] = probability[i] / sumOfProbability


def getDiscreteProbability(numOfTrainImages, imageSize, testImageStream, prior, likeliHood, likeliHoodSum, probability):
    testImage = np.zeros(imageSize, dtype=int)
    for i in range(imageSize):
        testImage[i] = int.from_bytes(testImageStream.read(1), byteorder='big') // 8

    for i in range(numOfLabels):
        probability[i] += np.log(float(prior[i] / numOfTrainImages))

        for j in range(imageSize):
            binValue = testImage[j]
            temp = likeliHood[i][j][binValue]
            if temp != 0:
                probability[i] += np.log(float(temp / likeliHoodSum[i][j]))
            else:
                probability[i] += np.log(float(1e-6) / likeliHoodSum[i][j])

    marginalize(probability)


def getPrediction(probability):
    return np.where(probability == np.amin(probability))[0][0]


def showResult(prediction, probability, answer):
    print('Posterior (in log scale):')
    for i in range(numOfLabels):
        print(i, ': ', probability[i])
    print('Prediction: ', prediction, ', Ans:', answer, '\n')


def showDiscreteImagination(numOfTrainRows, numOfTrainCols, likeliHood):
    print('Imagination of numbers in Bayesian classifier:')
    for i in range(numOfLabels):
        print(i, ':')
        for j in range(numOfTrainRows):
            for k in range(numOfTrainCols):
                temp = 0
                for l in range(numOfBins // 2):
                    temp -= likeliHood[i][numOfTrainCols * j + k][l]
                for l in range(numOfBins // 2, numOfBins):
                    temp += likeliHood[i][numOfTrainCols * j + k][l]
                if temp > 0:
                    print(1, end=' ')
                else:
                    print(0, end=' ')
            print('')
        print('')


def showErrorRate(errorRate):
    print('Error rate: ', errorRate)


def discreteClassifier(dataDirectory: string):
    # Read train data from files.
    numOfTrainImages, numOfTrainRows, numOfTrainCols, trainImageStream, trainLabelStream = getTrainData(dataDirectory)

    # The pixel number of each image.
    imageSize = numOfTrainRows * numOfTrainCols
    prior = np.zeros(numOfLabels, dtype=int)
    likeliHood = np.zeros((numOfLabels, imageSize, numOfBins), dtype=int)
    analyzeDiscreteTrainData(numOfTrainImages, imageSize, trainImageStream, trainLabelStream, prior, likeliHood)
    likeliHoodSum = np.zeros((numOfLabels, imageSize), dtype=int)
    getDiscreteLikeliHoodSum(imageSize, likeliHood, likeliHoodSum)

    # Read test data from files.
    numOfTestImages, numOfTestRows, numOfTestCols, testImageStream, testLabelStream = getTestData(dataDirectory)

    error = 0
    for i in range(numOfTestImages):
        answer = int.from_bytes(testLabelStream.read(1), byteorder='big')
        probability = np.zeros(numOfLabels, dtype=float)
        getDiscreteProbability(numOfTrainImages, imageSize, testImageStream, prior, likeliHood, likeliHoodSum,
                               probability)
        prediction = getPrediction(probability)
        showResult(prediction, probability, answer)

        if prediction != answer:
            error += 1

    errorRate = error / numOfTestImages
    showDiscreteImagination(numOfTrainRows, numOfTrainCols, likeliHood)
    showErrorRate(errorRate)


# Compute the mean (actually the sum currently) and square.
def analyzeContinuousTrainData(numOfTrainImages, imageSize, trainImageStream, trainLabelStream, prior, mean, square):
    for i in range(numOfTrainImages):
        label = int.from_bytes(trainLabelStream.read(1), byteorder='big')

        prior[label] += 1
        for j in range(imageSize):
            pixelValue = int.from_bytes(trainImageStream.read(1), byteorder='big')

            mean[label][j] += pixelValue
            square[label][j] += pixelValue ** 2


# Compute the variance based on square and mean.
# Formula: E(x-m)^2 = E(x^2) - E(x)^2
def getContinuousVariance(imageSize, prior, mean, square, variance):
    for i in range(numOfLabels):
        for j in range(imageSize):
            # Computer the actual mean of each pixel.
            mean[i][j] = mean[i][j] / prior[i]
            # Compute the avg of square of each pixel.
            square[i][j] = square[i][j] / prior[i]
            # Compute the variance of each pixel.
            tempVariance = square[i][j] - (mean[i][j] ** 2)
            if tempVariance != 0:
                variance[i][j] = tempVariance
            else:
                variance[i][j] = 1e-4


def getLikeliHoodByGaussian(pixelValue, mean, variance):
    return np.log(1 / (np.sqrt(2 * math.pi * variance))) - ((pixelValue - mean) ** 2 / (2 * variance))


def getContinuousProbability(numOfTrainImages, imageSize, testImageStream, prior, mean, variance, probability):
    testImage = np.zeros(imageSize, dtype=int)
    for i in range(imageSize):
        testImage[i] = int.from_bytes(testImageStream.read(1), byteorder='big')

    for i in range(numOfLabels):
        probability[i] += np.log(float(prior[i] / numOfTrainImages))

        for j in range(imageSize):
            pixelValue = testImage[j]
            meanValue = mean[i][j]
            varianceValue = variance[i][j]

            probability[i] += getLikeliHoodByGaussian(pixelValue, meanValue, varianceValue)

    marginalize(probability)


def showContinuousImagination(numOfTrainRows, numOfTrainCols, mean):
    print('Imagination of numbers in Bayesian classifier:')
    for i in range(numOfLabels):
        print(i, ':')
        for j in range(numOfTrainRows):
            for k in range(numOfTrainCols):
                if mean[i][numOfTrainCols * j + k] > 128:
                    print(1, end=' ')
                else:
                    print(0, end=' ')
            print('')
        print('')


def continuousClassifier(dataDirectory: string):
    # Read train data from files.
    numOfTrainImages, numOfTrainRows, numOfTrainCols, trainImageStream, trainLabelStream = getTrainData(dataDirectory)

    # The pixel number of each image.
    imageSize = numOfTrainRows * numOfTrainCols
    prior = np.zeros(numOfLabels, dtype=int)
    mean = np.zeros((numOfLabels, imageSize), dtype=float)
    square = np.zeros((numOfLabels, imageSize), dtype=float)
    variance = np.zeros((numOfLabels, imageSize), dtype=float)
    analyzeContinuousTrainData(numOfTrainImages, imageSize, trainImageStream, trainLabelStream, prior, mean, square)
    getContinuousVariance(imageSize, prior, mean, square, variance)

    # Read test data from files.
    numOfTestImages, numOfTestRows, numOfTestCols, testImageStream, testLabelStream = getTestData(dataDirectory)

    error = 0
    for i in range(numOfTestImages):
        answer = int.from_bytes(testLabelStream.read(1), byteorder='big')
        probability = np.zeros(numOfLabels, dtype=float)
        getContinuousProbability(numOfTrainImages, imageSize, testImageStream, prior, mean, variance, probability)
        prediction = getPrediction(probability)
        showResult(prediction, probability, answer)

        if prediction != answer:
            error += 1

    errorRate = error / numOfTestImages
    showContinuousImagination(numOfTrainRows, numOfTrainCols, mean)
    showErrorRate(errorRate)


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode')
    parse.add_argument('--data_dir')

    args = parse.parse_args()

    mode: int = int(args.mode)
    dataDirectory: string = args.data_dir

    # Discrete mode.
    if mode == 0:
        pass
        discreteClassifier(dataDirectory)
    # Continuous mode.
    elif mode == 1:
        continuousClassifier(dataDirectory)


if __name__ == '__main__':
    main()
