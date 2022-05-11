import argparse
from libsvm.svmutil import *
import numpy as np
import os
from scipy.spatial.distance import cdist
import string
import time

dataDirectory: string = 'data_project5'
filePathOfTrainingImages: string = os.path.join(dataDirectory, 'X_train.csv')
filePathOfTrainingLabels: string = os.path.join(dataDirectory, 'Y_train.csv')
filePathOfTestingImages: string = os.path.join(dataDirectory, 'X_test.csv')
filePathOfTestingLabels: string = os.path.join(dataDirectory, 'Y_test.csv')


def loadData():
    trainingImages = np.loadtxt(filePathOfTrainingImages, delimiter=',')
    trainingLabels = np.loadtxt(filePathOfTrainingLabels, delimiter=',', dtype=int)
    testingImages = np.loadtxt(filePathOfTestingImages, delimiter=',')
    testingLabels = np.loadtxt(filePathOfTestingLabels, delimiter=',', dtype=int)

    return trainingImages, trainingLabels, testingImages, testingLabels


# Compare the performance of different kernel functions (linear, polynomial, RBF).
def compareDiffKernels(trainingImages, trainingLabels, testingImages, testingLabels):
    # kernel names.
    kernels = ['Linear', 'Polynomial', 'Radial basis function']

    # Compute performance of each kernel.
    for index, kernel in enumerate(kernels):
        problem = svm_problem(trainingLabels, trainingImages)
        # Reference: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
        parameter = svm_parameter(f"-t {index} -q")

        print(f'#Kernel: {kernel}')

        startTime = time.time()
        model = svm_train(problem, parameter)
        svm_predict(testingLabels, testingImages, model)
        endTime = time.time()

        print(f'Total time: {endTime - startTime}')
        print('----------------------------------')


# Grid search with cross validation.
def gridSearchWithCV(trainingImages, trainingLabels, parameters, isKernel=False):
    parameter = svm_parameter(parameters + ' -v 3 -q')
    problem = svm_problem(trainingLabels, trainingImages, isKernel=isKernel)

    return svm_train(problem, parameter)


def gridSearch(trainingImages, trainingLabels, testingImages, testingLabels):
    # kernel names.
    kernels = ['Linear', 'Polynomial', 'Radial basis function']

    # Parameters
    costs = [0.1, 1, 10]
    degrees = [0, 1, 2]
    gammas = [1 / 784, 0.1, 1]
    constants = [-1, 0, 1]

    # Record the best parameters and accuracy.
    arrayOfBestAccuracy = []
    arrayOfBestParameters = []

    # Compute the best parameters and accuracy.
    for index, kernel in enumerate(kernels):
        bestParameter = []
        bestAccuracy = 0.0

        if kernel == 'Linear':
            for cost in costs:
                parameters = f'-t {index} -c {cost}'
                accuracy = gridSearchWithCV(trainingImages, trainingLabels, parameters)

                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    bestParameter = parameters

            arrayOfBestAccuracy.append(bestAccuracy)
            arrayOfBestParameters.append(bestParameter)
        elif kernel == 'Polynomial':
            for cost in costs:
                for degree in degrees:
                    for gamma in gammas:
                        for constant in constants:
                            parameters = f'-t {index} -c {cost} -d {degree} -g {gamma} -r {constant}'
                            accuracy = gridSearchWithCV(trainingImages, trainingLabels, parameters)

                            if accuracy > bestAccuracy:
                                bestAccuracy = accuracy
                                bestParameter = parameters

            arrayOfBestAccuracy.append(bestAccuracy)
            arrayOfBestParameters.append(bestParameter)
        elif kernel == 'Radial basis function':
            for cost in costs:
                for gamma in gammas:
                    parameters = f'-t {index} -c {cost} -g {gamma}'
                    accuracy = gridSearchWithCV(trainingImages, trainingLabels, parameters)

                    if accuracy > bestAccuracy:
                        bestAccuracy = accuracy
                        bestParameter = parameters

            arrayOfBestAccuracy.append(bestAccuracy)
            arrayOfBestParameters.append(bestParameter)

    # Print the result.
    print('----------------------------------')
    problem = svm_problem(trainingLabels, trainingImages)

    for index, kernel in enumerate(kernels):
        print(f'#Kernel: {kernel}')
        print(f'\tMax accuracy: {arrayOfBestAccuracy[index]}%')
        print(f'\tBest parameters: {arrayOfBestParameters[index]}')

        model = svm_train(problem, svm_parameter(arrayOfBestParameters[index] + ' -q'))
        svm_predict(testingLabels, testingImages, model)
        print()


def computeLinearKernel(x, y):
    return x.dot(y.T)


def computeRBFKernel(x, y, gamma):
    return np.exp(-gamma * cdist(x, y, 'sqeuclidean'))


def computeCombinationKernel(kernel1, kernel2, numOfData):
    return np.hstack((np.arange(1, numOfData + 1).reshape(-1, 1), kernel1 + kernel2))


# Use linear kernel and RBF kernel together.
def linearRBFCombination(trainingImages, trainingLabels, testingImages, testingLabels):
    # Parameters
    costs = [0.01, 0.1, 1.0, 10.0, 100.0]
    gammas = [1.0 / 784, 0.001, 0.01, 0.1, 1.0, 10.0]
    numOfTrainingData, _ = trainingImages.shape

    # Compute the best parameters.
    kernelOfLinear = computeLinearKernel(trainingImages, trainingImages)

    # Record the best parameters and accuracy.
    bestAccuracy = 0.0
    bestParameters = ''
    bestGamma = 1.0

    for cost in costs:
        for gamma in gammas:
            kernelOfRBF = computeRBFKernel(trainingImages, trainingImages, gamma)
            combination = computeCombinationKernel(kernelOfLinear, kernelOfRBF, numOfTrainingData)
            parameters = f'-t 4 -c {cost}'
            accuracy = gridSearchWithCV(combination, trainingLabels, parameters, True)

            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestParameters = parameters
                bestGamma = gamma

    # Print best parameters and best accuracy
    print('-------------------------------------------------------------------')
    print('#Kernel: Linear + RBF')
    print(f'\tMax accuracy: {bestAccuracy}%')
    print(f'\tBest parameters: {bestParameters} -g {bestGamma}\n')

    # Train the model based on the best parameters.
    kernelOfRBF = computeRBFKernel(trainingImages, trainingImages, bestGamma)
    combination = computeCombinationKernel(kernelOfLinear, kernelOfRBF, bestGamma)
    problem = svm_problem(trainingLabels, combination, isKernel=True)
    parameters = svm_parameter(bestParameters + ' -q')
    model = svm_train(problem, parameters)

    # Compute prediction based on best parameters.
    numOfTestingData, _ = testingImages.shape
    kernelOfLinear = computeLinearKernel(testingImages, testingImages)
    kernelOfRBF = computeRBFKernel(testingImages, testingImages, bestGamma)
    combination = computeCombinationKernel(kernelOfLinear, kernelOfRBF, bestGamma)
    svm_predict(testingLabels, combination, model)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mode')
    args = parse.parse_args()
    mode = int(args.mode)

    # Load data from files.
    trainingImages, trainingLabels, testingImages, testingLabels = loadData()

    if mode == 1:
        compareDiffKernels(trainingImages, trainingLabels, testingImages, testingLabels)
    elif mode == 2:
        gridSearch(trainingImages, trainingLabels, testingImages, testingLabels)
    elif mode == 3:
        linearRBFCombination(trainingImages, trainingLabels, testingImages, testingLabels)


if __name__ == '__main__':
    main()
