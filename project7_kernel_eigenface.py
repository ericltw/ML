import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy.spatial.distance import cdist
import string

imageDirectory: string = 'data_project7/Yale_Face_Database'


# Set up the input parameters, and return args.
def parseArguments():
    parse = argparse.ArgumentParser()

    # The algorithm will be used, 0 -> PCA, 1-> LDA.
    parse.add_argument('--algo', default=0)
    # Mode of PCA and LDA, 0 -> simple, 1 -> kernel.
    parse.add_argument('--mode', default=0)
    # The number of nearest neighbors used for classification.
    parse.add_argument('--numOfNeighbors', default=5)
    # The kernel type, 0 -> linear, 1 -> RBF.
    parse.add_argument('--kernelType', default=0)
    # The gamma of RBF kernel.
    parse.add_argument('--gamma', default=0.000001)

    return parse.parse_args()


def readTrainingImages():
    trainingImages, trainingLabels = None, None
    numOfImages = 0

    # Get the number of images first.
    with os.scandir(f'{imageDirectory}/Training') as directory:
        # Get number of files
        numOfImages = len([file for file in directory if file.is_file()])

    # Read the files.
    with os.scandir(f'{imageDirectory}/Training') as directory:
        trainingLabels = np.zeros(numOfImages, dtype=int)
        # Images will be resized to 29 * 24.
        trainingImages = np.zeros((numOfImages, 29 * 24))

        for index, file in enumerate(directory):
            if file.path.endswith('.pgm') and file.is_file():
                face = np.asarray(Image.open(file.path).resize((24, 29))).reshape(1, -1)
                trainingImages[index, :] = face
                trainingLabels[index] = int(file.name[7:9])

    return trainingImages, trainingLabels


def readTestingImages():
    testingImages, testingLabels = None, None
    numOfImages = 0

    # Get the number of images first.
    with os.scandir(f'{imageDirectory}/Testing') as directory:
        # Get number of files
        numOfImages = len([file for file in directory if file.is_file()])

    # Read the files.
    with os.scandir(f'{imageDirectory}/Testing') as directory:
        testingLabels = np.zeros(numOfImages, dtype=int)
        # Images will be resized to 29 * 24.
        testingImages = np.zeros((numOfImages, 29 * 24))

        for index, file in enumerate(directory):
            if file.path.endswith('.pgm') and file.is_file():
                face = np.asarray(Image.open(file.path).resize((24, 29))).reshape(1, -1)
                testingImages[index, :] = face
                testingLabels[index] = int(file.name[7:9])

    return testingImages, testingLabels


def simplePCA(numOfTrainingImages, trainingImages):
    # Compute covariance
    trainingImagesTransposed = trainingImages.T
    mean = np.mean(trainingImagesTransposed, axis=1)
    mean = np.tile(mean.T, (numOfTrainingImages, 1)).T
    difference = trainingImagesTransposed - mean
    covariance = difference.dot(difference.T) / numOfTrainingImages

    return covariance


def kernelPCA(trainingImages, kernelType, gamma):
    # Compute kernel.
    # Linear
    if kernelType == 0:
        kernel = trainingImages.T.dot(trainingImages)
    # RBF
    else:
        kernel = np.exp(-gamma * cdist(trainingImages.T, trainingImages.T, 'sqeuclidean'))

    # Get centered kernel.
    matrixN = np.ones((29 * 24, 29 * 24), dtype=float) / (29 * 24)
    matrix = kernel - matrixN.dot(kernel) - kernel.dot(matrixN) + matrixN.dot(kernel).dot(matrixN)

    return matrix


def findTargetEigenvectors(matrix):
    # Compute eigenvalues and eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Get 25 first largest eigenvectors.
    targetIndex = np.argsort(eigenvalues)[::-1][:25]
    targetEigenvectors = eigenvectors[:, targetIndex].real

    return targetEigenvectors


# Transform eigenvectors into eigenfaces/fisherfaces.
# algo parameter means the algorithm been used, 0 -> PCA, 1-> LDA.
def transformEigenvectorsToFaces(targetEigenvectors, algo):
    faces = targetEigenvectors.T.reshape((25, 29, 24))
    fig = plt.figure(1)
    fig.canvas.set_window_title(f'{"Eigenfaces" if algo == 0 else "Fisherfaces"}')

    for idx in range(25):
        plt.subplot(5, 5, idx + 1)
        plt.axis('off')
        plt.imshow(faces[idx, :, :], cmap='gray')


# Reconstruct the faces from eigenfaces and fisherfaces.
def reconstructFaces(numOfTrainingImages, trainingImages, targetEigenvectors):
    reconstructedImages = np.zeros((10, 29 * 24))
    choice = np.random.choice(numOfTrainingImages, 10)

    for index in range(10):
        reconstructedImages[index, :] = trainingImages[choice[index], :].dot(targetEigenvectors).dot(
            targetEigenvectors.T)

    fig = plt.figure(2)
    fig.canvas.set_window_title('Reconstructed faces')

    for index in range(10):
        # Original image.
        plt.subplot(10, 2, index * 2 + 1)
        plt.axis('off')
        plt.imshow(trainingImages[choice[index], :].reshape((29, 24)), cmap='gray')

        # Reconstructed image.
        plt.subplot(10, 2, index * 2 + 2)
        plt.axis('off')
        plt.imshow(reconstructedImages[index, :].reshape((29, 24)), cmap='gray')


# Decorrelate original images into components space.
def decorrelate(numOfImages, images, eigenvectors):
    decorrelatedImages = np.zeros((numOfImages, 25))

    for index, image in enumerate(images):
        decorrelatedImages[index, :] = image.dot(eigenvectors)

    return decorrelatedImages


# Classify and show predict result.
def classifyAndPredict(numOfTrainingImages, numOfTestingImages, trainingImages, trainingLabels, testingImages,
                       testingLabels,
                       targetEigenvectors, numOfNeighbors):
    decorrelatedTraining = decorrelate(numOfTrainingImages, trainingImages, targetEigenvectors)
    decorrelatedTesting = decorrelate(numOfTestingImages, testingImages, targetEigenvectors)
    error = 0
    distance = np.zeros(numOfTrainingImages)

    for testIndex, test in enumerate(decorrelatedTesting):
        for trainIndex, train in enumerate(decorrelatedTraining):
            distance[trainIndex] = np.linalg.norm(test - train)

        minDistances = np.argsort(distance)[:numOfNeighbors]
        predict = np.argmax(np.bincount(trainingLabels[minDistances]))

        if predict != testingLabels[testIndex]:
            error += 1
    print(f'Error count: {error}\nError rate: {float(error) / numOfTestingImages}')


def outputDiagram():
    # Plot
    plt.tight_layout()
    plt.show()


# Principal components analysis.
def PCA(mode, numOfNeighbors, kernelType, gamma, trainingImages, trainingLabels, testingImages, testingLabels):
    # Get the number of training images.
    numOfTrainingImages = len(trainingImages)
    numOfTestingImages = len(testingImages)

    # Simple PCA
    if mode == 0:
        matrix = simplePCA(numOfTrainingImages, trainingImages)
    # Kernel PCA
    else:
        matrix = kernelPCA(trainingImages, kernelType, gamma)

    # Find the first 25 largest eigenvectors.
    targetEigenvectors = findTargetEigenvectors(matrix)

    # Transform eigenvectors into eigenfaces.
    transformEigenvectorsToFaces(targetEigenvectors, 0)

    # Randomly reconstruct 10 eigenfaces.
    reconstructFaces(numOfTrainingImages, trainingImages, targetEigenvectors)

    # Classify and predict.
    classifyAndPredict(numOfTrainingImages, numOfTestingImages, trainingImages, trainingLabels, testingImages,
                       testingLabels, targetEigenvectors, numOfNeighbors)

    # Output the diagram.
    outputDiagram()


def simpleLDA(numOfEachClass, trainingImages, trainingLabels):
    # Compute the overall mean.
    overallMean = np.mean(trainingImages, axis=0)

    # Get mean of each class.
    numOfClasses = len(numOfEachClass)
    classMean = np.zeros((numOfClasses, 29 * 24))

    for label in range(numOfClasses):
        classMean[label, :] = np.mean(trainingImages[trainingLabels == label + 1], axis=0)

    # Compute between-class scatter.
    scatterB = np.zeros((29 * 24, 29 * 24), dtype=float)

    for idx, num in enumerate(numOfEachClass):
        difference = (classMean[idx] - overallMean).reshape((29 * 24, 1))
        scatterB += num * difference.dot(difference.T)

    # Compute within-class scatter.
    scatterW = np.zeros((29 * 24, 29 * 24), dtype=float)
    for idx, mean in enumerate(classMean):
        difference = trainingImages[trainingLabels == idx + 1] - mean
        scatterW += difference.T.dot(difference)

    # Compute Sw^(-1) * Sb.
    matrix = np.linalg.pinv(scatterW).dot(scatterB)

    return matrix


def kernelLDA(numOfEachClass, trainingImages, trainingLabels, kernelType, gamma):
    # Compute kernel.
    numOfClasses = len(numOfEachClass)
    numOfImages = len(trainingImages)

    if not kernelType:
        # Linear
        kernelOfEachClass = np.zeros((numOfClasses, 29 * 24, 29 * 24))
        for idx in range(numOfClasses):
            images = trainingImages[trainingLabels == idx + 1]
            kernelOfEachClass[idx] = images.T.dot(images)
        kernelOfAll = trainingImages.T.dot(trainingImages)
    else:
        # RBF
        kernelOfEachClass = np.zeros((numOfClasses, 29 * 24, 29 * 24))
        for idx in range(numOfClasses):
            images = trainingImages[trainingLabels == idx + 1]
            kernelOfEachClass[idx] = np.exp(-gamma * cdist(images.T, images.T, 'sqeuclidean'))
        kernelOfAll = np.exp(-gamma * cdist(trainingImages.T, trainingImages.T, 'sqeuclidean'))

    # Compute N.
    matrixN = np.zeros((29 * 24, 29 * 24))
    identityMatrix = np.eye(29 * 24)

    for index, num in enumerate(numOfEachClass):
        matrixN += kernelOfEachClass[index].dot(identityMatrix - num * identityMatrix).dot(
            kernelOfEachClass[idx].T)

    # Compute M.
    matrixMI = np.zeros((numOfClasses, 29 * 24))

    for index, kernel in enumerate(kernelOfEachClass):
        for rowIndex, row in enumerate(kernel):
            matrixMI[index, rowIndex] = np.sum(row) / numOfEachClass[idx]
    matrixMStar = np.zeros(29 * 24)
    for index, row in enumerate(kernelOfAll):
        matrixMStar[index] = np.sum(row) / numOfImages
    matrixM = np.zeros((29 * 24, 29 * 24))
    for idx, num in enumerate(numOfEachClass):
        difference = (matrixMI[idx] - matrixMStar).reshape((29 * 24, 1))
        matrixM += num * difference.dot(difference.T)

    # Get N^(-1) * M.
    matrix = np.linalg.pinv(matrixN).dot(matrixM)

    return matrix


# Linear discriminative analysis.
def LDA(mode, numOfNeighbors, kernelType, gamma, trainingImages, trainingLabels, testingImages, testingLabels):
    # Get number of each class and the number of training images.
    _, numOfEachClass = np.unique(trainingLabels, return_counts=True)
    numOfTrainingImages = len(trainingImages)
    numOfTestingImages = len(testingImages)

    # Simple LDA
    if not mode:
        matrix = simpleLDA(numOfEachClass, trainingImages, trainingLabels)
    # Kernel LDA
    else:
        matrix = kernelLDA(numOfEachClass, trainingImages, trainingLabels, kernelType, gamma)

    # Find the first 25 largest eigenvectors.
    targetEigenvectors = findTargetEigenvectors(matrix)

    # Transform eigenvectors into eigenfaces.
    transformEigenvectorsToFaces(targetEigenvectors, 1)

    # Randomly reconstruct 10 eigenfaces.
    reconstructFaces(numOfTrainingImages, trainingImages, targetEigenvectors)

    # Classify and predict.
    classifyAndPredict(numOfTrainingImages, numOfTestingImages, trainingImages, trainingLabels, testingImages,
                       testingLabels, targetEigenvectors, numOfNeighbors)

    # Output the diagram.
    outputDiagram()


def main():
    args = parseArguments()
    # Get parameters.
    algo = int(args.algo)
    mode = int(args.mode)
    numOfNeighbors = int(args.numOfNeighbors)
    kernelType = int(args.kernelType)
    gamma = float(args.gamma)

    trainingImages, trainingLabels = readTrainingImages()
    testingImages, testingLabels = readTestingImages()

    # PCA
    if algo == 0:
        PCA(mode, numOfNeighbors, kernelType, gamma, trainingImages, trainingLabels, testingImages, testingLabels)
    # LDA
    else:
        LDA(mode, numOfNeighbors, kernelType, gamma, trainingImages, trainingLabels, testingImages, testingLabels)


if __name__ == '__main__':
    main()
