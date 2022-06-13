import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from project6_kmeans import getCurrentImageState, computeKernel
import string

imageDirectory: string = 'data_project6'
filePathOfImage1: string = os.path.join(imageDirectory, 'image1.png')
filePathOfImage2: string = os.path.join(imageDirectory, 'image2.png')


# Set up the input parameters, and return args.
def parseArguments():
    parse = argparse.ArgumentParser()

    # Number of clusters.
    parse.add_argument('--cluster', default=2)
    # Parameter gamma_s in kernel.
    parse.add_argument('--gammas', default=0.0001)
    # Parameter gamma_c in kernel.
    parse.add_argument('--gammac', default=0.001)
    # Type for cut, 0 -> ratio cut, 1 -> normalized cut.
    parse.add_argument('--cutmode', default=0)
    # Mode for initializing clusters, 0 -> randomly, 1 -> kmeans++.
    parse.add_argument('--initmode', default=0)

    return parse.parse_args()


def readImagesInNpArray():
    # Read the image files.
    images = [Image.open(filePathOfImage1), Image.open(filePathOfImage2)]

    # Convert image to numpy array.
    images[0] = np.asarray(images[0])
    images[1] = np.asarray(images[1])

    return images


# Compute matrixU which containing eigenvectors.
def computeMatrixU(matrixW, cutMode, numOfClusters):
    # Compute degree matrixD and Laplacian matrixL.
    matrixD = np.zeros_like(matrixW)
    for index, row in enumerate(matrixW):
        matrixD[index, index] += np.sum(row)
    matrixL = matrixD - matrixW

    # Normalized cut.
    if cutMode == 1:
        # Compute the normalized Laplacian matrixL.
        for idx in range(len(matrixD)):
            matrixD[idx, idx] = 1.0 / np.sqrt(matrixD[idx, idx])
        matrixL = matrixD.dot(matrixL).dot(matrixD)

    # Compute eigenvalues and eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eig(matrixL)
    eigenvectors = eigenvectors.T

    # Sort the eigenvalues and find indices of nonzero eigenvalues.
    sortedIdx = np.argsort(eigenvalues)
    sortedIdx = sortedIdx[eigenvalues[sortedIdx] > 0]

    return eigenvectors[sortedIdx[:numOfClusters]].T


def initCenters(numOfRows, numOfCols, numOfClusters, matrixU, initMode):
    if initMode == 1:
        # Random strategy.
        numOfPixels = numOfRows * numOfCols
        return matrixU[np.random.choice(numOfPixels, numOfClusters)]
    else:
        # k-means++ strategy.
        # Compute indices of a grid.
        indices = np.indices((numOfRows, numOfCols))
        indicesOfRow = indices[0]
        indicesOfCol = indices[1]

        # Compute the indices vector.
        indicesVector = np.hstack((indicesOfRow.reshape(-1, 1), indicesOfCol.reshape(-1, 1)))

        # Randomly pick first center.
        numOfPixels = numOfRows * numOfCols
        centers = [indices[np.random.choice(numOfPixels, 1)[0]].tolist()]

        # Find remaining centers.
        for _ in range(numOfClusters - 1):
            # Compute min distance for each point to all found centers.
            distance = np.zeros(numOfPixels)

            for index, indice in enumerate(indicesVector):
                minDistance = np.Inf

                for center in centers:
                    dist = np.linalg.norm(indice - center)
                    minDistance = dist if dist < minDistance else minDistance
                distance[index] = minDistance

            # Divide the distance by its sum to get probability.
            distance /= np.sum(distance)
            # Get a new center.
            centers.append(indices[np.random.choice(numOfPixels, 1, p=distance)[0]].tolist())

        # Change from index to feature index.
        for index, center in enumerate(centers):
            centers[index] = matrixU[center[0] * numOfRows + center[1], :]

        return np.array(centers)


def kmeansClustering(numOfPixels, numOfClusters, matrixU, currentCenters):
    newClusters = np.zeros(numOfPixels, dtype=int)

    for pixel in range(numOfPixels):
        # Find min distance from data point to centers.
        distance = np.zeros(numOfClusters)
        for index, center in enumerate(currentCenters):
            distance[index] = np.linalg.norm((matrixU[pixel] - center), ord=2)
        # Classify data point into cluster according to the closest center.
        newClusters[pixel] = np.argmin(distance)

    return newClusters


def kmeansRecomputeCenters(numOfClusters, matrixU, currentClusters):
    newCenters = []

    for cluster in range(numOfClusters):
        pointsInCluster = matrixU[currentClusters == cluster]
        newCenter = np.average(pointsInCluster, axis=0)
        newCenters.append(newCenter)

    return np.array(newCenters)


def kmeans(numOfRows, numOfCols, numOfClusters, matrixU, centers, index, initMode, cutMode):
    # Record the image state.
    numOfPixels = numOfRows * numOfCols
    imageStates = []

    # Kernel k-means.
    currentCenters = centers.copy()
    newClusters = np.zeros(numOfPixels, dtype=int)
    count = 0
    iteration = 100

    while True:
        # Compute new cluster.
        newClusters = kmeansClustering(numOfPixels, numOfClusters, matrixU, currentCenters)

        # Compute new centers.
        newCenters = kmeansRecomputeCenters(numOfClusters, matrixU, newClusters)

        # Get new state.
        imageStates.append(getCurrentImageState(numOfRows, numOfCols, newClusters))

        if np.linalg.norm((newCenters - currentCenters), ord=2) < 0.01 or count >= iteration:
            break

        # Update current parameters.
        currentCenters = newCenters.copy()
        count += 1

    # Output the gif result.
    filename = f'./output/spectral_clustering/spectral_clustering_{index}_' \
               f'cluster{numOfClusters}_' \
               f'{"kmeans++" if initMode else "random"}_' \
               f'{"normalized" if cutMode else "ratio"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if len(imageStates) > 1:
        imageStates[0].save(filename, save_all=True, append_images=imageStates[1:], optimize=False, loop=0,
                            duration=100)
    else:
        imageStates[0].save(filename)

    return newClusters


def plotResult(matrixU, clusters, index, initMode, cutMode):
    colors = ['r', 'b']
    plt.clf()

    for idx, point in enumerate(matrixU):
        plt.scatter(point[0], point[1], c=colors[clusters[idx]])

    # Save the figure.
    filename = f'./output/spectral_clustering/eigenspace_{index}_' \
               f'{"kmeans++" if initMode else "random"}_' \
               f'{"normalized" if cutMode else "ratio"}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)


def spectralClustering(numOfRows, numOfCols, numOfClusters, matrixU, initMode, cutMode, index):
    # Init centers.
    centers = initCenters(numOfRows, numOfCols, numOfClusters, matrixU, initMode)

    # K-means.
    clusters = kmeans(numOfRows, numOfCols, numOfClusters, matrixU, centers, index, initMode, cutMode)

    # Plot data points in eigenspace if number of clusters is 2.
    if numOfClusters == 2:
        plotResult(matrixU, clusters, index, initMode, cutMode)


def main():
    args = parseArguments()
    # Get parameters.
    numOfClusters = int(args.cluster)
    gammaS = float(args.gammas)
    gammaC = float(args.gammac)
    cutMode = int(args.cutmode)
    initMode = int(args.initmode)

    images = readImagesInNpArray()

    for index, image in enumerate(images):
        # Compute the kernel.
        kernel = computeKernel(image, gammaS, gammaC)

        # Compute matrixU (containing eigenvectors).
        matrixU = computeMatrixU(kernel, cutMode, numOfClusters)
        # Normalized cut.
        if cutMode == 1:
            sumOfEachRow = np.sum(matrixU, axis=1)
            for row in range(len(matrixU)):
                matrixU[row, :] /= sumOfEachRow[row]

        # Spectral clustering.
        rows, cols, _ = image.shape
        spectralClustering(rows, cols, numOfClusters, matrixU, initMode, cutMode, index)


if __name__ == '__main__':
    main()
