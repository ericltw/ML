import argparse
import numpy as np
import os
from PIL import Image
from scipy.spatial.distance import cdist
import string

imageDirectory: string = 'data_project6'
filePathOfImage1: string = os.path.join(imageDirectory, 'image1.png')
filePathOfImage2: string = os.path.join(imageDirectory, 'image2.png')

# Used for output the gif.
colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])


# Set up the input parameters, and return args.
def parseArguments():
    parse = argparse.ArgumentParser()

    # Number of clusters.
    parse.add_argument('--cluster', default=2)
    # Parameter gamma_s in kernel.
    parse.add_argument('--gammas', default=0.0001)
    # Parameter gamma_c in kernel.
    parse.add_argument('--gammac', default=0.001)
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


def computeKernel(image, gammaS, gammaC):
    # Get image shape.
    rows, cols, colors = image.shape

    # Compute the color distance.
    numOfPixels = rows * cols
    colorDist = cdist(image.reshape(numOfPixels, colors), image.reshape(numOfPixels, colors), 'sqeuclidean')

    # Compute the indices of a grid.
    indices = np.indices((rows, cols))
    indicesOfRow = indices[0]
    indicesOfCol = indices[1]

    # Compute the indices vector.
    indicesVector = np.hstack((indicesOfRow.reshape(-1, 1), indicesOfCol.reshape(-1, 1)))

    # Compute the spatial distance.
    spatialDist = cdist(indicesVector, indicesVector, 'sqeuclidean')

    # The kernel formula in spec.
    return np.multiply(np.exp(-gammaS * spatialDist), np.exp(-gammaC * colorDist))


def initCenters(numOfRows, numOfCols, numOfClusters, initMode):
    if initMode == 0:
        # Random strategy.
        return np.random.choice(100, (numOfClusters, 2))
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
        centers = [indicesVector[np.random.choice(numOfPixels, 1)[0]].tolist()]

        # Find remaining centers.
        for _ in range(numOfClusters - 1):
            # Compute the min distance for each point to all found centers.
            distance = np.zeros(numOfPixels)

            for index, indice in enumerate(indicesVector):
                minDistance = np.Inf

                for center in centers:
                    dist = np.linalg.norm(indice - center)
                    minDistance = dist if dist < minDistance else minDistance
                distance[index] = minDistance

            # Divide the distance by its sum.
            distance /= np.sum(distance)
            # Compute the new center and append to centers array.
            centers.append(indicesVector[np.random.choice(numOfPixels, 1, p=distance)[0]].tolist())

        return np.array(centers)


def initClusters(numOfRows, numOfCols, numOfClusters, kernel, initMode):
    # Init centers.
    centers = initCenters(numOfRows, numOfCols, numOfClusters, initMode)

    # K-means.
    numOfPixels = numOfRows * numOfCols
    clusters = np.zeros(numOfPixels, dtype=int)

    for pixel in range(numOfPixels):
        # Compute the distance of every pixel to all centers.
        distance = np.zeros(numOfClusters)

        for index, center in enumerate(centers):
            seqOfCenter = center[0] * numOfRows + center[1]
            distance[index] = kernel[pixel, pixel] + kernel[seqOfCenter, seqOfCenter] - 2 * kernel[pixel, seqOfCenter]
        # Pick the index of minimum distance as the cluster of the point
        clusters[pixel] = np.argmin(distance)

    return clusters


def getCurrentImageState(numOfRows, numOfCols, clusters):
    numOfPixels = numOfRows * numOfCols
    state = np.zeros((numOfPixels, 3))

    # Give every point a color according to its cluster.
    for pixel in range(numOfPixels):
        state[pixel, :] = colors[clusters[pixel], :]
    state = state.reshape((numOfRows, numOfCols, 3))

    return Image.fromarray(np.uint8(state))


def getSumOfPairwiseDistance(numOfPixels, numOfClusters, numOfMembers, kernel, clusters):
    pairwise = np.zeros(numOfClusters)

    for cluster in range(numOfClusters):
        tempKernel = kernel.copy()

        for pixel in range(numOfPixels):
            # Set distance to 0 if the point doesn't belong to the cluster
            if clusters[pixel] != cluster:
                tempKernel[pixel, :] = 0
                tempKernel[:, pixel] = 0
        pairwise[cluster] = np.sum(tempKernel)

    # Avoid division by 0
    numOfMembers[numOfMembers == 0] = 1

    return pairwise / numOfMembers ** 2


def kernelClustering(numOfPixels, numOfClusters, kernel, clusters):
    # Get number of members in each cluster
    numOfMembers = np.array([np.sum(np.where(clusters == c, 1, 0)) for c in range(numOfClusters)])

    # Get sum of pairwise kernel distances of each cluster
    pairwise = getSumOfPairwiseDistance(numOfPixels, numOfClusters, numOfMembers, kernel, clusters)

    newClusters = np.zeros(numOfPixels, dtype=int)
    for p in range(numOfPixels):
        distance = np.zeros(numOfClusters)
        for c in range(numOfClusters):
            distance[c] += kernel[p, p] + pairwise[c]

            # Get distance from given data point to others in the target cluster
            distToOthers = np.sum(kernel[p, :][np.where(clusters == c)])
            distance[c] -= 2.0 / numOfMembers[c] * distToOthers
        newClusters[p] = np.argmin(distance)

    return newClusters


def kernelKmeans(numOfRows, numOfCols, numOfClusters, clusters, kernel, initMode, index):
    # Record the image state.
    imageStates = []

    # Get first image state.
    firstImageState = getCurrentImageState(numOfRows, numOfCols, clusters)
    imageStates.append(firstImageState)

    # Kernel k-means.
    currentClusters = clusters.copy()
    count = 0
    iteration = 100

    while True:
        # Compute new clusters.
        numOfPixels = numOfRows * numOfCols
        newClusters = kernelClustering(numOfPixels, numOfClusters, kernel, currentClusters)

        # Get the image state.
        imageState = getCurrentImageState(numOfRows, numOfCols, newClusters)
        imageStates.append(imageState)

        if np.linalg.norm((newClusters - currentClusters), ord=2) < 0.001 or count >= iteration:
            break

        currentClusters = newClusters.copy()
        count += 1

    # Output the gif result.
    filename = f'./output/kernel_kmeans/kernel_kmeans_{index}_' \
               f'cluster{numOfClusters}_' \
               f'{"kmeans++" if initMode else "random"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageStates[0].save(filename, save_all=True, append_images=imageStates[1:], optimize=False, loop=0, duration=100)


def main():
    args = parseArguments()
    # Get parameters.
    numOfClusters = int(args.cluster)
    gammaS = float(args.gammas)
    gammaC = float(args.gammac)
    initMode = int(args.initmode)

    images = readImagesInNpArray()

    for index, image in enumerate(images):
        # Compute the kernel.
        kernel = computeKernel(image, gammaS, gammaC)

        # Init clusters.
        numOfRows, numOfCols, _ = image.shape
        clusters = initClusters(numOfRows, numOfCols, numOfClusters, kernel, initMode)

        # Start kernel k-means.
        kernelKmeans(numOfRows, numOfCols, numOfClusters, clusters, kernel, initMode, index)


if __name__ == '__main__':
    main()
