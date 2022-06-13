#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

file_path_of_image = 'data_project7/tsne_python/mnist2500_X.txt'
file_path_of_label = 'data_project7/tsne_python/mnist2500_labels.txt'


def parseArguments():
    parse = argparse.ArgumentParser()

    # Mode for SNE, 0 -> t-SNE, 1 -> symmetric SNE.
    parse.add_argument('--mode', default=0)
    parse.add_argument('--perplexity', default=20.0)

    return parse.parse_args()


def readInputFile():
    x = np.loadtxt(file_path_of_image)
    label_of_x = np.loadtxt(file_path_of_label)

    return x, label_of_x


# Compute the perplexity and the P-row for a specific value of the
# precision of a Gaussian distribution.
def Hbeta(D=np.array([]), beta=1.0):
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


# Performs a binary search to get P-values in such a way that each
# conditional Gaussian has the same perplexity.
def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


# Runs PCA on the NxD array X in order to reduce its dimensionality to
# no_dims dimensions.
def pca(X=np.array([]), no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def captureState(Y, labels, mode, perplexity):
    plt.clf()
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.title(f'{"t-SNE" if not mode else "symmetric SNE"}, perplexity = {perplexity}')
    plt.tight_layout()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()

    return Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())


def drawSimilarities(p, q, labels):
    # Get sorted index.
    index = np.argsort(labels)
    plt.clf()
    plt.figure(1)

    # Plot p.
    log_p = np.log(p)
    sorted_p = log_p[index][:, index]
    plt.subplot(121)
    img = plt.imshow(sorted_p, cmap='gray', vmin=np.min(log_p), vmax=np.max(log_p))
    plt.colorbar(img)
    plt.title('High dim space')

    # Plot q.
    log_q = np.log(q)
    sorted_q = log_q[index][:, index]
    plt.subplot(122)
    img = plt.imshow(sorted_q, cmap='gray', vmin=np.min(log_q), vmax=np.max(log_q))
    plt.colorbar(img)
    plt.title('Low dim space')

    plt.tight_layout()


# Runs t-SNE on the dataset in the NxD array X to reduce its
# dimensionality to no_dims dimensions. The syntaxis of the function is
# `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
def sne(images, labels, mode, no_dims: int = 2, initial_dims: int = 50,
        perplexity: float = 20.0):
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(images, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    image = []

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iteration in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        # t-SNE
        if mode == 0:
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        # symmetric SNE
        else:
            num = np.exp(-1. * np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        if mode == 0:
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        # symmetric SNE
        else:
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iteration < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iteration + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iteration + 1, C))
            image.append(captureState(Y, labels, mode, perplexity))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Save gif
    filename = f'./output/{"t-SNE" if not mode else "symmetric-SNE"}_{perplexity}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image[0].save(filename, save_all=True, append_images=image[1:], optimize=False, loop=0, duration=200)

    # Plot pairwise similarities in high-dimensional space and low-dimensional space
    drawSimilarities(P, Q, labels)

    # Return solution
    return Y


def main():
    args = parseArguments()
    # Get parameters.
    mode = int(args.mode)
    perplexity = float(args.perplexity)

    x, label_of_x = readInputFile()

    y = sne(x, label_of_x, mode, 2, 50, perplexity)
    plt.figure(2)
    plt.scatter(y[:, 0], y[:, 1], 20, label_of_x)
    plt.title(f'{"t-SNE" if not mode else "symmetric SNE"}, perplexity = {perplexity}')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
