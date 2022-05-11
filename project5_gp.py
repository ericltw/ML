import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import string

inputFilePath: string = './data_project5/input.data'
noise: float = 5.0


def loadData():
    # Load data from input file.
    data = np.loadtxt(inputFilePath, dtype=float)

    # Reshape the data to x and y.
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)

    return x, y


# Reference: https://en.wikipedia.org/wiki/Rational_quadratic_covariance_function
def rationalQuadraticKernel(x1, x2, alpha, lengthScale):
    # variance = (1 + d ^ 2 / 2αl ^ 2) ^ (-α)
    return (1 + cdist(x1, x2, 'sqeuclidean') / (2 * alpha * lengthScale ** 2)) ** -alpha


def gaussianProcess(xOfTraining, yOfTraining, alpha=1.0, lengthScale=1.0):
    # Generate the testing points.
    numOfTestingPoints = 1000
    xOfTesting = np.linspace(-60, 60, numOfTestingPoints).reshape(-1, 1)

    # Compute covariance matrix of training data.
    covOfTraining = rationalQuadraticKernel(xOfTraining, xOfTraining, alpha, lengthScale)

    # Compute the kernel of testing data to testing data.
    kernelStar = np.add(rationalQuadraticKernel(xOfTesting, xOfTesting, alpha, lengthScale),
                        np.eye(len(xOfTesting)) / noise)

    # Compute the kernel of training data to testing data.
    kernel = rationalQuadraticKernel(xOfTraining, xOfTesting, alpha, lengthScale)

    # Compute mean and variance.
    mean = kernel.T.dot(np.linalg.inv(covOfTraining)).dot(yOfTraining).ravel()
    variance = kernelStar - kernel.T.dot(np.linalg.inv(covOfTraining)).dot(kernel)

    # Compute 95% confidence upper and lower bound.
    upperBound = mean + 1.96 * variance.diagonal()
    lowerBound = mean - 1.96 * variance.diagonal()

    # Output the graph.
    plt.xlim(-60, 60)
    plt.title(f'Gaussian process, alpha={alpha:.3f}, length={lengthScale:.3f}')
    plt.scatter(xOfTraining, yOfTraining, c='k')
    plt.plot(xOfTesting.ravel(), mean, 'b')
    plt.fill_between(xOfTesting.ravel(), upperBound, lowerBound, color='b', alpha=0.5)
    plt.tight_layout()
    plt.show()


def marginalLogLikelihood(theta):
    # Compute covariance matrix
    covariance = rationalQuadraticKernel(x, x, alpha=theta[0], lengthScale=theta[1])

    # - ln p(y|θ) = 0.5 * ln|C| + 0.5 * y ^ T * C ^ (-1) * y + N / 2 * ln(2π)
    return 0.5 * np.log(np.linalg.det(covariance)) + 0.5 * y.ravel().T.dot(np.linalg.inv(covariance)).dot(
        y.ravel()) + numOfPoints / 2.0 * np.log(2.0 * np.pi)


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode')

    args = parse.parse_args()

    mode = int(args.mode)

    global x, y, numOfPoints
    x, y = loadData()
    numOfPoints = len(x)

    if mode == 1:
        gaussianProcess(x, y)
    elif mode == 2:
        initGuess = np.array([1.0, 1.0])
        result = minimize(marginalLogLikelihood, initGuess)

        optAlpha, optLengthScale = result.x
        gaussianProcess(x, y, optAlpha, optLengthScale)


if __name__ == '__main__':
    main()
