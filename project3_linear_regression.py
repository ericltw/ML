import argparse
import matplotlib.pyplot as plt
import numpy as np

# For drawing the variance line.
numOfXPoints: int = 40


# Central Limit Theorem (standard normal).
# Reference: https://en.wikipedia.org/wiki/Normal_distribution
def univariateGaussian(mean, variance):
    # Value of standard normal.
    value = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    # Since a N(μ, σ^2) can be generated as X = μ + σZ, where Z is standard normal.
    return mean + variance ** 0.5 * value


# Polynomial basis linear model data generator.
def dataGenerator(n, a, w):
    x = np.random.uniform(-1, 1)

    y = 0
    for i in range(n):
        y += w[i] * x ** i
    y += univariateGaussian(0, a)

    return x, y


# One dimension.
def getDesignMatrix(n, x):
    result = np.zeros((1, n))

    for i in range(n):
        result[0, i] = x ** i

    return result


def outputOnlineLearning(n, x, y, postVariance, postMean, count, predictiveVariance, predictiveMean):
    print('Add data point({}, {}):'.format(x, y))

    print('\nPosterior mean:')
    for i in range(n):
        print('\t', format(postMean[i, 0], '0.10f'))

    print('\nPosterior Variance:')
    for i in range(n):
        print('\t', end='')
        for j in range(n):
            print(format(postVariance[i, j], '0.10f'), end=', ')
        print('')

    print('\nPredictive distribution ~ N({}, {})'.format(predictiveMean, predictiveVariance))
    print('------------------------------------------------------------------------------')


def draw(x, y, var, lower_bound, upper_bound):
    plt.plot(x, y, color='black')
    plt.plot(x, y + var, color='red')
    plt.plot(x, y - var, color='red')
    plt.xlim(-2.0, 2.0)
    plt.ylim(lower_bound, upper_bound)


# TODO
def visualize(a, n, w, postVariance, postMean, varianceOfTen, meanOfTen, varianceOfFifty, meanOfFifty, recordOfX,
              recordOfY):
    # For drawing the variance line.
    x = np.linspace(-2.0, 2.0, numOfXPoints)

    # Ground truth.
    plt.subplot(221)
    func = np.poly1d(np.flip(w))
    y = func(x)
    var = (1 / a)
    lower_bound = min(y - var) - 10
    upper_bound = max(y + var) + 10
    plt.title('Ground truth')
    draw(x, y, var, lower_bound, upper_bound)

    # Predict result.
    plt.subplot(222)
    func = np.poly1d(np.flip(np.reshape(postMean, n)))
    y = func(x)
    var = np.zeros(numOfXPoints)
    for i in range(numOfXPoints):
        X = getDesignMatrix(n, x[i])
        # Calculate every point's variance.
        var[i] = 1 / a + X.dot(np.linalg.inv(postVariance).dot(X.T))
    plt.title('Predict result')
    plt.scatter(recordOfX, recordOfY, s=7.0)
    draw(x, y, var, lower_bound, upper_bound)

    # After ten incomes.
    plt.subplot(223)
    func = np.poly1d(np.flip(np.reshape(meanOfTen, n)))
    y = func(x)
    var = np.zeros(numOfXPoints)
    for i in range(numOfXPoints):
        X = getDesignMatrix(n, x[i])
        var[i] = 1 / a + X.dot(varianceOfTen.dot(X.T))
    plt.title("After 10 incomes")
    plt.scatter(recordOfX[0:10], recordOfY[0:10], s=7.0)
    draw(x, y, var, lower_bound, upper_bound)

    # After 50 incomes.
    plt.subplot(224)
    func = np.poly1d(np.flip(np.reshape(meanOfFifty, n)))
    y = func(x)
    var = np.zeros(numOfXPoints)
    for i in range(numOfXPoints):
        X = getDesignMatrix(n, x[i])
        var[i] = 1 / a + X.dot(varianceOfFifty.dot(X.T))
    plt.title("After 50 incomes")
    plt.scatter(recordOfX[0:50], recordOfY[0:50], s=7.0)
    draw(x, y, var, lower_bound, upper_bound)

    plt.tight_layout()
    plt.show()


def baysianLinearRegression(a, b, n, w, recordOfX, recordOfY):
    aInverse = 1 / a

    count = 1
    priorMean = np.zeros((1, n))
    priorVariance = np.zeros((n, n))

    while True:
        x, y = dataGenerator(n, a, w)
        recordOfX.append(x)
        recordOfY.append(y)
        # The design matrix of x.
        X = getDesignMatrix(n, x)

        if count == 1:
            bI = b * np.identity(n)
            # Λ = aX^TX+bI
            # The 'a' here is actually the inverse of input 'a', and the following are same.
            postVariance = aInverse * X.T.dot(X) + bI
            # μ = aΛ^-1X^TY
            postMean = aInverse * np.linalg.inv(postVariance).dot(X.T) * y
        else:
            # C = aX^TX+Λ (C actually is also variance)
            postVariance = aInverse * X.T.dot(X) + priorVariance
            # μ = C^-1(aX^TY+Λμ)
            postMean = np.linalg.inv(postVariance).dot(aInverse * X.T.dot(y) + priorVariance.dot(priorMean))

        if count == 10:
            # TODO
            varianceOfTen = np.linalg.inv(postVariance).copy()
            meanOfTen = postMean.copy()
        if count == 50:
            # TODO
            varianceOfFifty = np.linalg.inv(postVariance).copy()
            meanOfFifty = postMean.copy()

        # Predictive Distribution
        predictiveVariance = (1 / aInverse) + X.dot(np.linalg.inv(postVariance).dot(X.T))
        predictiveMean = np.dot(X, postMean)

        outputOnlineLearning(n, x, y, postVariance, postMean, count, predictiveVariance, predictiveMean)

        # TODO
        if np.linalg.norm(priorMean - postMean, ord=2) < 1e-3 and count > 500:
            break

        count += 1
        priorVariance = postVariance
        priorMean = postMean

    visualize(aInverse, n, w, postVariance, postMean, varianceOfTen, meanOfTen, varianceOfFifty, meanOfFifty, recordOfX,
              recordOfY)


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--a')
    parse.add_argument('--b')
    parse.add_argument('--n')
    parse.add_argument('--w')

    args = parse.parse_args()

    a: float = float(args.a)
    b: float = float(args.b)
    n: int = int(args.n)
    w: np.array = np.array([float(item) for item in args.w.split(',')])
    recordOfX = []
    recordOfY = []

    baysianLinearRegression(a, b, n, w, recordOfX, recordOfY)


if __name__ == '__main__':
    main()
