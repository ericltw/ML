import argparse
import math
import matplotlib.pyplot as plt
import numpy as np


# Central Limit Theorem (standard normal).
# Reference: https://en.wikipedia.org/wiki/Normal_distribution
def univariateGaussian(mean, variance):
    # Value of standard normal.
    value = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    # Since a N(μ, σ^2) can be generated as X = μ + σZ, where Z is standard normal.
    return mean + variance ** 0.5 * value


def firstDerivative(X, Y, w):
    # A^T(1/(1+e^(-Xi*w))-Yi).
    return X.T.dot(Y - (1 / (1 + np.exp(-X.dot(w)))))


def gradientDescent(X, Y):
    weight = np.random.rand(3, 1)

    count = 0
    while True:
        count += 1
        oldWeight = weight

        # Calculate deltaJ.
        deltaJ = firstDerivative(X, Y, weight)
        # The key of gradient descent.
        weight = weight + deltaJ

        if np.linalg.norm(weight - oldWeight) < 1e-2 or (count > 1e4 and np.linalg.norm(weight - oldWeight) < 80) or count > 1e5:
            break
    return weight


def newton(N, X, Y):
    weight = np.random.rand(3, 1)
    D = np.zeros((N * 2, N * 2))

    count = 0
    while True:
        count += 1
        oldWeight = weight

        # Calculate D.
        for i in range(N * 2):
            e = np.exp(-X[i].dot(weight))
            # TODO: Check the reason of this operation.
            if math.isinf(e):
                e = np.exp(100)
            D[i][i] = e / ((1 + e) ** 2)

        # Calculate H (A^TDA).
        H = X.T.dot(D.dot(X))

        # Calculate deltaF.
        deltaF = firstDerivative(X, Y, weight)

        if np.linalg.det(H) == 0:
            weight = weight + deltaF
        else:
            weight = weight + np.linalg.inv(H).dot(deltaF)

        if np.linalg.norm(weight - oldWeight) < 1e-2 or (count > 1e4 and np.linalg.norm(weight - oldWeight) < 5) or count > 1e5:
            break

    return weight


def visualize(d1, d2, X, Y, weightOfGD, weightOfNewton):
    GDD1 = []
    GDD2 = []
    GDTP = GDFP = GDTN = GDFN = 0
    # Start to evaluate the result of gradient descent.
    for i in range(len(X)):
        # TODO: Check the operation.
        if X[i].dot(weightOfGD) >= 0:
            GDD1.append(X[i, 0:2])
            if Y[i, 0] == 1:
                GDTP += 1
            else:
                GDFP += 1
        else:
            GDD2.append(X[i, 0:2])
            if Y[i, 0] == 0:
                GDTN += 1
            else:
                GDFN += 1
    GDD1 = np.array(GDD1)
    GDD2 = np.array(GDD2)

    ND1 = []
    ND2 = []
    NTP = NFP = NTN = NFN = 0
    # Start to evaluate the result of newton's method.
    for i in range(len(X)):
        # TODO: Check the operation.
        if X[i].dot(weightOfNewton) >= 0:
            ND1.append(X[i, 0:2])
            if Y[i, 0] == 1:
                NTP += 1
            else:
                NFP += 1
        else:
            ND2.append(X[i, 0:2])
            if Y[i, 0] == 0:
                NTN += 1
            else:
                NFN += 1
    ND1 = np.array(ND1)
    ND2 = np.array(ND2)

    # Print the results.
    print('Gradient descent:\n')
    print('w:\n', weightOfGD[0, 0], '\n', weightOfGD[1, 0], '\n', weightOfGD[2, 0])
    print('Confusion Matrix:')
    print('\t\t\t Is cluster 1\t Is cluster 2')
    print('Predict cluster 1\t   ', GDTP, '\t\t   ', GDFN)
    print('Predict cluster 2\t   ', GDFP, '\t\t   ', GDTN)
    print('\nSensitivity (Successfully predict cluster 1): ', GDTP / (GDTP + GDFN))
    print('Specificity (Successfully predict cluster 2): ', GDTN / (GDTP + GDFP))

    print('\n--------------------------------------')
    print('Newton\'s method\n')
    print('w:\n', weightOfNewton[0, 0], '\n', weightOfNewton[1, 0], '\n', weightOfNewton[2, 0])
    print('Confusion Matrix:')
    print('\t\t\t Is cluster 1\t Is cluster 2')
    print('Predict cluster 1\t   ', NTP, '\t\t   ', NFN)
    print('Predict cluster 2\t   ', NFP, '\t\t   ', NTN)
    print('\nSensitivity (Successfully predict cluster 1): ', NTP / (NTP + NFN))
    print('Specificity (Successfully predict cluster 2): ', NTN / (NTP + NFN))

    # Output ground truth.
    plt.figure()
    plt.subplot(131)
    plt.scatter(d1[:, 0], d1[:, 1], c='r')
    plt.scatter(d2[:, 0], d2[:, 1], c='b')

    # Output gradient descent.
    plt.subplot(132)
    if len(GDD1) != 0:
        plt.scatter(GDD1[:, 0], GDD1[:, 1], c='b')
    if len(GDD2) != 0:
        plt.scatter(GDD2[:, 0], GDD2[:, 1], c='r')

    # Output newton's method.
    plt.subplot(133)
    if len(ND1) != 0:
        plt.scatter(ND1[:, 0], ND1[:, 1], c='b')
    if len(ND2) != 0:
        plt.scatter(ND2[:, 0], ND2[:, 1], c='r')

    plt.tight_layout()
    plt.show()


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--N')
    parse.add_argument('--mx1my1')
    parse.add_argument('--mx2my2')
    parse.add_argument('--vx1vy1')
    parse.add_argument('--vx2vy2')

    args = parse.parse_args()

    N = int(args.N)
    mx1 = my1 = float(args.mx1my1)
    mx2 = my2 = float(args.mx2my2)
    vx1 = vy1 = float(args.vx1vy1)
    vx2 = vy2 = float(args.vx2vy2)

    d1 = np.zeros((N, 2))
    d2 = np.zeros((N, 2))

    # Generate data points.
    for i in range(N):
        d1[i, 0] = univariateGaussian(mx1, vx1)
        d1[i, 1] = univariateGaussian(my1, vy1)
        d2[i, 0] = univariateGaussian(mx2, vx2)
        d2[i, 1] = univariateGaussian(my2, vy2)

    # TODO: Check why the width of X is 3.
    # Init X.
    X = np.ones((N * 2, 3))
    X[0:N, 0:2] = d1
    X[N:N * 2, 0:2] = d2

    # Init Y, i.e. the ground truth class of data points.
    Y = np.zeros((N * 2, 1), dtype=int)
    Y[N:N * 2, 0] = 1

    weightOfGD = gradientDescent(X, Y)
    weightOfNewton = newton(N, X, Y)

    visualize(d1, d2, X, Y, weightOfGD, weightOfNewton)


if __name__ == '__main__':
    main()
