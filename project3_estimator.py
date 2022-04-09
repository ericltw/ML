import argparse
import numpy as np


# Central Limit Theorem (standard normal).
# Reference: https://en.wikipedia.org/wiki/Normal_distribution
def univariateGaussian(mean, variance):
    # Value of standard normal.
    value = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    # Since a N(μ, σ^2) can be generated as X = μ + σZ, where Z is standard normal.
    return mean + variance ** 0.5 * value


# Welford's online algorithm.
#
# Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def sequenceEstimator(mean, variance):
    count = 1
    newValue = univariateGaussian(mean, variance)
    sampleVariance = 0
    sampleMean = newValue
    M2 = 0

    while True:
        count += 1
        newValue = univariateGaussian(mean, variance)
        delta = newValue - sampleMean
        sampleMean += delta / count
        delta2 = newValue - sampleMean
        M2 += delta * delta2
        sampleVariance = M2 / (count - 1)

        print("Add data point: {}".format(newValue))
        print("Mean = {}\tVariance = {}".format(sampleMean, sampleVariance))

        error = 1e-2
        if abs(sampleMean - mean) < error and abs(sampleVariance - variance) < error:
            break


def main():
    parse = argparse.ArgumentParser()

    # mean and variance of univariate gaussian data generator.
    parse.add_argument('--m')
    parse.add_argument('--s')

    args = parse.parse_args()

    mean: float = float(args.m)
    variance: float = float(args.s)

    print('Data point source function: N({}, {})\n'.format(mean, variance))
    sequenceEstimator(mean, variance)


if __name__ == '__main__':
    main()
