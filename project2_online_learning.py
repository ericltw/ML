import argparse
import math

fileName = './data_project2/testfile.txt'


def readFile():
    outcomes = []
    with open(fileName, 'r') as fp:
        for line in fp:
            outcomes.append(line.strip())
    return outcomes


def analyzeOutcomes(outcome):
    numOfZero = 0
    numOfOne = 0

    for element in outcome:
        if element == '0':
            numOfZero += 1
        elif element == '1':
            numOfOne += 1

    return numOfZero, numOfOne


def getCombination(n, m):
    return math.factorial(n) / (math.factorial(m) * math.factorial(n - m))


def getBinomial(numOfFail, numOfOne, theta):
    return (1 - theta) ** numOfFail * theta ** numOfOne


def getLikelihood(numOfFail, numOfSuccess):
    n = numOfFail + numOfSuccess
    m = numOfSuccess
    combination = getCombination(n, m)

    theta = numOfSuccess / (numOfFail + numOfSuccess)
    binomial = getBinomial(numOfFail, numOfSuccess, theta)

    return combination * binomial


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--a')
    parse.add_argument('--b')

    args = parse.parse_args()

    priorA: int = int(args.a)
    priorB: int = int(args.b)

    outcomes = readFile()

    for i in range(len(outcomes)):
        numOfFail, numOfSuccess = analyzeOutcomes(outcomes[i])
        likelihood = getLikelihood(numOfFail, numOfSuccess)
        # Update as posterior.
        posteriorA = numOfSuccess + priorA
        posteriorB = numOfFail + priorB

        print("case", i+1, ":", outcomes[i])
        print("Likelihood:", likelihood)
        print("Beta prior:\ta =", priorA, "b =", priorB)
        print("Beta posterior:\ta =", posteriorA, "b =", posteriorB, "\n")

        priorA = posteriorA
        priorB = posteriorB


if __name__ == '__main__':
    main()
