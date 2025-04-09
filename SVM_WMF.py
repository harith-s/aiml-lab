import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import numpy as np

def add_bit(testcases):
    new_testcases = []

    for i in testcases:
        j = i.copy()
        j.append(0)
        new_testcases.append(j)

    for i in testcases:
        print(i)
        i.append(1)
        new_testcases.append(i)

    return new_testcases

def testcases(n):
    if n == 1:
        return ([0, 1], [0, 0])
    elif n == 2:
        return ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
    else:
        testcases = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i in range(n - 2):
            testcases = add_bit(testcases)
        outputs = []
        for testcase in testcases:
            one_count = testcase.count(1)
            if one_count > n / 2:
                outputs.append(1)
        return (testcases, outputs)

n = int(input("Enter the number of bits: "))

X, Y = testcases(n)
X = np.array(X)
Y = np.array(Y)

svm = SVC(kernel="linear", gamma=0.5, C=1.0)
svm.fit(X, Y)

DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        cmap=plt.cm.Spectral,
        alpha=0.8,
        xlabel="X",
        ylabel="Y",
    )

plt.scatter(X[:, 0], X[:, 1], 
            c=Y, 
            s=20, edgecolors="k")
plt.show()
