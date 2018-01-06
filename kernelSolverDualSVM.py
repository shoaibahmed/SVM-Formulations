import numpy as np
from sklearn import datasets

# Import the optimization package
import cvxopt

# Define constants
epsilon = 1e-4
gamma = 1
normalizeWeights = False

# Define the kernel function
from enum import Enum
class KernelType(Enum):
	LINEAR = 1
	RBF = 2
kernelToUse = KernelType.RBF

def kernel(X, XPrime, kernelType=kernelToUse):
	if kernelType == KernelType.LINEAR:
		out = np.dot(X, XPrime)
	elif kernelType == KernelType.RBF:
		diff = X - XPrime
		out = np.exp(-gamma * np.dot(diff, diff))
	else:
		print ("Error: Unknown Kernel")
		exit (-1)

	return out

# Load the datset
iris = datasets.load_iris()

### Binary classifier for setosa and versicolor
X = iris.data
y = iris.target
print ("Before filtering")
print ("Data (shape):", X.shape)
print ("Labels (shape):", y.shape)
print ("Data:", X[:5, :])
print ("Labels:", np.unique(y))

# Filter the classes
selectedIndices = np.logical_or(y == 0, y == 1)
X = X[selectedIndices]
y = y[selectedIndices]
# X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1) # Append a 1 at the end of the feature vector to deal with the bias
y[y == 0] = -1

print ("After filtering")
print ("Data (shape):", X.shape)
print ("Labels (shape):", y.shape)
print ("Data:", X[:5, :])
print ("Labels:", np.unique(y))

### Create the optimization problem
# Define the objective
# 1/2 a.T * (y * y * X.T * X) * a - a
numExamples = X.shape[0]
numFeatures = X.shape[1]
P = np.zeros((numExamples, numExamples), np.double)
for i in range(numExamples):
	for j in range(numExamples):
		P[i, j] = y[i] * y[j] * kernel(X[i, :], X[j, :])
		# P[i, j] = y[i] * y[j] * np.dot(X[i, :], X[j, :])
# P = (y[:, None] * X).astype(np.double)
# P = np.dot(P, P.T)
q = np.full(numExamples, -1, np.double)

# Inequality constraints
# -a <= 0
G = -np.eye(numExamples, dtype=np.double)
h = np.zeros(numExamples, np.double)

# Equality constraints
# y.T * a = 0
A = y.astype(np.double).reshape(1, numExamples)
b = np.zeros(1, np.double)

# Solve the problem
res = cvxopt.solvers.qp(P=cvxopt.matrix(P), q=cvxopt.matrix(q), G=cvxopt.matrix(G), h=cvxopt.matrix(h), A=cvxopt.matrix(A), b=cvxopt.matrix(b))
optimalAlpha = np.squeeze(res['x'])
print ("Optimal Alpha:", optimalAlpha)

# Determine b
supportVectors = [idx for idx in range(optimalAlpha.shape[0]) if optimalAlpha[idx] > epsilon]
supportVectorsAlpha = optimalAlpha[supportVectors]
print ("Support vectors (Indices):", supportVectors)
print ("Support vectors (Alpha):", supportVectorsAlpha)

for currentSupportVectorIdx in supportVectors:
	y_m = y[currentSupportVectorIdx]
	X_m = X[currentSupportVectorIdx]
	bInitial = None
	currentSum = 0
	for supportVectorIdx in supportVectors:
		currentSum += (optimalAlpha[supportVectorIdx] * y[supportVectorIdx] * kernel(X[supportVectorIdx, :], X_m))

	# Compute b
	b = y_m - currentSum
	print ("Found b:", b)
	if bInitial is None:
		bInitial = b
	assert (np.abs(bInitial - b) < epsilon)

# Check the classification performance
predictions = []
for i in range(numExamples):
	y_m = y[i]
	X_m = X[i]
	currentSum = 0
	for supportVectorIdx in supportVectors:
		currentSum += (optimalAlpha[supportVectorIdx] * y[supportVectorIdx] * kernel(X[supportVectorIdx, :], X_m))
	prediction = currentSum + b
	predictions.append(1 if prediction >= 0 else -1)
print ("Predictions:", predictions)
meanAccuracy = np.mean(predictions == y)
print ("Accuracy:", meanAccuracy)