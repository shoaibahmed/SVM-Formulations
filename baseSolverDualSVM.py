import numpy as np
from sklearn import datasets

# Import the optimization package
import cvxopt

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
		P[i, j] = y[i] * y[j] * np.dot(X[i, :], X[j, :])
# P = X * np.expand_dims(y, -1)
# P = np.dot(P, P.T).astype(np.double)
q = np.full(numExamples, -1, np.double)

# Inequality constraints
# -a <= 0
G = -np.eye(numExamples, dtype=np.double)
h = np.zeros(numExamples, np.double)

# Equality constraints
# y.T * a = 0
A = y * np.eye(numExamples, dtype=np.double)
b = np.zeros(numExamples, np.double)

# Solve the problem
res = cvxopt.solvers.qp(P=cvxopt.matrix(P), q=cvxopt.matrix(q), G=cvxopt.matrix(G), h=cvxopt.matrix(h), A=cvxopt.matrix(A), b=cvxopt.matrix(b))
optimalAlpha = np.squeeze(res['x'])
print ("Optimal Alpha:", optimalAlpha)

weights = np.zeros(numFeatures)
for i in range(numExamples):
	weights += optimalAlpha[i] * y[i] * X[i]
print ("Optimal Weights:", weights)

# Check the classification performance
predictions = y * np.dot(X, weights)
print ("Predictions:", predictions)
meanAccuracy = np.mean(predictions > 0)
print ("Accuracy:", meanAccuracy)