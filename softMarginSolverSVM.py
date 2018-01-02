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
X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1) # Append a 1 at the end of the feature vector to deal with the bias
y[y == 0] = -1

print ("After filtering")
print ("Data (shape):", X.shape)
print ("Labels (shape):", y.shape)
print ("Data:", X[:5, :])
print ("Labels:", np.unique(y))

### Create the optimization problem
C = 1
numExamples = X.shape[0]
numFeatures = X.shape[1]
numTotalVars = numExamples + numFeatures # Num examples to add sai_i

# Define the objective
P = np.zeros((numTotalVars, numTotalVars)).astype(np.double)
for i in range(numFeatures - 1): # Discard the bias and sai_i for the computation of objective
	P[i, i] = 1.0
q = np.full(numTotalVars, C).astype(np.double)
for i in range(numFeatures): # Discard the weights and bias term in the computation of the regularization term
	q[i] = 0.0

# Inequality constraints
# 1 - y_i(w.T * x_i + b) <= 0
G = -(X * np.expand_dims(y, 1))
I = np.eye(numExamples)
G = np.concatenate((G, -I), axis=1)
lowerG = np.concatenate((np.zeros(X.shape), -I), axis=1)
G = np.concatenate((G, lowerG), axis=0)
G = G.astype(np.double)
print (G.shape)
h = np.full((200, 1), -1.0, np.double)
h[100:] = 0.0

# Solve the problem
res = cvxopt.solvers.qp(P=cvxopt.matrix(P), q=cvxopt.matrix(q), G=cvxopt.matrix(G), h=cvxopt.matrix(h))
optimalX = weights = np.squeeze(res['x'])
print ("Optimal Weights:", optimalX)

# Check the classification performance
predictions = y * np.dot(X, weights[:numFeatures])
print (predictions)
meanAccuracy = np.mean(predictions > 0)
print ("Accuracy:", meanAccuracy)