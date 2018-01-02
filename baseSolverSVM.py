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
# exit(-1)

### Create the optimization problem
# Define the objective
numFeatures = X.shape[1]
P = cvxopt.matrix(np.eye(numFeatures).astype(np.double))
P[numFeatures-1, numFeatures-1] = 0.0 # Don't use the bias term in the objective
q = cvxopt.matrix(np.zeros(numFeatures).astype(np.double))

# Inequality constraints
# 1 - y_i(w.T * x_i + b) <= 0
G = -(X * np.expand_dims(y, 1)).astype(np.double)
G = cvxopt.matrix(G)
h = cvxopt.matrix(x=-1.0, size=(100, 1))

# Solve the problem
res = cvxopt.solvers.qp(P=P, q=q, G=G, h=h)
optimalX = weights = np.squeeze(res['x'])
print ("Optimal Weights:", optimalX)

# Check the classification performance
predictions = y * np.dot(X, weights)
print (predictions)
meanAccuracy = np.mean(predictions > 0)
print ("Accuracy:", meanAccuracy)