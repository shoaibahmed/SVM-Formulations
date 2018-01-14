import os
import numpy
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from sklearn.metrics import confusion_matrix

#datapoints to predict and their answers
data = load_svmlight_file("validate.txt", n_features=1199846)
X, y = data[0], data[1]
print ("Y shape: %s" % str(y.shape))

predictions = numpy.load('./predictions.npy')
print ("predictions shape: %s" % str(predictions.shape))

weightsDir = './Weights'
weightFileList = os.listdir(weightsDir)
print (y[:10])
print (predictions[:10])
print (weightFileList[predictions[0]])

# Compare the predictions and the ground-truth
confMat = confusion_matrix(y, predictions)
print (confMat)