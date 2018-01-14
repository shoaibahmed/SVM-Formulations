import os
import numpy
from sklearn.datasets import load_svmlight_file
import scipy.sparse

weightsDir = './Weights'

#datapoints to predict and their answers
data = load_svmlight_file("validate.txt", n_features=1199845, zero_based=True)
X, y = data[0], data[1]
X = sparse.csr_matrix(X)
X = sparse.hstack([X,numpy.ones((X.shape[0],1))])
print ("X shape: %s" % str(X.shape))

#this is a LIST of all hyperplanes
evaluation = []
weightFileList = os.listdir(weightsDir)
for idx, weightVectorFileName in enumerate(weightFileList):
	# weightVector = numpy.load(os.path.abspath(os.path.join(weightsDir, weightVectorFileName)))
	# weightVector = sparse.csr_matrix(weightVector)
	weightVector = scipy.sparse.load_npz(os.path.abspath(os.path.join(weightsDir, weightVectorFileName)))
	print ("Loading weight vector # %d | Weights shape: %s" % (idx, str(weightVector.shape)))
	out = X.dot(weightVector)
	out = out.toarray()
	# print ("Output shape: %s" % str(out.shape))
	evaluation.append(numpy.squeeze(out))

#perform prediction for each data point over all hyperplanes
evaluation = numpy.array(evaluation)
evaluation = evaluation.T
print ("Evaluation shape: %s" % str(evaluation.shape))

#get argmax from each point prediction over all hyperplanes
predictions = numpy.argmax(evaluation, axis=1)
print ("Predictions shape: %s" % str(predictions.shape))

actualYEntries = []
for weightFileName in weightFileList:
	yEntry = int(weightFileName[weightFileName.rfind('_')+1:weightFileName.rfind('.')])
	actualYEntries.append(yEntry)
print ("Actual Y Entries shape: %s" % str(actualYEntries.shape))
print (actualYEntries[:10])

# Convert the labels to the actual labels that we have at hand
for i in range(predictions.shape[0]):
	predictions[i] = actualYEntries[predictions[i]]

#now compare between y & yresults, compute whatever accuracy measure
print ("Saving predictions")
numpy.save('./predictions.npy', predictions)
