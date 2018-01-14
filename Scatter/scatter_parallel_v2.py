from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import scipy.sparse
import numpy
import time
from scatter_qp import QP
import multiprocessing as mp

import shutil
import pickle
import os

pickleDir = './PickleDir'
weightsDir = './Weights'
logFile = open("LogFile.txt", "w")

def parallel(y_entry): 
	# Load the data
	y_entry = int(y_entry)
	fileName = os.path.abspath(os.path.join('./PickleDir', str(y_entry) + '.txt'))
	dataInClass = load_svmlight_file(fileName, n_features=numInputDims, zero_based=True)
	g1, currentY = dataInClass[0], dataInClass[1]

	g1 = scipy.sparse.hstack([-g1,-numpy.ones((g1.shape[0],1))])
	g1 = scipy.sparse.csc_matrix(g1)
	# h1 = numpy.array([-1+val.dot(w_bar) for val in g1])
	h1 = g1.dot(w_bar).toarray() - 1
	h1.shape = (g1.shape[0],1)

	# Shape of the weights will vary based on the number of examples in g1
	weights = QP(numInputDims,g1.shape[0],w_bar.toarray(),numpy.ones((g1.shape[0],1)),g1,h1)
	# Discard the sai_i's
	weights = weights[:-g1.shape[0]].reshape(-1, 1)
	weights = scipy.sparse.csc_matrix(weights) # For sparse vectors
	fileName = os.path.abspath(os.path.join('./Weights', 'w_' + str(y_entry) + '.npz'))
	scipy.sparse.save_npz(fileName, weights)
	# numpy.save(fileName, weights)
	print ("Weight vector computed for class: %d" % (y_entry))
	
print ("Loading dataset")
data = load_svmlight_file("train.txt", zero_based=True)
X, y = data[0], data[1]
numInputDims = X.shape[1]
print ("Number of input dimensions: %d" % numInputDims)

y_entries, y_count = numpy.unique(y, return_counts=True)
print ("Dataset loaded successfully") 

# Create the weight vector files
resumeTraining = False
if os.path.exists(weightsDir):
	print ("Weights directory already exists. Resuming from the last training.")
	# exit (-1)
	weightFileList = os.listdir(weightsDir)
	if len(weightFileList) < len(y_entries):
		print ("Warning: Weights directory cannot be used. Removing previous directory.")
		shutil.rmtree(weightsDir)
		os.mkdir(weightsDir)
	else:
		resumeTraining = True
else:
	print ("Creating new weights directory")
	os.mkdir(weightsDir)

# Compute w_bar and write it to the directory
startingIteration = 0
if resumeTraining:
	weightFileList = os.listdir(weightsDir)
	lastWBarFileName = None	# Find the last w_bar file saved
	for fileName in weightFileList:
		if "w_bar" in fileName:
			if lastWBarFileName is None:
				lastWBarFileName = os.path.abspath(os.path.join(weightsDir, fileName))
			else:
				# Compare the iteration number
				wBarLastIterationNumber = int(lastWBarFileName[lastWBarFileName.rfind('_')+1:lastWBarFileName.rfind('.')])
				wBarCurrentIterationNumber = int(fileName[fileName.rfind('_')+1:fileName.rfind('.')])
				if wBarCurrentIterationNumber > wBarLastIterationNumber:
					lastWBarFileName = os.path.abspath(os.path.join(weightsDir, fileName))

			# Update the stating iteration number
			startingIteration = int(lastWBarFileName[lastWBarFileName.rfind('_')+1:lastWBarFileName.rfind('.')]) + 1

	print ("Loading w_bar from file: %s" % (lastWBarFileName))	
	# w_bar = numpy.load(lastWBarFileName)
	w_bar = scipy.sparse.load_npz(lastWBarFileName)
	print ("Training will resume from iteration # %d" % (startingIteration))
else:
	w_bar = scipy.sparse.csc_matrix(numpy.zeros((X.shape[1] + 1, 1)))

# Create class specific pickles
print ("Number of classes found in data file: %d" % y_entries.shape[0])
createClassInstanceFiles = False
if os.path.exists(pickleDir):
	list = os.listdir(pickleDir)
	numFiles = len(list)
	print ("Number of files found in directory: %d" % numFiles)
	if numFiles < y_entries.shape[0]:
		shutil.rmtree(pickleDir)
		createClassInstanceFiles = True
else:
	createClassInstanceFiles = True

if createClassInstanceFiles:
	print ("Creating directory for holding pickle data")
	os.mkdir(pickleDir)

	for i in range(y_entries.shape[0]):
		classX = X[y == y_entries[i]]
		classY = y[y == y_entries[i]]
		fileName = os.path.abspath(os.path.join(pickleDir, str(int(y_entries[i])) + '.txt'))
		dump_svmlight_file(classX, classY, f=fileName)
		print ("Class # %d | Class ID: %d | Data shape: %s | Label shape: %s | Output file: %s" % (i, y_entries[i], classX.shape, classY.shape, fileName))

numIterations = 100
for i in range(startingIteration, numIterations):
	try:
		startingTime = time.time()
		print ("Starting SVM solver for iteration # %d" % i)
		numProcesses = mp.cpu_count() / 2
		print ("Starting %d processes" % numProcesses)
		pool = mp.Pool(processes=numProcesses)
		results = pool.map(parallel, y_entries)
		# parallel(y_entries[0])
		print ("All subproblems solved for the current iteration")
		endTime = time.time()
		print ("Time elapsed: %s secs" % (endTime - startingTime))
		logFile.write("Iteration: %d | Time elapsed in computation: %s secs\n" % (i, str(endTime - startingTime)))
		pool.terminate()

		# w' is the combination of the w's found for each class
		# Read all of the saved numpy files to compute the optimal hyperplane
		w_bar = scipy.sparse.csc_matrix(numpy.zeros((X.shape[1] + 1, 1)))
		weightFileList = os.listdir(weightsDir)
		for weightVectorFileName in weightFileList:
			# weightVector = numpy.load(os.path.abspath(os.path.join(weightsDir, weightVectorFileName)))
			weightVector = scipy.sparse.load_npz(os.path.abspath(os.path.join(weightsDir, weightVectorFileName)))
			w_bar += weightVector
		w_bar /= len(weightFileList)
		fileName = os.path.abspath(os.path.join(weightsDir, 'w_bar_' + str(i) + '.npz'))
		scipy.sparse.save_npz(fileName, w_bar)
		# numpy.save(fileName, w_bar)

		endTime = time.time()
		print ("Total time elapsed (one iteration): %s secs" % (endTime - startingTime))
		logFile.write ("Iteration: %d | Total time elapsed (one iteration): %s secs\n" % (i, str(endTime - startingTime)))

	except KeyboardInterrupt:
		print ("Caught KeyboardInterrupt, terminating workers")
		pool.terminate()
		pool.join()

logFile.close()