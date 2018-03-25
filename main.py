from math import log
import operator


#
# This function will subtract the first parameter from the second parameter
#
def subtract_points(a,b):
	# Subtract the points
	return a - b

#
# This function will square the parameter
#
def square_it(a):
	# Square parameter by multiplying it by itself
	return a * a

#
# This function will divide the first parameter by the second parameter
#
def divide_a_by_b(a,b):
	# Check to make sure that the divisor is not zero
	if b == 0:
		# If the divisor is 0 return 0
		return 0
	# Divid a by b
	return float(a)/float(b)

#
# Return the greater of the two parameters that were passed in
# If they are equal, the second parameter will be returned
#
def get_max_of(a, b):
	# If a is larger than b 
	if a > b:
		# Return a
		return a
	# Otherwise return b
	return b

#
# This function will take the square root of the parameter passed to it
#
def take_square_root(a):
	# Return the square root of a
	return np.sqrt(a)

#
# This function will get the average value of the array that was passed in
#
def get_average(arr):
	# Set a running total to zero
	total = 0
	# Set a running count to zero
	count = 0
	# For each element in the array
	for ele in arr:
		# Add the element to the running total
		total += ele
		# Increment the count by 1
		count += 1
	# Return the total divided by the count
	return divide_a_by_b(float(total),float(count))

#
# This function will get the entropy of the data passed to it
#
def get_entropy(data):
	# Get the number of entries of the data
	num_of_entries = len(data)
	# Create an empty set to house all of the labels
	labels = {}
	# For each feature in the data passed in
	for feature in data:
		# Get the label of the particular feature
		label = feature[-1]
		# If the label for this feature is not in the labels set
		if label not in labels.keys():
			# Enter an entry for the label in the label set (set it to 0)
			labels[label] = 0
		# Add one to the entry for the label in the label set
		labels[label] += 1
	# Set entropy to 0
	entropy = 0.0
	# For each key in the label set
	for key in labels:
		# Get the probability by getting the number of that label divided by
		# the maximum number of entries
		probability = divide_a_by_b(float(labels[key]), num_of_entries)
		# Take the log base 2 of the probability and multiply it by the probability
		# then subtract from the entropy you have calculated so far
		entropy -= probability * log(probability,2)
	# Return the entropy that you found
	return entropy

#
# This function will split the data passed in on a given feauture
#
def split_data(data, axis, val):
	# Make a empty list
	new_data = []
	# For each feature in the dataset passed in
	for feature in data:
		# If that feature on the particular axis has that value
		if feature[axis] == val:
			# Create a reduced feature set by removing that particular feater
			reducedFeat = feature[:axis]
			reducedFeat.extend(feature[axis+1:])
			# Put the new feature set on the new data list
			newData.append(reducedFeat)
	# Return the new data set that has the feature  column removed
	return newData

#
# This function will choose the best feature of the data that was passed in
#
def choose_feature(data):
	# Get the number of features
	features = len(data[0]) - 1
	# Get the entropy of the data that was passed in
	baseEntropy = get_entropy(data)
	# Set the best gain so far to be 0
	bestInfoGain = 0.0;
	# Set the best feature to be -1
	bestFeat = -1
	# For all of the features
	for i in range(features):
		# Get a list of the corresponding features from the dataset
		featList = [row[i] for row in data]
		# Get an unordered set of all unique values for the corresponding feature
		uniqueVals = set(featList)
		# Set a new entropy to be 0
		newEntropy = 0.0
		# For each value in the set of unique values
		for value in uniqueVals:
			# Split the data on the value
			newData = split_data(data, i, value)
			# Get the probablility of the new data by dividing by the total
			# number of data rows passed in
			probability = len(newData)/float(len(data))
			# The new entropy is the probability multiplied by the entropy of the data
			newEntropy += probability * get_entropy(newData)
		# Take the base entropy and subtract the new entropy that you just got
		infoGain = baseEntropy - newEntropy
		# If the information gain is better than the best information gain so far
		if (infoGain > bestInfoGain):
			# Set a new best information gain so far
			bestInfoGain = infoGain
			# Set the best feature so far
			bestFeat = i
	# Return the best feature that you found
	return bestFeat

def majority(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys(): classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def tree(data,labels):
	# For every row in the data take the last column which is the class
	class_list = [row[-1] for row in data]
	# If you have all the same classes
	if class_list.count(class_list[0]) == len(class_list):
		# Just return that class
		return classList[0]
	# If there is only classes
	if len(data[0]) == 1:
		# Return the majority of the classes
		return majority(classList)
	# Choose the best feature of the data that you are on
	bestFeat = choose_feature(data)
	# Get the label of the best feature you just chose
	bestFeatLabel = labels[bestFeat]
	# Start the tree with the best features label you just chose
	theTree = {bestFeatLabel:{}}
	# Remove that feature from the label list so it does not get chose further down the tree
	del(labels[bestFeat])
	# 
	featValues = [ex[bestFeat] for ex in data]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		theTree[bestFeatLabel][value] = tree(split_data(data, bestFeat, value),subLabels)
	return theTree