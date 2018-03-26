import csv
import math
import random
import sys
import pandas
import numpy as np

#
# DecisionTree
#
# This class will build the decision tree using its method "learn"
#
class DecisionTree():
	# Create an empty tree
	tree = {}
	# This function will create the tree and set it to self.tree for reference
	def learn(self, training_set, attributes, target):
		# Build the tree and set it to self.tree
		self.tree = build_tree(training_set, attributes, target)


#
# Node
#
# This class will represent a node/vertex of the tree when trying to classify
#
class Node():
	# Set the value to be an empty string to set on initilization
	value = ""
	# Set a list of children to be filled up on initilization
	children = []
	# On initilization you need to set your value and children
	def __init__(self, val, dictionary):
		# Set your value to whatever you passed in as the parameter "val"
		self.value = val
		# Error check to make sure you passed in a dictionary in the parameter "dictionary"
		if (isinstance(dictionary, dict)):
			# Set your children to be the keys of the dictionary that was passed in 
			# as "dictionary"
			self.children = dictionary.keys()


#
# This function will return which class has the majority of entries in the given dataset "data"
#
def majorClass(attributes, data, target):
	# Create an empyt set to house the frequency of classes
	freq = {}
	# Get the index of the target attribute based upon the parameters
	index = attributes.index(target)
	# For each of the entries in the dataset
	for tuple in data:
		# If there already is a key of that particular entry
		if (freq.has_key(tuple[index])):
			# Increment its frequency 
			freq[tuple[index]] += 1 
		# If there is no entry of that key in the frequency set
		else:
			# Set the frequency of that key to 1
			freq[tuple[index]] = 1
	# Set a maximum to be 0 to be used later
	max = 0
	# Set the corresponding majority class to be empty for now
	major = ""
	# For each key in the frequency set
	for key in freq.keys():
		# If the key has been seen more times that the max
		if freq[key]>max:
			# Set the max to be that keys number of times it was seen
			max = freq[key]
			# Set the majority class to be the key
			major = key
	# Return the class who has the majority
	return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetAttr):

	freq = {}
	dataEntropy = 0.0

	i = 0
	for entry in attributes:
		if (targetAttr == entry):
			break
		i = i + 1

	i = i - 1

	for entry in data:
		if (freq.has_key(entry[i])):
			freq[entry[i]] += 1.0
		else:
			freq[entry[i]]  = 1.0

	for freq in freq.values():
		dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
		
	return dataEntropy


# Calculates the information gain (reduction in entropy) in the data when a particular attribute is chosen for splitting the data.
def info_gain(attributes, data, attr, targetAttr):

	freq = {}
	subsetEntropy = 0.0
	i = attributes.index(attr)

	for entry in data:
		if (freq.has_key(entry[i])):
			freq[entry[i]] += 1.0
		else:
			freq[entry[i]]  = 1.0

	for val in freq.keys():
		valProb        = freq[val] / sum(freq.values())
		dataSubset     = [entry for entry in data if entry[i] == val]
		subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

	return (entropy(attributes, data, targetAttr) - subsetEntropy)


# This function chooses the attribute among the remaining attributes which has the maximum information gain.
def attr_choose(data, attributes, target):

	best = attributes[0]
	maxGain = 0;

	for attr in attributes:
		newGain = info_gain(attributes, data, attr, target) 
		if newGain>maxGain:
			maxGain = newGain
			best = attr

	return best


# This function will get unique values for that particular attribute from the given data
def get_values(data, attributes, attr):

	index = attributes.index(attr)
	values = []

	for entry in data:
		if entry[index] not in values:
			values.append(entry[index])

	return values

# This function will get all the rows of the data where the chosen "best" attribute has a value "val"
def get_data(data, attributes, best, val):

	new_data = [[]]
	index = attributes.index(best)

	for entry in data:
		if (entry[index] == val):
			newEntry = []
			for i in range(0,len(entry)):
				if(i != index):
					newEntry.append(entry[i])
			new_data.append(newEntry)

	new_data.remove([])    
	return new_data


# This function is used to build the decision tree using the given data, attributes and the target attributes. It returns the decision tree in the end.
def build_tree(data, attributes, target):

	data = data[:]
	vals = [record[attributes.index(target)] for record in data]
	default = majorClass(attributes, data, target)

	if not data or (len(attributes) - 1) <= 0:
		return default
	elif vals.count(vals[0]) == len(vals):
		return vals[0]
	else:
		best = attr_choose(data, attributes, target)
		tree = {best:{}}
	
		for val in get_values(data, attributes, best):
			new_data = get_data(data, attributes, best, val)
			newAttr = attributes[:]
			newAttr.remove(best)
			subtree = build_tree(new_data, newAttr, target)
			tree[best][val] = subtree
	
	return tree

# global that will be the dataset to use
data_set = None
# global that will be the feature set of the data
feature_set = None
# global that will be the class set of the data
class_set = None


#
# This function will read in a csv which is located at the parameter passed to it
# 
def read_csv(filepath):
	# Get a reference to the global variable data_set
	global data_set
	# Read csv in using pandas and set it to the data_set global
	data_set = pandas.read_csv(filepath)
	# Randomize the data
	# print data_set.shape[0]
	# data_set = data_set.sample(n=data_set.shape[0])

#
# This function will split the global data_set into classes
#
def format_data_set():
		# Get the columns of the dataset
	cols = data_set.columns
	# Get a reference to the global feature_set
	global feature_set
	# Get all of the features by reading in all but the last column of the first row
	feature_set = np.asarray(cols.tolist()[:-1]).tolist()
	# Get a reference tp the global class_set
	global class_set
	# Get the entire last column of the dataset
	class_set = np.asarray(data_set.iloc[:,-1]).tolist()
	global data_set
	# Now take all of the columns but the last
	data_set = np.asarray(data_set.iloc[:,:]).tolist()

# This function runs the decision tree algorithm. It parses the file for the data-set, and then it runs the 10-fold cross-validation. It also classifies a test-instance and later compute the average accurracy
# Improvements Used: 
# 1. Discrete Splitting for attributes "age" and "fnlwght"
# 2. Random-ness: Random Shuffle of the data before Cross-Validation
def run_decision_tree():
    data = []

    with open("image.csv") as tsv:
        for line in csv.reader(tsv, delimiter=","):
            data.append(tuple(line))

	print "Number of records: %d" % len(data)

	attributes = ["REGION-CENTROID-COL","REGION-CENTROID-ROW","REGION-PIXEL-COUNT","SHORT-LINE-DENSITY-5",\
	"SHORT-LINE-DENSITY-2","VEDGE-MEAN","VEDGE-SD","HEDGE-MEAN","HEDGE-SD","INTENSITY-MEAN","RAWRED-MEAN",\
	"RAWBLUE-MEAN","RAWGREEN-MEAN","EXRED-MEAN","EXBLUE-MEAN","EXGREEN-MEAN","VALUE-MEAN","SATURATION-MEAN",\
	"HUE-MEAN","classes"]
	target = attributes[-1]

	K = 10
	acc = []
	for k in range(K):
		random.shuffle(data)
		training_set = [x for i, x in enumerate(data) if i % K != k]
		test_set = [x for i, x in enumerate(data) if i % K == k]
		tree = DecisionTree()
		tree.learn( training_set, attributes, target )
		results = []

		for entry in test_set:
			tempDict = tree.tree.copy()
			result = ""
			while(isinstance(tempDict, dict)):
				root = Node(tempDict.keys()[0], tempDict[tempDict.keys()[0]])
				tempDict = tempDict[tempDict.keys()[0]]
				index = attributes.index(root.value)
				value = entry[index]
				if(value in tempDict.keys()):
					child = Node(value, tempDict[value])
					result = tempDict[value]
					tempDict = tempDict[value]
				else:
					result = "Null"
					break
			if result != "Null":
				results.append(result == entry[-1])

		accuracy = float(results.count(True))/float(len(results))
		acc.append(accuracy)

	avg_acc = sum(acc)/len(acc)
	print "Average accuracy: %.4f" % avg_acc

	# Writing results to a file (DO NOT CHANGE)
	f = open("result.txt", "w")
	f.write("accuracy: %.4f" % avg_acc)
	f.close()

if __name__ == "__main__":
	run_decision_tree()