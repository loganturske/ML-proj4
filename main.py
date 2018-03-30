import csv
import math
import random
import sys
import pandas
import numpy as np

# global that will be the dataset to use
data_set = None
# global that will be the feature set of the data
feature_set = None
# global that will be the class set of the data
class_set = None


#
# DecisionTree
#
# This class will build the decision tree using its method "learn"
#
class DecisionTree():
	# Create an empty tree
	tree = None
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
	# Set the attribute that the node corresponds to to empty
	attr = ""
	# Set the value to be an empty string to set on initilization
	values = ""
	# Set a list of children to be filled up on initilization
	children = None
	# Set a value to know if you are marking this node for pruning
	prune = 0
	# On initilization you need to set your value and children
	def __init__(self, attr, val, tree, prune):
		# Set your value to whatever you passed in as the parameter "val"
		self.attr = attr
		self.values = val
		self.children = tree
		self.prune = prune
		# Error check to make sure you passed in a dictionary in the parameter "dictionary"
		# if (isinstance(dictionary, dict)):
		# 	# Set your children to be the keys of the dictionary that was passed in 
		# 	# as "dictionary"
		# 	self.children = dictionary.keys()

#
# This function will read in a csv which is located at the parameter passed to it
# and return the things read
# 
def read_csv(filepath):
	# Make a reference to the data set
	data = []
	# Make a reference to the attributes
	attributes = []
	# Make a counter variable
	counter = 0
	# Open the file at filepath
	with open(filepath) as tsv:
		# For each line in the file separated by commas
		for line in csv.reader(tsv, delimiter=","):
			# If you are on the first line
			if counter == 0:
				# Set the attributes variable
				attributes = line
				# Make the counter 1
				counter = 1
			# Append to data set
			else:
				data.append(tuple(line))
	# Return a tuple with the information we read
	return (attributes, data)

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

#
# This function calculates the entropy of the data given the target attribute
#
def entropy(attributes, data, targetAttr):
	# Create and empty set that will be the frequency of the attributes
	freq = {}
	# Set the entropy to be 0 to be used later
	dataEntropy = 0.0
	# Set a counter
	i = 0
	# For each entry in the attributes
	for entry in attributes:
		# If you found the target entry
		if (targetAttr == entry):
			# Break out of the loop
			break
		# Increment the counter
		i = i + 1
	# Subtract 1 from the counter so that you get the appropriate amount
	i = i - 1
	# For each entry in the data that was passed in
	for entry in data:
		# If there exists a key for that entry
		if (freq.has_key(entry[i])):
			# Increment its frequency by 1
			freq[entry[i]] += 1.0
		# If you have not seen this entry yet
		else:
			# Set its frequency to 1
			freq[entry[i]]  = 1.0
	# For each of the frequencies in the freq list
	for freq in freq.values():
		# Calculate its entropy given the entropy formula
		dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
	# Return the complete calculated entropy
	return dataEntropy

#
# This function calculates the information gain (reduction in entropy) 
# in the data when a particular attribute is chosen for splitting the data.
#
def info_gain(attributes, data, attr, targetAttr):
	# Create an empty set for setting the frequency of entries
	freq = {}
	# Set a entropy variable to 0 to be used later
	subsetEntropy = 0.0
	# Get the index of the particular attribute passed in as "attr"
	i = attributes.index(attr)
	# For each entry in the data passed in
	for entry in data:
		# If there exists a key for that entry
		if (freq.has_key(entry[i])):
			# Increment its frequency by 1
			freq[entry[i]] += 1.0
		# If you have not seen this entry yet
		else:
			# Set its frequency to 1
			freq[entry[i]]  = 1.0
	# For each of the entries in the frequncy set
	for val in freq.keys():
		# Calculate the probability of that entry
		valProb        = freq[val] / sum(freq.values())
		# Get the data of that particual entry
		dataSubset     = [entry for entry in data if entry[i] == val]
		# Calculate the entropy for that particular entry and mulitply it by the probability
		subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)
	# Return the information gain by calculating the entropy and subtracting the subset entropy you found
	return (entropy(attributes, data, targetAttr) - subsetEntropy)

#
# This function chooses the attribute among the remaining attributes which has the maximum information gain.
#
def attr_choose(data, attributes, target):
	# Set an arbitrary attribute to be compared against later
	best = attributes[0]
	# Set the maximum info gain to be 0
	maxGain = 0;
	# For each attribute
	for attr in attributes:
		# Calculate the information gain of that attribute
		newGain = info_gain(attributes, data, attr, target) 
		# If the info gain for the attribute is better than the maximum
		if newGain>maxGain:
			# Set the new maximum info gain
			maxGain = newGain
			# Set the best attribute to be the one you are currently on
			best = attr
	# Return the bet attribute you found
	return best

#
# This function will get unique values for the particular attribute from the given data
#
def get_values(data, attributes, attr):
	# Get the index of the attribute passed in
	index = attributes.index(attr)
	# Set an empty list to be filled in later
	values = []
	# For each entry in the data passed in
	for entry in data:
		# If there is a entry for the particular attribute that is not in the values list
		if entry[index] not in values:
			# Append that entry attribute value to the values list
			values.append(entry[index])
	# Return the list of values for the particular attribute
	return values

#
# This function will get all the rows of the data where the chosen "best" attribute has a value "val"
#
def get_data(data, attributes, best, val):
	# Set a 2 dimensional list to be empyt to be filled in later
	new_data = [[]]
	# Get the index of the attribute that you have chose passed in as "best"
	index = attributes.index(best)
	# For each entry in the data set
	for entry in data:
		# If the entry has the attribute as the value "val", which was passed to the function
		if (entry[index] == val):	
			# Create a new empty list
			newEntry = []
			# For each item in entry
			for i in range(0,len(entry)):
				# If the current count you are on is not the index
				if(i != index):
					# Append the particular entry attribute to the newEntry list
					newEntry.append(entry[i])
			# Append the newEntry list to the new_data list to be returned
			new_data.append(newEntry)
	# Remove any empty lists
	new_data.remove([])
	# Return the new_data list which are the rows with the best attribute with value "val"
	return new_data

#
# This function is used to build the decision tree using the given data, attributes and the target attributes.
# It returns the decision tree in the end.
#
def build_tree(data, attributes, target):
	# Create a copy of the data set
	data = data[:]
	# Get the values for each row of the given target
	vals = [record[attributes.index(target)] for record in data]
	# Get the majority of classes based on the given target
	default = majorClass(attributes, data, target)
	# If you ran out of data
	if not data or (len(attributes) - 1) <= 0:
		# Return the default class you found
		tree = Node("class", default, None, 0)
		return tree
	# If you have all of the same class
	elif vals.count(vals[0]) == len(vals):
		# Return that class
		tree = Node("class", vals[0], None, 0)
		return tree
	# Otherwise
	else:
		# Get the best attribute based on the target
		best = attr_choose(data, attributes, target)
		# Create an empty tree
		tree = []
		# Create an empty values set
		values = []
		# For each value in the rest of the data set
		for val in get_values(data, attributes, best):
			# Get the of the rows based upon the best attribute and the corresponding value
			new_data = get_data(data, attributes, best, val)
			# Make a copy of the attributes
			newAttr = attributes[:]
			# Remove the "best" attribute, so you don't choose it again
			newAttr.remove(best)
			# Recurse down based on the new data and attributes you just did
			subtree = build_tree(new_data, newAttr, target)
			# Set the sub tree to be the one you went and recursed down to
			# tree[best][val] = subtree
			tree.append(subtree)
			values.append(val)

	# Return the tree
	return Node(best, values, tree, 1)

def recurse_tree(tree, test_entry, attributes):

	if tree.children == None:
		if test_entry[-1] == tree.values:
			return 1
		else:
			return 0
	index = attributes.index(tree.attr)
	value = test_entry[index]
	i = 0
	for val in tree.values:
		if value == val:
			return recurse_tree(tree.children[i], test_entry, attributes)
		i += 1
# This function runs the decision tree algorithm. It parses the file for the data-set, and then it runs the 10-fold cross-validation. It also classifies a test-instance and later compute the average accurracy
# Improvements Used: 
# 1. Discrete Splitting for attributes "age" and "fnlwght"
# 2. Random-ness: Random Shuffle of the data before Cross-Validation
def run_decision_tree(data, attributes, prune):

	target = attributes[-1]

	K = 5
	acc = []
	for k in range(K):
		random.shuffle(data)
		training_set = [x for i, x in enumerate(data) if i % K != k]
		validation_set = [x for i, x in enumerate(data) if i % K == k]
		tree = DecisionTree()
		# Grow the tree
		tree.learn( training_set, attributes, target )

		results = []
		for test_entry in validation_set:
			answer = recurse_tree(tree.tree, test_entry, attributes)
			if answer != None:
				results.append(answer)
		accuracy = float(results.count(1))/float(len(results))
		print accuracy

	# 	acc.append(accuracy)

	# avg_acc = sum(acc)/len(acc)
	# print "Average accuracy: %.4f" % avg_acc

if __name__ == "__main__":
	# Read in the file passed in by the command line when script started
	info = read_csv(sys.argv[1])
	prune = sys.argv[2]
	run_decision_tree(info[1], info[0], prune)