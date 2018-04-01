import csv
import math
import random
import sys
import pandas
import numpy as np
import copy
import decision_tree
import tree_node
# global that will be the dataset to use
data_set = None
# global that will be the feature set of the data
feature_set = None
# global that will be the class set of the data
class_set = None

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
# This function will recurse down the tree and once it reaches a leaf node determine
# if it got the correct answer
#
def recurse_tree(tree, test_entry, attributes):
	# If you have no children you are a leaf
	if not tree.children:
		# If the class of the leaf matches the test
		if test_entry[-1] == tree.default:
			# Return a 1 which is true
			return 1
		# Other wise you are incorrect
		else:
			# Return a 0 which is incorrect
			return 0
	# Get the index of the attribute you are about to recurse on
	index = attributes.index(tree.attr)
	# Get the value of the attribute of the test entry
	value = test_entry[index]
	# Set a counter
	count = 0
	# For each value in the tress values
	for val in tree.values:
		# If you found the corresponding value to the test entry
		if value == val:
			# Recurse down
			return recurse_tree(tree.children[count], test_entry, attributes)
		# Increment counter
		count += 1


#
# This function will determine if you can still prune the tree
#
def can_be_pruned(tree):
	# Set a value to be returned
	ret = False
	# If the tree you are on has a 1 for prune
	if tree.prune == 1:
		# Set the return value to be true
		ret =  True
	# If you have children
	if tree.children != None:
		# For each subtree in tree
		for subtree in tree.children:
			# Recurse down and see if they can be pruned, if so
			if can_be_pruned(subtree) == True:
				# Set return value to be true
				ret = True
	# Return the value
	return ret

#
# This function will prune the tree, it will only prune one node
#
def prune_tree(tree):
	# If you you found a tree is marked to be pruned
	if tree.prune == 1:
		# Set it to not be pruned again
		tree.prune = 0
		# Remove all of your children
		tree.children = None
		# Return the tree and a 1 to say you pruned already
		return (tree, 1)
	# Set a vairbale to check if you are done pruning
	done = 0
	# If you have children
	if tree.children != None:
		# Set a counter
		count = 0
		# For each of the trees subtrees
		for subtree in tree.children:
			# Recurse down and get the values
			ret = prune_tree(subtree)
			# Set your subtree to get the one returned
			tree.children[count] = ret[0]
			# If you got a 1 saying you are done pruning
			if ret[1] == 1:
				# Set done to 1
				done = 1
				# Break out of the loop to stop recursing
				break
			# Increment counter by 1
			count += 1
	# Return yourself and the done variabel
	return (tree, done)


#
# This function will mark the node of a tree to be pruned next, to 0
# this is used for when you need to just say that you do not need to prune that node
#
def mark_node(tree):
	# If you found the tree that needs to be pruned
	if tree.prune == 1:
		# Set prune of the tree to be 0
		tree.prune = 0
		# Return yourself and 1 to say that you are done
		return (tree, 1)
	# Set a variable that will determine if you are done looking for the tree to mark
	done = 0
	# If you have children
	if tree.children != None:
		# Set a counter
		count = 0
		# For each subtree of the tree
		for subtree in tree.children:
			# Recurse down and get the variables
			ret = prune_tree(subtree)
			# Set your child to the one returnd 
			tree.children[count] = ret[0]
			# If you got back a 1 to say you are done looking
			if ret[1] == 1:
				# Set the done variable to be 1
				done = 1
				# Break out of the loop
				break
			# Increment counter by 1
			count += 1
	# Return yourself and the done variable
	return (tree, done)

#
# This function will give back a fold in the cross fold validation
#
def cross_fold_sets(data, k, K):
	# Randomize the data
	random.shuffle(data)
	# Get the training set of the fold
	training_set = [x for i, x in enumerate(data) if i % K != k]
	# Get the validation set of the fold
	validation_set = [x for i, x in enumerate(data) if i % K == k]
	# Return the sets
	return (training_set, validation_set)
#
# This function will run a decisionon tree on the data set and attributes passed in.
# You can also set if you want to run reduced error pruning
def run_decision_tree(data, attributes, prune):
	# Get all of the classes for the dataset
	target = attributes[-1]
	# Cross fold validation K
	K = 5
	# Set a vairbale for the overall accuracy
	acc = []
	# For each fold of the cross fold validation
	for k in range(K):
		ret = cross_fold_sets(data, k, K)
		# Get the training set of the fold
		training_set = ret[0]
		# Get the validation set of the fold
		validation_set = ret[1]
		# Create a tree
		tree = decision_tree.DecisionTree()
		# Grow the tree to completion based on the training set
		tree.learn( training_set, attributes, target )
		# If you do not want to do error reduced pruning
		if (prune == str(0)):
			# Set a list to house the results of the test
			results = []
			# For each test entry in the validation set
			for test_entry in validation_set:
				# Get the answer right or wrong
				answer = recurse_tree(tree.tree, test_entry, attributes)
				# If you got an answer
				if answer != None:
					# Add it to the results list
					results.append(answer)
			# Get the accuracy of the test
			accuracy = float(results.count(1))/float(len(results))
			# Add the accuracy of this fold test to the acc list
			acc.append(accuracy)
		else:
			# Save a copy of the tree off
			saved = copy.deepcopy(tree.tree)
			# See if the tree can be pruned
			can_prune = can_be_pruned(saved)
			# Set a variable to be the best accuracy you found
			best_accuracy = 0.0
			# While you are able to prune the tree
			while(can_prune):
				# Save a copy of the tree so far
				temp = copy.deepcopy(saved)
				# Set a variable to keep track of the results
				results = []
				# For each test entry in the validation set
				for test_entry in validation_set:
					# See if you can classify the test entry 
					answer = recurse_tree(temp, test_entry, attributes)
					# If you got an answer right or wrong
					if answer != None:
						# Append it to the results list
						results.append(answer)
				# Calculate the accruacy of the tree against the validation set
				old_accuracy = float(results.count(1))/float(len(results))
				# Prune the tree
				prune_tree(temp)
				# Set a new results list
				new_results = []
				# For each test entry in the validation set
				for test_entry in validation_set:
					# See if you can classify the test entry
					answer = recurse_tree(temp, test_entry, attributes)
					# If you got an anser right or wrong
					if answer != None:
						# Add the answer to the results list
						new_results.append(answer)
				# Calculate the accuracy of the pruned tree
				new_accuracy = float(new_results.count(1))/float(len(new_results))

				# print str(old_accuracy) + " VS " + str(new_accuracy)

				# Set a variable to be the best accuracy you found so far
				best_accuracy_so_far = 0.0
				# If you increased accuracy pruning the tree
				if old_accuracy < new_accuracy:
					# The saved copy of the tree is now the pruned tree
					saved = copy.deepcopy(temp)
					# Set the best accuracy you found
					best_accuracy_so_far = new_accuracy
				# If you did not increase the accuracy by pruning
				else:
					# Mark the node to not be pruned again
					mark_node(saved)
					# Set the best accuracy you found
					best_accuracy_so_far = old_accuracy
				if best_accuracy_so_far > best_accuracy:
					best_accuracy = best_accuracy_so_far
				# See if you can prune the tree still
				can_prune = can_be_pruned(saved)
			acc.append(best_accuracy)
	# Get the average accuracy
	avg_acc = sum(acc)/len(acc)
	# Print what you found
	print "Average accuracy: %.4f" % avg_acc

if __name__ == "__main__":
	# Read in the file passed in by the command line when script started
	info = read_csv(sys.argv[1])
	# Get the argument for reduced error pruning
	prune = sys.argv[2]
        if prune == str(1):
            print "Reduced Error Pruning: enabled"
	# Run the decision tree
	run_decision_tree(info[1], info[0], prune)
