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
	# Set a defualt class for when you prune the tree
	default = ""
	# On initilization you need to set your value and children
	def __init__(self, attr, val, tree, default, prune):
		# Set your value to whatever you passed in as the parameter "val"
		self.attr = attr
		self.values = val
		self.children = tree
		self.default = default
		self.prune = prune