class Node:
	def __init__(self, splitAttribute=None, attributeValue=None, result=None, children=None, childrenCount=None, isLeaf=False):
		self.splitAttribute = splitAttribute # Feature to split on
		self.attributeValue = attributeValue	 # Value of the feature to split on
		self.result = result # Stores class labels if node is a leaf node
		self.children = children
		self.childrenCount = childrenCount
		self.isLeaf = isLeaf
