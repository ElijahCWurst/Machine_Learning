import sys
import pandas as pd
from sklearn import tree
import graphviz

# Read the input file from command line argument
input_file = sys.argv[1]
data = pd.read_csv(input_file)

# Separate the features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Create the decision tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Visualize the decision tree
dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.view()