from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus

iris = load_iris()
iris_X = iris.data
iris_Y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_Y, test_size=0.3)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
