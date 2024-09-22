from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Model:

    """
    Class that contains a KNN model to classify iris dataset from sklearn.

    Methods:
        __init__ - initializes the model by training it on the iris dataset.
        __call__ - takes a list of data records and returns the predicted labels.
        _init_data - loads the iris dataset, defines the features and labels, and splits the data into training and evaluation sets.
        _score - calculates the accuracy score of the model on the evaluation data.
        _train - trains the KNeighborsClassifier model on the training data.
    """

    def __init__(self, test_size=0.5):
        self.dataset = "iris"
        self.architecture = "KNN"
        self._train(test_size)

    def __call__(self, data):
        for record in data:
            if len(record) != len(self.labels) and not all([isinstance(val, float) or isinstance(val, int) for val in record]):
                raise ValueError(f"Malformed data record {record}")

        yield from (self.labels[label] for label in self.model.predict(data))

    def _init_data(self, test_size=0.5):
        iris_dataset = load_iris()
        self.features = iris_dataset.feature_names
        self.labels = iris_dataset.target_names
        x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.5)
        self._train_data = (x_train, y_train)
        self._eval_data = (x_test, y_test)

    def _score(self):
        preds = self.model.predict(self._eval_data[0])
        return accuracy_score(preds, self._eval_data[1])

    def _train(self, test_size=0.5):
        self._init_data()
        classifier = KNeighborsClassifier()
        classifier.fit(*self._train_data)
        self.model = classifier
        self.eval = self._score()