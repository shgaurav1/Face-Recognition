import logging
import numpy as np
import operator as op
from sklearn.ensemble import RandomForestClassifier


class AbstractDistance(object):
    def __init__(self, name):
        self._name = name
        
    def __call__(self,p,q):
        raise NotImplementedError("Every AbstractDistance must implement the __call__ method.")
        
    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name
        
class EuclideanDistance(AbstractDistance):
    def __init__(self):
        AbstractDistance.__init__(self,"EuclideanDistance")

    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p-q),2)))


def asRowMatrix(X):
    """
    Creates a row-matrix from multi-dimensional data items in list l.
    
    X [list] List with multi-dimensional data.
    """
    if len(X) == 0:
        return np.array([])
    total = 1
    for i in range(0, np.ndim(X[0])):
        total = total * X[0].shape[i]
    mat = np.empty([0, total], dtype=X[0].dtype)
    for row in X:
        mat = np.append(mat, row.reshape(1,-1), axis=0) # same as vstack
    return np.asmatrix(mat)


class AbstractClassifier(object):

    def compute(self,X,y):
        raise NotImplementedError("Every AbstractClassifier must implement the compute method.")
    
    def predict(self,X):
        raise NotImplementedError("Every AbstractClassifier must implement the predict method.")

    def update(self,X,y):
        raise NotImplementedError("This Classifier is cannot be updated.")

class NearestNeighbor(AbstractClassifier):
    """
    Implements a k-Nearest Neighbor Model with a generic distance metric.
    """
    def __init__(self, dist_metric=EuclideanDistance(), k=1):
        AbstractClassifier.__init__(self)
        self.k = k
        self.dist_metric = dist_metric
        self.X = []
        self.y = np.array([], dtype=np.int32)

    def update(self, X, y):
        """
        Updates the classifier.
        """
        self.X.append(X)
        self.y = np.append(self.y, y)

    def compute(self, X, y):
        self.X = X
        self.y = np.asarray(y)
    
    def predict(self, q):

        distances = []
        for xi in self.X:
            xi = xi.reshape(-1,1)
            d = self.dist_metric(xi, q)
            distances.append(d)
        if len(distances) > len(self.y):
            raise Exception("More distances than classes. Is your distance metric correct?")
        distances = np.asarray(distances)
        idx = np.argsort(distances)
        sorted_y = self.y[idx]
        sorted_distances = distances[idx]
        sorted_y = sorted_y[0:self.k]
        sorted_distances = sorted_distances[0:self.k]
        hist = dict((key,val) for key, val in enumerate(np.bincount(sorted_y)) if val)
        predicted_label = max(hist.iteritems(), key=op.itemgetter(1))[0]
 
        return [predicted_label, { 'labels' : sorted_y, 'distances' : sorted_distances }]
        
    def __repr__(self):
        return "NearestNeighbor (k=%s, dist_metric=%s)" % (self.k, repr(self.dist_metric))



class RandomForest(AbstractClassifier):
    """
    Implements a Random forest Model.
    """
    def __init__(self, k=50):
        AbstractClassifier.__init__(self)
        self.k = k
        self.X = []
        self.y = np.array([], dtype=np.int32)
        self.clf = None

    def update(self, X, y):
        """
        Updates the classifier.
        """
        self.X.append(X)
        self.y = np.append(self.y, y)

    def compute(self, X, y):
        self.X = np.reshape(np.array(X),(np.array(X).shape[0],np.array(X).shape[1]))
        self.y = y
        self.clf = RandomForestClassifier(n_estimators=50)
        self.clf.fit(self.X,self.y)
    
    def predict(self, q):
        #clf = RandomForestClassifier(n_estimators=50)
        #clf.fit(X,y)
        q = np.array(q)
        y_pred = self.clf.predict(q.transpose())
        return [y_pred]
