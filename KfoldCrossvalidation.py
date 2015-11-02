import numpy as np
import math as math
import random as random
import logging


class PredictableModel(object):
    def __init__(self, feature, classifier):
        """
        if not isinstance(feature, AbstractFeature):
            raise TypeError("feature must be of type AbstractFeature!")
        if not isinstance(classifier, AbstractClassifier):
            raise TypeError("classifier must be of type AbstractClassifier!")
        """
        self.feature = feature
        self.classifier = classifier
    
    def compute(self, X, y):
        features = self.feature.compute(X,y)
        return self.classifier.compute(features,y)

    def predict(self, X):
        q = self.feature.extract(X)
        return self.classifier.predict(q)
        
    def __repr__(self):
        feature_repr = repr(self.feature)
        classifier_repr = repr(self.classifier)
        return "PredictableModel (feature=%s, classifier=%s)" % (feature_repr, classifier_repr)

class AbstractClassifier(object):

    def compute(self,X,y):
        raise NotImplementedError("Every AbstractClassifier must implement the compute method.")
    
    def predict(self,X):
        raise NotImplementedError("Every AbstractClassifier must implement the predict method.")

    def update(self,X,y):
        raise NotImplementedError("This Classifier is cannot be updated.")

# TODO The evaluation of a model should be completely moved to the generic ValidationStrategy. The specific Validation 
#       implementations should only care about partition the data, which would make a lot sense. Currently it is not 
#       possible to calculate the true_negatives and false_negatives with the way the predicitions are generated and 
#       data is prepared.
#       
#     The mentioned problem makes a change in the PredictionResult necessary, which basically means refactoring the 
#       entire framework. The refactoring is planned, but I can't give any details as time of writing.
#
#     Please be careful, when referring to the Performance Metrics at the moment, only the Precision is implemented,
#       and the rest has no use at the moment. Due to the missing refactoring discussed above.
#

def shuffle(X, y):

    idx = np.argsort([random.random() for i in xrange(len(y))])
    y = np.asarray(y)
    X = [X[i] for i in idx]
    y = y[idx]
    return (X, y)
    
def slice_2d(X,rows,cols):
    return [X[i][j] for j in cols for i in rows]

def precision(true_positives, false_positives):

    return accuracy(true_positives, 0, false_positives, 0)
    
def accuracy(true_positives, true_negatives, false_positives, false_negatives, description=None):

    true_positives = float(true_positives)
    true_negatives = float(true_negatives)
    false_positives = float(false_positives)
    false_negatives = float(false_negatives)
    if (true_positives + true_negatives + false_positives + false_negatives) < 1e-15:
       return 0.0
    return (true_positives+true_negatives)/(true_positives+false_positives+true_negatives+false_negatives)

class ValidationResult(object):
    """Holds a validation result.
    """
    def __init__(self, true_positives, true_negatives, false_positives, false_negatives, description):
        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.description = description
        
    def __repr__(self):
        res_precision = precision(self.true_positives, self.false_positives) * 100
        res_accuracy = accuracy(self.true_positives, self.true_negatives, self.false_positives, self.false_negatives) * 100
        return "ValidationResult (Description=%s, Precision=%.2f%%, Accuracy=%.2f%%)" % (self.description, res_precision, res_accuracy)
    
class ValidationStrategy(object):
    """ Represents a generic Validation kernel for all Validation strategies.
    """
    def __init__(self, model):
        """    
        Initialize validation with empty results.
        
        Args:
        
            model [PredictableModel] The model, which is going to be validated.
        """
        if not isinstance(model,PredictableModel):
            raise TypeError("Validation can only validate the type PredictableModel.")
        self.model = model
        self.validation_results = []
    
    def add(self, validation_result):
        self.validation_results.append(validation_result)
        
    def validate(self, X, y, description):
        """
        
        Args:
            X [list] Input Images
            y [y] Class Labels
            description [string] experiment description
        
        """
        raise NotImplementedError("Every Validation module must implement the validate method!")
        
    
    def print_results(self):
        print self.model
        for validation_result in self.validation_results:
            print validation_result

    def __repr__(self):
        return "Validation Kernel (model=%s)" % (self.model)
        
class KFoldCrossValidation(ValidationStrategy):
    """ 
    
    Divides the Data into 10 equally spaced and non-overlapping folds for training and testing.
    
    Here is a 3-fold cross validation example for 9 observations and 3 classes, so each observation is given by its index [c_i][o_i]:
                
        o0 o1 o2        o0 o1 o2        o0 o1 o2  
    c0 | A  B  B |  c0 | B  A  B |  c0 | B  B  A |
    c1 | A  B  B |  c1 | B  A  B |  c1 | B  B  A |
    c2 | A  B  B |  c2 | B  A  B |  c2 | B  B  A |
    
    Please note: If there are less than k observations in a class, k is set to the minimum of observations available through all classes.
    """
    def __init__(self, model, k=10):
        """
        Args:
            k [int] number of folds in this k-fold cross-validation (default 10)
        """
        super(KFoldCrossValidation, self).__init__(model=model)
        self.k = k
        self.logger = logging.getLogger("facerec.validation.KFoldCrossValidation")

    def validate(self, X, y, description="ExperimentName"):
        """ Performs a k-fold cross validation
        
        Args:

            X [dim x num_data] input data to validate on
            y [1 x num_data] classes
        """
        X,y = shuffle(X,y)
        c = len(np.unique(y))
        foldIndices = []
        n = np.iinfo(np.int).max
        for i in range(0,c):
            idx = np.where(y==i)[0]
            n = min(n, idx.shape[0])
            foldIndices.append(idx.tolist()); 

        # I assume all folds to be of equal length, so the minimum
        # number of samples in a class is responsible for the number
        # of folds. This is probably not desired. Please adjust for
        # your use case.
        if n < self.k:
            self.k = n

        foldSize = int(math.floor(n/self.k))
        
        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        for i in range(0,self.k):
        
            self.logger.info("Processing fold %d/%d." % (i+1, self.k))
                
            # calculate indices
            l = int(i*foldSize)
            h = int((i+1)*foldSize)
            testIdx = slice_2d(foldIndices, cols=range(l,h), rows=range(0, c))
            trainIdx = slice_2d(foldIndices,cols=range(0,l), rows=range(0,c))
            trainIdx.extend(slice_2d(foldIndices,cols=range(h,n),rows=range(0,c)))
            
            # build training data subset
            Xtrain = [X[t] for t in trainIdx]
            ytrain = y[trainIdx]
                        
            self.model.compute(Xtrain, ytrain)
            
            # TODO I have to add the true_negatives and false_negatives. Models also need to support it,
            # so we should use a PredictionResult, instead of trying to do this by simply comparing
            # the predicted and actual class.
            #
            # This is inteneded of the next version! Feel free to contribute.
            for j in testIdx:
                prediction = self.model.predict(X[j])[0]
                if prediction == y[j]:
                    true_positives = true_positives + 1
                else:
                    false_positives = false_positives + 1
                    
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))
        self.print_results()
    def __repr__(self):
        return "k-Fold Cross Validation (model=%s, k=%s)" % (self.model, self.k)
