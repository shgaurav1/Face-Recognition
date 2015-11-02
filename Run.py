+from sklearn.ensemble import RandomForestClassifier
from classifier import *
from kFoldCrossValidation import *
from PIL import Image
import numpy as np
import sys, os
sys.path.append("../..")

from features import *


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 
    Returns:
        A list [X,y]
            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]


def read_image(path, sz = None):
    im = Image.open(path)
    im = im.convert("L")
                    # resize to given size (if given)
    if (sz is not None):
        im = im.resize(sz, Image.ANTIALIAS)
    testX = np.asarray(im, dtype=np.uint8)
    return testX


Path = './dataset'
[X,y] = read_images(Path)
feature = Fisherfaces()

while True:
    num = input("Enter 1 for NearestNeighbor classifier and 2 for RandomForest classifier: ")
    if num == 1:
        classifier = NearestNeighbor() 
        break
    elif num == 2:
        classifier = NearestNeighbor()
        break
    else:
        print "please make a valid entry "

model = PredictableModel(feature = feature, classifier = classifier)
model.compute(X,y)

print "sit tight for kFoldCrossValidation results..\n\n"
cv = KFoldCrossValidation(model,k = 10)
cv.validate(X,y)

print cv

