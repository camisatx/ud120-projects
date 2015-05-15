#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
from sklearn import tree
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print('Training took %.2f seconds' % float(time() - t0))

t1 = time()
pred = clf.predict(features_test)
print('Predicting took %.2f seconds' % float(time() - t1))

acc = accuracy_score(pred, labels_test)
print('The accuracy was %.2f percent' % float(acc * 100))