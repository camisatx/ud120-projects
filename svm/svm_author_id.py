#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Uncomment the lines below if you want to train on 1 percent of data
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# Through limited testing, these arguments provided the best accuracy
clf = SVC(kernel='rbf', C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print('Training took %.2f seconds' % float(time() - t0))

t1 = time()
pred = clf.predict(features_test)
print('Predicting took %.2f seconds' % float(time() - t1))

acc = accuracy_score(pred, labels_test)
print('The accuracy was %.2f percent' % float(acc * 100))

# Finds how many predicted emails (~1700) were from Chris
# chris_class = 0
# for i in range(0, len(pred)):
#     if pred[i] == 1:
#         chris_class += 1
# print('There were %i emails predicted to be from Chris' % chris_class)