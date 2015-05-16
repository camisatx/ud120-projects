#!/usr/bin/python

from time import time
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

# The training data (features_train, labels_train) have both "fast" and "slow" points mixed in together.
# Separate them so we can give them different colors in the scatterplot, and visually identify them.
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()

clf = neighbors.KNeighborsClassifier(n_neighbors=18)      # Accuracy = 0.94
# clf = ensemble.RandomForestClassifier(min_samples_split=10)       # Accuracy = 0.928
# clf = ensemble.AdaBoostClassifier()     # Accuracy = 0.924

t0 = time()
clf.fit(features_train, labels_train)
print('Training took %.2f seconds' % float(time() - t0))

t1 = time()
pred = clf.predict(features_test)
print('Predicting took %.2f seconds' % float(time() - t1))

acc = accuracy_score(pred, labels_test)
print('The accuracy was %.2f percent' % float(acc * 100))

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
