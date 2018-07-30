import numpy as np
import csv
from os import listdir, getcwd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def read_the_file(file_name):
    ans = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for i in reader:
            ans.append(i)
        ans = ans[100: len(ans) - 100]
        return ans


files = listdir(getcwd())
temp = []
for file in files:
    if file.endswith(".csv"):
        temp = temp + read_the_file(file)
temp = np.asarray(temp)
data = temp[:, 0:90].astype(np.float32)
target = temp[:, 90].astype(np.int)
shuffle = np.random.permutation(len(data))
data = data[shuffle]
target = target[shuffle]

print("SGD_clf:")
SGD_clf = SGDClassifier(loss="log")
print(confusion_matrix(target, cross_val_predict(SGD_clf, data, target, cv=10)))
print(cross_val_score(SGD_clf, data, target, cv=10, scoring="accuracy"))

print("Forest_clf:")
Forest = RandomForestClassifier(random_state=42, n_estimators=15)
print(confusion_matrix(target, cross_val_predict(Forest, data, target, cv=10)))
print(cross_val_score(Forest, data, target, cv=10, scoring="accuracy"))

print("KNN: ")
KNN = KNeighborsClassifier(n_neighbors=10)
print(confusion_matrix(target, cross_val_predict(KNN, data, target, cv=10)))
print(cross_val_score(KNN, data, target, cv=10, scoring="accuracy"))
