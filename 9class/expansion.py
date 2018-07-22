import csv
import numpy as np
from os import getcwd, listdir
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def read_the_file(file_names):
    ans = []
    for file_name in file_names:
        if not file_name.endswith(".csv"):
            file_names.remove(file_name)
    for file_name in file_names:
        temp = []
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            for i in reader:
                temp.append(i)
        temp = temp[100: len(temp) - 100]
        ans = ans + temp
    ans = np.asarray(ans)
    data_ = ans[:, 0: 90].astype(np.float32)
    target_ = ans[:, 90].astype(np.int)
    return data_, target_


def get_the_mean(data, target, class_number):
    x = []
    t = []
    for i in range(class_number):
        x.append([])
        t.append([])
    for i in range(len(target)):
        x[target[i]].append(data[i])
        t[target[i]].append(target[i])
    x_ = []
    t_ = []
    for i in range(class_number):
        if not len(x[i]) == 0:
            print(len(x[i]), i)
            x_.append(x[i])
            t_.append(t[i])
    x = x_
    t = t_
    for i in range(len(x)):
        x[i] = np.asarray(x[i])
        x[i] = np.concatenate((x[i], x[i] - np.mean(x[i], axis=0)), axis=1)
    for i in range(1, len(x)):
        x[0] = np.concatenate((x[0], x[i]))
        t[0] = np.concatenate((t[0], t[i]))
    return x[0].astype(np.float32), t[0].astype(np.int)

train_list = listdir(getcwd())
train_list.remove("7.csv")
train_list.remove("8.csv")
train_list.remove("empty_room_1.csv")
train_list.remove("5_chp_3_movazii.csv")
train_list.remove("5_aghab_2_movazii.csv")
data, target = read_the_file(train_list)
data, target = get_the_mean(data, target, 9)
shuffle = np.random.permutation(len(target))
data, target = data[shuffle], target[shuffle]

data_test, target_test = read_the_file(["7.csv", "8.csv", "empty_room_1.csv", "5_chp_3_movazii.csv", "5_aghab_2_movazii.csv"])
data_test, target_test = get_the_mean(data_test, target_test, class_number=9)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(data)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[100, 300, 600], n_classes=9, feature_columns=feature_columns)
dnn_clf.fit(x=data, y=target, batch_size=50, steps=40000)
prediction = list(dnn_clf.predict(data_test))
print(accuracy_score(prediction, target_test))
print(confusion_matrix(prediction, target_test))