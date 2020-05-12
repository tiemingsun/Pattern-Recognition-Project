from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import preprocessing, get_X_and_label, accuracy_std 
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron


'''Preprocessing data
standardize testing data using training data's mean and standard deviation
'''
data_train = pd.read_csv('D_train.csv')
data_test = pd.read_csv('D_test.csv')

data_train_new, new_label = preprocessing(data_train)
data_test_new, new_label = preprocessing(data_test)
# new_label = ['x_mean','y_mean','z_mean','x_st','y_st','z_st','x_max','y_max','z_max','x_min','y_min','z_min']
scaler = StandardScaler()
data_train_new[new_label] = scaler.fit_transform(data_train_new[new_label].to_numpy())
data_test_new[new_label] = scaler.transform(data_test_new[new_label].to_numpy())

train_X, train_label = get_X_and_label(data_train_new)
test_X, test_label = get_X_and_label(data_test_new)

'''Using Parameters to print confusion matrix and Classification Report for testing data
The following are:
1. Naive Bayes with Gaussian Density Estimation
2. Bayes with density Estimation, KNN
3. Support Vector Machine with radial basis funtion Kernel
4. Support Vector Machine with Polynomial Kernel
5. Neural Network: Multilayer Perceptron
6. Linear Model: Perceptron
7. Decision Tree
8. Random Forest
'''
gnb = GaussianNB()
gnb.fit(train_X, train_label)
print("Confusion matrix for Naive Bayes: \n", confusion_matrix(test_label, gnb.predict(test_X)))
print("Classification for Naive Bayes: \n",classification_report(test_label, gnb.predict(test_X)))

knn = KNeighborsClassifier(n_neighbors=14, p=1)
knn.fit(train_X, train_label)
print("Confusion matrix for KNN: \n", confusion_matrix(test_label, knn.predict(test_X)))
print("Classification for KNN neighbor=14: \n",classification_report(test_label, knn.predict(test_X)))

svc_rbf = SVC(C=0.78804628, gamma='auto', kernel='rbf')
svc_rbf.fit(train_X, train_label)
print("Confusion matrix for SVC RBF: \n", confusion_matrix(test_label, svc_rbf.predict(test_X)))
print("Classification for SVC RBF, C=0.788: \n",classification_report(test_label, svc_rbf.predict(test_X)))

print("proportion of support vectors, RBF kernel: ", len(svc_rbf.support_vectors_) / len(train_X))
print("number of svc non-zero items for coeffcients, RBF kernel: \n", len(np.where(svc_rbf.dual_coef_[0] != 0)[0]))

svc_poly = SVC(C=0.006723357536499335, gamma='auto', kernel='poly')
svc_poly.fit(train_X, train_label)
print("Confusion matrix for SVC Poly: \n", confusion_matrix(test_label, svc_poly.predict(test_X)))
print("Classification for SVC Poly, C=0.0067: \n",classification_report(test_label, svc_poly.predict(test_X)))

mlp = MLPClassifier(hidden_layer_sizes=(300,))
mlp.fit(train_X, train_label)
print("Confusion matrix for MLP: \n", confusion_matrix(test_label, mlp.predict(test_X)))
print("Classification for MLP, hidden_layer=(300,): \n",classification_report(test_label, mlp.predict(test_X)))

clf = Perceptron()
clf.fit(train_X, train_label)
print("Confusion matrix for Perceptron: \n", confusion_matrix(test_label, clf.predict(test_X)))
print("Classification for Perceptron: \n",classification_report(test_label, clf.predict(test_X)))
print("weight vector for perceptron is: \n", clf.coef_)

treeclf = tree.DecisionTreeClassifier()
treeclf.fit(train_X, train_label)
print("Confusion matrix for DecisionTree: \n", confusion_matrix(test_label, treeclf.predict(test_X)))
print("Classification for DecisionTree: \n",classification_report(test_label, treeclf.predict(test_X)))

rforest = RandomForestClassifier(n_estimators=600)
rforest.fit(train_X, train_label)
print("Confusion matrix for RandomForest: \n", confusion_matrix(test_label, rforest.predict(test_X)))
print("Classification for RandomForest 600 trees: \n",classification_report(test_label, rforest.predict(test_X)))
