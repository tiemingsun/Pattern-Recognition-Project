import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import preprocessing, get_X_and_label, accuracy_std, accuracy_std_nested
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.dummy import DummyClassifier

'''Preprocessing data
standardize testing data using training data's mean and standard deviation
'''
data_train = pd.read_csv('D_train.csv')
data_test = pd.read_csv('D_test.csv')

data_train_new, new_label = preprocessing(data_train)
data_test_new, new_label = preprocessing(data_test)
scaler = StandardScaler()
data_train_new[new_label] = scaler.fit_transform(data_train_new[new_label].to_numpy())
data_test_new[new_label] = scaler.transform(data_test_new[new_label].to_numpy())

##------------------------------------------------------------
'''Support vector machine

Grid Search for C and gamma to kernel=rbf
Plot for Grid Search
Search best C for kernel=rbf
Search best C for kernel=poly

'''
numsamples = 4
gamma_range = np.logspace(-5, 5,num= numsamples) 
C_range = np.logspace(-1, 3, num= numsamples)
ACC = np.zeros((len(gamma_range), len(C_range)))
DEV = np.zeros((len(gamma_range), len(C_range)))

for i in range(numsamples):
    for j in range(numsamples):
        gamma = gamma_range[i]
        C = C_range[j]
        svc_posture = SVC(C=C, gamma=gamma, kernel='rbf')
        ACC[i][j], DEV[i][j] = accuracy_std(data_train_new, svc_posture)

plt.imshow(ACC,interpolation = 'nearest', cmap ='Blues')
plt.xlabel('log(gamma)')
plt.ylabel('log(C)')
plt.xticks([0,3],['-5','5']) 
plt.yticks([0,3],['-1','3'])
plt.colorbar()
plt.title("accuracy for SVM")
plt.show()


C_range_1 = np.logspace(-3, 3, num=30)
accuracy_array_rbf = np.zeros((len(C_range_1)))
deviation_array_rbf = np.zeros((len(C_range_1)))
for i in range(len(C_range_1)):
    C = C_range_1[i]
    svc_posture = SVC(C=C, kernel='rbf', gamma='auto')
    accuracy_array_rbf[i], deviation_array_rbf[i] = accuracy_std(data_train_new, svc_posture)
index_max = np.where(accuracy_array_rbf == np.max(accuracy_array_rbf))
print("Best C for rbf kernel is:", C_range_1[index_max[0]] )
print("SVM rbf: Average CV accuracy is: ", np.max(accuracy_array_rbf))

C_range_2 = np.logspace(-3, 3, num=30)
accuracy_array_poly = np.zeros((len(C_range_2)))
deviation_array_poly = np.zeros((len(C_range_2)))
for i in range(len(C_range_2)):
    C = C_range_2[i]
    svc_posture = SVC(C=C, kernel='poly', gamma='auto')
    accuracy_array_poly[i], deviation_array_poly[i] = accuracy_std(data_train_new, svc_posture)
index_max = np.where(accuracy_array_poly == np.max(accuracy_array_poly))
print("Best C for poly kernel is:", C_range_2[index_max[0]] )
print("SVM poly: Average CV accuracy is: ", np.max(accuracy_array_poly))

##------------------------------------------------------------
'''K nearest neighbors

PCA when n_neighbors is low, avoid the 'curse of dimensionality'
Nested Cross Validation for KNN, k=3
Search best n_neighbors for KNN

'''
#PCA KNN
valid_array = []
valid_std = []
score_valid_3nn_avg = 0
score_test_3nn_avg=0
for j in range(3, 14):
    temp = []
    for i in [x for x in range(12) if x!=3 and x!=4 and x!=7]:
        pca = PCA(n_components=j)
        df_valid, df_train = data_train_new[data_train_new['User'] == i], data_train_new[data_train_new['User'] != i]

        train_X, train_label = get_X_and_label(df_train)
        valid_X, valid_label = get_X_and_label(df_valid)
        test_X, test_label = get_X_and_label(data_test_new)

        train_X = pca.fit_transform(train_X)
        valid_X = pca.transform(valid_X)
        test_X = pca.transform(test_X)
        
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(train_X, train_label)
        
        score_3nn = neigh.score(valid_X, valid_label)
        temp.append(score_3nn)
    print('score_valid_3nn_avg if {}features:'.format(j), np.mean(temp))
    valid_std.append(np.std(temp))
    valid_array.append(np.mean(temp))

# nested cross validation KNN
param = {'n_neighbors':range(1,50), 'p':[1,2]}
knn = KNeighborsClassifier()
mean, std, best = accuracy_std_nested(data_train_new, knn, param, KNeighborsClassifier)
print("According to nested cross validation, find best n_neighbors", best)

final_mean_knn = []
final_std_knn = []
for j in range(5, 50, 3):
    neigh = KNeighborsClassifier(n_neighbors=j)
    mean, std = accuracy_std(data_train_new, neigh)
    final_mean_knn.append(mean)
    final_std_knn.append(std)
index_max = np.where(final_mean_knn == np.max(final_mean_knn))
print("Best n_neighbors is:", 5 + 3 * index_max[0][0] )
print("KNN: Average CV accuracy is: ", np.max(final_mean_knn))

##------------------------------------------------------------
'''Naive Bayes with Gaussian Density

Get Average CV accuracy
'''
gnb = GaussianNB()
mean_gnb, std_gnb = accuracy_std(data_train_new, gnb)
print("Naive Bayes: Average CV accuracy is ", mean_gnb)

##------------------------------------------------------------
'''Perceptron

Get Average CV accuracy
'''
clf = Perceptron()
mean_perc, std_perc = accuracy_std(data_train_new, clf)
print("Perceptron: Average CV accuracy is ", mean_perc)

##------------------------------------------------------------
'''Decision Tree Theorem

Get Average CV accuracy
'''

treeclf = tree.DecisionTreeClassifier()
mean_tree, std_tree = accuracy_std(data_train_new, treeclf)
print("Decision Tree: Average CV accuracy is ", mean_tree)

##------------------------------------------------------------
'''Random Forest

Get Average CV accuracy
Search for best n_estimators for random forest
'''
final_mean_rand = []
final_std_rand = []
for j in range(400, 1000, 100):
    rforest = RandomForestClassifier(n_estimators=j)
    mean, std = accuracy_std(data_train_new, rforest)
    final_mean_rand.append(mean)
    final_std_rand.append(std)
index_max = np.where(final_mean_rand == np.max(final_mean_rand))
print("Best tree number is:", 400 + 100 * index_max[0][0] )
print("Random Forest: Average CV accuracy is: ", np.max(final_mean_rand))

##------------------------------------------------------------
'''Multi-layer Perceptron

Get Average CV accuracy
Search for best hidden_layer_sizes for MLP
'''
final_mean_mlp = []
final_std_mlp = []
for j in range(100, 501, 100):
    mlp = MLPClassifier(hidden_layer_sizes=(j,))
    mean, std = accuracy_std(data_train_new, mlp)
    final_mean_mlp.append(mean)
    final_std_mlp.append(std)
index_max = np.where(final_mean_mlp == np.max(final_mean_mlp))
print("Best neuron number is:", 100 + 100 * index_max[0][0] )
print("Multi-layer Perceptron: Average CV accuracy is: ", np.max(final_mean_mlp))
##------------------------------------------------------------
'''Dummy Classifier

Get Average CV accuracy
'''
dummy_clf = DummyClassifier(strategy="stratified")
mean_dummy, std_dummy = accuracy_std(data_train_new, dummy_clf)
print("Dummy: test accuracy is ", mean_dummy)
