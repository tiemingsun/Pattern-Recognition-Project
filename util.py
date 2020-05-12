import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib as mpb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV



def preprocessing(data_train):
    '''Preprocess raw data into 13 new features

    count
    x_mean, y_mean, z_mean
    x_st, y_st, z_st
    x_max, y_max, z_max
    x_min, y_min, z_min
    @return
    a copy of the refined dataframe
    labels to be standardized
    '''
    # first we create a list of index name to fill in
    # create ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']
    A = []
    for i in range(12):
        A.append('X' + str(i))
    B = []
    for i in range(12):
        B.append('Y' + str(i))
    C = []
    for i in range(12):
        C.append('Z' + str(i))
    # feature extraction part: 13 features as example said
    data_train['count'] = (data_train.count(axis=1)- 3) / 3

    data_train['x_mean'] = data_train[A].mean(axis=1,skipna=True)
    data_train['y_mean'] = data_train[B].mean(axis=1,skipna=True)
    data_train['z_mean'] = data_train[C].mean(axis=1,skipna=True)

    data_train['x_st'] = data_train[A].std(axis=1, skipna=True)
    data_train['y_st'] = data_train[B].std(axis=1, skipna=True)
    data_train['z_st'] = data_train[C].std(axis=1, skipna=True)
    data_train['x_max'] = data_train[A].max(axis=1)
    data_train['y_max'] = data_train[B].max(axis=1)
    data_train['z_max'] = data_train[C].max(axis=1)
    data_train['x_min'] = data_train[A].min(axis=1)
    data_train['y_min'] = data_train[B].min(axis=1)
    data_train['z_min'] = data_train[C].min(axis=1)

    # deep copy to a new dataframe
    new_label = list(data_train.columns)[1:3] + list(data_train.columns)[-13:]
    data_train_new = data_train.loc[:,new_label].copy(deep=True)
    # preprocessing for the new 13 features
    # We use StandardScaler to preprocess features except count (as count is integer)
    # data_train_new[[x_mean]] = preprocessing.StandardScaler.fit_transform(data_train_new['x_mean'])
    new_label.remove('Class')
    new_label.remove('User')
    new_label.remove('count')
    return data_train_new, new_label

    


def get_X_and_label(df):
    '''Separate dataframe into 2 numpy array
    Could apply to both training dataset and testing dataset
    @return
    X_np: 13 feature data
    label_np: label of data
    '''
    df_1 = df[['Class']]
    label_np = df_1.to_numpy()
    label_np = np.ravel(label_np)
    df_2 = df.iloc[:,2:]
    X_np = df_2.to_numpy()
    return X_np, label_np


def accuracy_std(df_train_new, model):
    '''Apply cross validation for dataset and the model to train
    @return
    average of cross validation accuracy
    standard deviation of cross validation accuracy
    '''
    array = []
    for i in [x for x in range(12) if x!=3 and x!=4 and x!=7]:
        df_valid, df_train = df_train_new[df_train_new['User'] == i], df_train_new[df_train_new['User'] != i]
        train_X, train_label = get_X_and_label(df_train)
        valid_X, valid_label = get_X_and_label(df_valid)
    
        model.fit(train_X, train_label)
        score_valid_svc = model.score(valid_X, valid_label)
        array.append(score_valid_svc)
    
    #print('score_valid_svc_avg', np.mean(array))
    return np.mean(array), np.std(array)


def accuracy_std_nested(df_train_new, model, param, M):
    '''Apply nested cross validation for dataset and the model to train
    Call Randomized Search CV to accelerate parameter selection
    @return
    average of cross validation accuracy
    standard deviation of cross validation accuracy
    Candidate for the best parameter
    '''
    best = []
    array = []
    for i in [x for x in range(12) if x!=3 and x!=4 and x!=7]:
        df_valid, df_train = df_train_new[df_train_new['User'] == i], df_train_new[df_train_new['User'] != i]
        train_X, train_label = get_X_and_label(df_train)
        valid_X, valid_label = get_X_and_label(df_valid)
        
        randscv = RandomizedSearchCV(model, param, cv=5, n_iter=10)
        randscv.fit(train_X, train_label)
        print("Best params are: ", randscv.best_params_)
        model = M(p=randscv.best_params_['p'], n_neighbors=randscv.best_params_['n_neighbors'])
        best.append(randscv.best_params_['n_neighbors'])
        model.fit(train_X, train_label)
        score_valid_svc = model.score(valid_X, valid_label)

        array.append(score_valid_svc)
    
    print('score_valid_svc_avg', np.mean(array))
    return np.mean(array), np.std(array), best


