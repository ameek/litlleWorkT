## import section as basic inputs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##data collection
# Data = pd.read_csv('monodata.csv', header=None) ## not to remove the first row
# Data = pd.read_csv('bidata.csv', header=None) ## not to remove the first row
# Data = pd.read_csv('tridata.csv', header=None) ## not to remove the first row
Data = pd.read_csv('mono_biData.', header=None) ## not to remove the first row


# Data = Data.drop_duplicates() ## to get the uniqe value and romove duplicate values that leads to a overfitting problem

row,col = Data.shape
print(Data.shape) ##optional for geting to know the (row,col) of the datasets
#
#  print(Data) ## to see the data set
#
# ##check for any missing value in the dataset
# X = Data.iloc[:, 0:col-1]
# y = Data.iloc[:, col-1]
# print("there is null value in the feature list=> ",any(X.isnull()))
# print("there is null vaue  in the label=> ",any(y.isnull()))

#
# ##divide the feature X and the label y col

X = Data.iloc[:, 0:col-1].values
y = Data.iloc[:, col-1].values

#
# ## find the missing values and fit them as per mean,median,mode
from sklearn.preprocessing import Imputer
# X[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]=Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True).fit_transform\
#     (X[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]])
##loop Method
X[:,:]  = Imputer(missing_values="NaN", strategy="mean", axis=0).fit_transform(X[:,:])


##dataset spliting
from sklearn.model_selection import train_test_split
## the whole data set will now part as per bellow
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70,random_state=0)

#featture scalling
from sklearn.preprocessing import StandardScaler, Normalizer
## StandardScaler the values are in between [-1,1]
## Normalizer the value of each feature will be in between [0,1]

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)


## Its time to use the Algorithms :D
##calling the algorithms from library

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
                             RandomForestClassifier,\
                             AdaBoostClassifier,\
                             ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

##now to evaluate the acuurecy and others
from sklearn.metrics import accuracy_score, log_loss, \
                            classification_report, \
                            confusion_matrix



classifiers =[BaggingClassifier(GaussianNB()),
              BaggingClassifier(KNeighborsClassifier(algorithm="kd_tree")),
              RandomForestClassifier(n_estimators=500),
              BaggingClassifier(ExtraTreesClassifier()),
              AdaBoostClassifier(),
              ExtraTreesClassifier(),
              KNeighborsClassifier(n_neighbors=5)
              ]


##model learning one by one iterrative...
for learner in classifiers:
    model = learner
    model = model.fit(X_train,y_train)
##making a new label:: y_predicted from X_test from the learned model.
    y_predicted = model.predict(X_test)

##evalutaion begins
    TN, FP, FN, TP = confusion_matrix(y_true=y_test,y_pred=y_predicted).ravel()
    accuracy = accuracy_score(y_true=y_test, y_pred=y_predicted)*100
##presentation or outut
    nameOfClassifer = model.__class__.__name__
    print('_' * 43)
    print('Classifier : {}'.format(nameOfClassifer))
    print('Accuracy : {0:.3f} %'.format(accuracy))
    # print('_'*40)
    print('TN = {}'.format(TN))
    print('FP = {}'.format(FP))
    print('FN = {}'.format(FN))
    print('TP = {}'.format(TP))