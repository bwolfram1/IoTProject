# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:01:17 2019

@author: brandon wolfram
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('C:/Users/brand/OneDrive/Documents/IoT Project/SWaT_Dataset_Attack.csv')

#dfait5 = df[['AIT501','AIT502','AIT503','AIT504','Normal/Attack']]
dfait5 = df[['AIT501','AIT502','AIT503','AIT504','FIT501','Normal/Attack']]
print(dfait5)

dfait5['Type'] = np.where(dfait5['Normal/Attack'] == 'Normal',0,1)

dfait5 = dfait5.drop('Normal/Attack',axis=1)
print(dfait5.head())

uniques, ucounts = np.unique(dfait5['Type'],return_counts=True)
print(uniques, ucounts)

dfattacks = dfait5[dfait5['Type'] == 1].sample(20)
dfnormal = dfait5[dfait5['Type'] == 0].sample(20)
print(dfattacks.head())
print(dfnormal.head())

df400 = pd.concat([dfnormal,dfattacks])
df400 = shuffle(df400)
df400 = df400.reset_index(drop=True)
print(df400.head())

df400['AIT501']  = df400['AIT501'].astype(float)
df400['AIT502']  = df400['AIT502'].astype(float)
df400['AIT503']  = df400['AIT503'].astype(float)
df400['AIT504']  = df400['AIT504'].astype(float)

corr = df400.corr()
sns.heatmap(corr)
plt.figure()

X = df400.drop('Type',axis=1)
y = df400['Type']

sns.pairplot(df400,hue = 'Type')
plt.figure()

################
#LOG REGRESSION#
################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot = True)
plt.figure()

print("Logistic Regression -----------")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred))


#filename = 'finalized_model.rx'
#pickle.dump(logreg, open(filename, 'wb'))
#print(logreg)

####################
#SVM Classification#
####################

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train,y_train)
y_predc = clf.predict(X_test)

cnf_matrixc = metrics.confusion_matrix(y_test, y_predc)
print(cnf_matrixc)
sns.heatmap(cnf_matrixc,annot = True)
plt.figure()

print("SVC --------")
print("Accuracy:",metrics.accuracy_score(y_test, y_predc))
print("Precision:",metrics.precision_score(y_test, y_predc))
print("Recall:",metrics.recall_score(y_test, y_predc))
print("F1:",metrics.f1_score(y_test, y_predc))


##################
# Decision Trees #
##################

from sklearn import tree
tlf = tree.DecisionTreeClassifier()
tlf.fit(X_train,y_train)
y_predt = tlf.predict(X_test)

cnf_matrixt = metrics.confusion_matrix(y_test, y_predt)
print(cnf_matrixt)
sns.heatmap(cnf_matrixt,annot = True)
plt.figure()

print("Decision Tree ---------")
print("Accuracy:",metrics.accuracy_score(y_test, y_predt))
print("Precision:",metrics.precision_score(y_test, y_predt))
print("Recall:",metrics.recall_score(y_test, y_predt))
print("F1:",metrics.f1_score(y_test, y_predt))


#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestClassifier
rlf = RandomForestClassifier()
rlf.fit(X_train,y_train)
y_predr = rlf.predict(X_test)

cnf_matrixr = metrics.confusion_matrix(y_test, y_predt)
print(cnf_matrixr)
sns.heatmap(cnf_matrixr,annot = True)
plt.figure()

print("Random Forest ---------------")
print("Accuracy:",metrics.accuracy_score(y_test, y_predr))
print("Precision:",metrics.precision_score(y_test, y_predr))
print("Recall:",metrics.recall_score(y_test, y_predr))
print("F1:",metrics.f1_score(y_test, y_predr))


#######
# KNN #
#######

from sklearn.neighbors import KNeighborsClassifier
klf = KNeighborsClassifier(n_neighbors=3)
klf.fit(X_train,y_train)
y_predk = klf.predict(X_test)

cnf_matrixk = metrics.confusion_matrix(y_test, y_predt)
print(cnf_matrixk)
sns.heatmap(cnf_matrixk,annot = True)
plt.figure()

print("KNN --------------")
print("Accuracy:",metrics.accuracy_score(y_test, y_predk))
print("Precision:",metrics.precision_score(y_test, y_predk))
print("Recall:",metrics.recall_score(y_test, y_predk))
print("F1:",metrics.f1_score(y_test, y_predk))

#plt.plot(df400['AIT502'][0:6])
#plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2).fit(X_train)
y_predm = kmeans.predict(X_test)

cnf_matrixm = metrics.confusion_matrix(y_test, y_predt)
print(cnf_matrixm)
sns.heatmap(cnf_matrixm,annot = True)
plt.figure()

print("KMeans --------------")
print("Accuracy:",metrics.accuracy_score(y_test, y_predm))
print("Precision:",metrics.precision_score(y_test, y_predm))
print("Recall:",metrics.recall_score(y_test, y_predm))
print("F1:",metrics.f1_score(y_test, y_predm))


