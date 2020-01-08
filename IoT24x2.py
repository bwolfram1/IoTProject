# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 04:57:29 2020

@author: brand
"""

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import fftpack
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('C:/Users/brand/OneDrive/Documents/IoT Project/SWaT_Dataset_Attack.csv')
df['Type'] = np.where(df['Normal/Attack'] == 'Normal',0,1)
###############
#corr = df.corr()
#cop = sns.heatmap(corr)
###############
#Using AIT502 because it has the highest positive correlation for attacks from the AIT50X. 

dfait502 = df[['AIT502','Type']]

seglen = 200
dflen = len(dfait502)

segments = [dfait502.iloc[i:i + seglen] for i in range(0, len(dfait502), seglen)]
del segments[-1]


svals = []
for seg in range(len(segments)):
    curr = segments[seg]
    curr = curr.reset_index(drop=True)
    curvals = curr['AIT502']
    uniques = curr['Type'].unique()
    if len(uniques) == 1:
        if uniques[0] == 0:
            curvals[200] = 0
        else:
            curvals[200] = 1
    else:
        curvals[200] = 1
    svals.append(curvals)
svalzip = zip(svals)
dfx = pd.DataFrame(svals,columns=range(201))
dfx = dfx.reset_index(drop=True)

dfattacks = dfx[dfx[200] == 1].sample(seglen)
dfnormal = dfx[dfx[200] == 0].sample(seglen)

df4x2 = pd.concat([dfnormal,dfattacks])
df4x2 = shuffle(df4x2)
df4x2 = df4x2.reset_index(drop=True)

y = df4x2[200]
df4x2 = df4x2.drop(200,axis=1)

meta = []
xi = range(200)
pca = PCA(n_components=3)
xs = StandardScaler().fit_transform(df4x2)
pc = pca.fit_transform(xs)
for i in range(len(df4x2)):
    sc = df4x2.iloc[i]
    b, m = polyfit(xi, sc, 1)
    mb = [m,b]
    meta.append(mb)
dfmeta = pd.DataFrame(meta,columns=['Slope','Intercept'])
dfpc = pd.DataFrame(pc, columns=['PC1','PC2','PC3'])
dfpcy = pd.concat([dfpc,y],axis = 1)
dfall = pd.concat([dfmeta, dfpc, y],axis = 1)
dfpcy = dfpcy.rename(columns={200 : 'Type'})

plt.scatter(x=dfpcy['PC2'],y=dfpcy['PC1'], c=dfpcy['Type'])
plt.show()
plt.scatter(x=dfall['Intercept'],y=dfall['PC1'], c=dfpcy['Type'])
plt.show()
corr2 = dfall.corr()
sns.heatmap(corr2)

#Testing some theories#
plt.plot(range(200),df4x2.iloc[2])
sample_freq = fftpack.fftfreq(200, d = 0.2)
sig_fft = fftpack.fft(df4x2.iloc[3])
plt.plot(range(199),sig_fft[1:])
################
#LOG REGRESSION#
################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

Xx = dfall.drop(200,axis=1)
yy = dfall[200]

X_train,X_test,y_train,y_test=train_test_split(Xx,yy,test_size=0.25,random_state=0)

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
#how to get deep learning algo to find metadata features.
# foruir transformation

#deep learning encoding 
        
    