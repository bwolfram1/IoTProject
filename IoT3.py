# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 07:01:23 2020

@author: brand
"""
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import fftpack
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('C:/Users/Brandon/Documents/Data Science/Attack.csv')
df['Type'] = np.where(df['Normal/Attack'] == 'Normal',0,1)
###############
#corr = df.corr()
#cop = sns.heatmap(corr)
###############
#Using AIT502 because it has the highest positive correlation for attacks from the AIT50X. 
np.random.seed(1)
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
            curvals[seglen] = 0
        else:
            curvals[seglen] = 1
    else:
        curvals[seglen] = 1
    svals.append(curvals)
svalzip = zip(svals)
dfx = pd.DataFrame(svals,columns=range(seglen+1))
dfx = dfx.reset_index(drop=True)

dfattacks = dfx[dfx[seglen] == 1].sample(200) #200 samples
dfnormal = dfx[dfx[seglen] == 0].sample(200) #200 samples

df4x2 = pd.concat([dfnormal,dfattacks])
df4x2 = shuffle(df4x2)
df4x2 = df4x2.reset_index(drop=True)

#End of 200 segment
segmented = True 
if segmented == True: 
    segment = 200
    RAW = df4x2.iloc[:,:segment]
    RAW
else:
    RAW = df4x2.iloc[:,:seglen]

##END OF SLICING ORIGINAL
    
#Model 2 - Smoothing
m2Raw = RAW
yReal = df4x2[seglen] 
xBase = list(range(segment))
xR = pd.DataFrame(xBase)
xH = xR.values.reshape(-1,1)
sf = 0.1

m2Smo = []
mDer = []
mSI = []


for x in range(len(m2Raw)):
    r = m2Raw.iloc[x,:]
    rh = r.values.reshape(1, -1)
    regressor = LinearRegression()  
    # plt.plot(xH, r)
    # plt.title(x)
    # plt.show()
    regressor.fit(xH,r)
    slo = regressor.coef_[0]
    intr = regressor.intercept_
    slInt = [slo, intr]
    s = UnivariateSpline(xBase, r, s=sf)
    sm = s(xBase)
    sd = s.derivatives(1)
    mSI.append(slInt)
    mDer.append(sd)
    m2Smo.append(sm)
    
plt.plot(xBase,r)
#regular spline - Model 2 
smozip = zip(m2Smo)
df2 = pd.DataFrame(smozip)
df22 = pd.DataFrame(df2[0].values.tolist())
#Derivative spline 
derzip = zip(mDer)
dfd = pd.DataFrame(derzip)
dfd2 = pd.DataFrame(dfd[0].values.tolist())
#Slope-Int Model
slzip = zip(mSI)
dfsi = pd.DataFrame(slzip)
dfsi2 = pd.DataFrame(dfsi[0].values.tolist())

#Model 3
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(df22)
df3 = pd.DataFrame(data = principalComponents, columns = ['pca 1', 'pca 2','pca 3'])

#Model 5 

pca1 = PCA(n_components = 3)
vca = pca1.fit_transform(dfd2)
df5 = pd.DataFrame(data = vca, columns = ['vca 1', 'vca 2','vca 3'])

#Model 4 
combo3 = pd.concat([df3, df5], axis = 1)

#Model 6

pca6 = PCA(n_components = 2)
leanpca = pca6.fit_transform(df22)
df61 = pd.DataFrame(data = leanpca, columns = ['pca 1', 'pca 2'])
leanvca = pca6.fit_transform(dfd2)
df62 = pd.DataFrame(data = leanvca, columns = ['vca 1', 'vca 2'])
combo2 = pd.concat([df61, df62], axis = 1)


##Predictions
#Model 1 - Log
X_train1,X_test1,y_train1,y_test1=train_test_split(dfsi2,yReal,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train1,y_train1)
y_pred1 = logreg.predict(X_test1)

cnf_matrix = metrics.confusion_matrix(y_test1, y_pred1)
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot = True)
plt.figure()

print("Logistic Regression ----------- Model 1")
print("Accuracy:",metrics.accuracy_score(y_test1, y_pred1))
print("Precision:",metrics.precision_score(y_test1, y_pred1))
print("Recall:",metrics.recall_score(y_test1, y_pred1))
print("F1:",metrics.f1_score(y_test1, y_pred1))

#Model 2 - Log
X_train,X_test,y_train,y_test=train_test_split(m2Raw,yReal,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot = True)
plt.figure()

print("Logistic Regression ----------- Model 2")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred))

#Model 3 - Log
X_train3,X_test3,y_train3,y_test3=train_test_split(df3,yReal,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train3,y_train3)

y_pred3 = logreg.predict(X_test3)

cnf_matrix = metrics.confusion_matrix(y_test3, y_pred3)
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot = True)
plt.figure()

print("Logistic Regression ----------- Model 3")
print("Accuracy:",metrics.accuracy_score(y_test3, y_pred3))
print("Precision:",metrics.precision_score(y_test3, y_pred3))
print("Recall:",metrics.recall_score(y_test3, y_pred3))
print("F1:",metrics.f1_score(y_test3, y_pred3))
#Model 4 - Log
X_train4,X_test4,y_train4,y_test4=train_test_split(combo3,yReal,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train4,y_train4)

y_pred4 = logreg.predict(X_test4)

cnf_matrix = metrics.confusion_matrix(y_test4, y_pred4)
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot = True)
plt.figure()

print("Logistic Regression ----------- Model 4")
print("Accuracy:",metrics.accuracy_score(y_test4, y_pred4))
print("Precision:",metrics.precision_score(y_test4, y_pred4))
print("Recall:",metrics.recall_score(y_test4, y_pred4))
print("F1:",metrics.f1_score(y_test4, y_pred4))

#Model 5 
X_train5,X_test5,y_train5,y_test5=train_test_split(df5,yReal,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train5,y_train5)

y_pred5 = logreg.predict(X_test5)

cnf_matrix = metrics.confusion_matrix(y_test5, y_pred5)
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot = True)
plt.figure()

print("Logistic Regression ----------- Model 5")
print("Accuracy:",metrics.accuracy_score(y_test5, y_pred5))
print("Precision:",metrics.precision_score(y_test5, y_pred5))
print("Recall:",metrics.recall_score(y_test5, y_pred5))
print("F1:",metrics.f1_score(y_test5, y_pred5))

#Model 6 
X_train6,X_test6,y_train6,y_test6=train_test_split(combo2,yReal,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train6,y_train6)

y_pred6 = logreg.predict(X_test6)

cnf_matrix = metrics.confusion_matrix(y_test6, y_pred6)
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot = True)
plt.figure()

print("Logistic Regression ----------- Model 6")
print("Accuracy:",metrics.accuracy_score(y_test6, y_pred6))
print("Precision:",metrics.precision_score(y_test6, y_pred6))
print("Recall:",metrics.recall_score(y_test6, y_pred6))
print("F1:",metrics.f1_score(y_test6, y_pred6))

##SVC'
#Model 1
#Model 2
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
#Model 3
clf = svm.SVC()
clf.fit(X_train3,y_train3)
y_predc3 = clf.predict(X_test3)

cnf_matrixc3 = metrics.confusion_matrix(y_test3, y_predc3)
print(cnf_matrixc3)
sns.heatmap(cnf_matrixc3,annot = True)
plt.figure()

print("SVC --------")
print("Accuracy:",metrics.accuracy_score(y_test3, y_predc3))
print("Precision:",metrics.precision_score(y_test3, y_predc3))
print("Recall:",metrics.recall_score(y_test3, y_predc3))
print("F1:",metrics.f1_score(y_test3, y_predc3))
#Model 4
clf = svm.SVC()
clf.fit(X_train4,y_train4)
y_predc4 = clf.predict(X_test4)

cnf_matrixc4 = metrics.confusion_matrix(y_test4, y_predc4)
print(cnf_matrixc4)
sns.heatmap(cnf_matrixc4,annot = True)
plt.figure()

print("SVC --------")
print("Accuracy:",metrics.accuracy_score(y_test4, y_predc4))
print("Precision:",metrics.precision_score(y_test4, y_predc4))
print("Recall:",metrics.recall_score(y_test4, y_predc4))
print("F1:",metrics.f1_score(y_test4, y_predc4))
#Model 5
clf = svm.SVC()
clf.fit(X_train5,y_train5)
y_predc5 = clf.predict(X_test5)

cnf_matrixc5 = metrics.confusion_matrix(y_test5, y_predc5)
print(cnf_matrixc5)
sns.heatmap(cnf_matrixc5,annot = True)
plt.figure()

print("SVC --------")
print("Accuracy:",metrics.accuracy_score(y_test5, y_predc5))
print("Precision:",metrics.precision_score(y_test5, y_predc5))
print("Recall:",metrics.recall_score(y_test5, y_predc5))
print("F1:",metrics.f1_score(y_test5, y_predc5))
#Model 6 - SVC
clf = svm.SVC()
clf.fit(X_train6,y_train6)
y_predc6 = clf.predict(X_test6)

cnf_matrixc6 = metrics.confusion_matrix(y_test6, y_predc6)
print(cnf_matrixc6)
sns.heatmap(cnf_matrixc6,annot = True)
plt.figure()

print("SVC --------")
print("Accuracy:",metrics.accuracy_score(y_test6, y_predc6))
print("Precision:",metrics.precision_score(y_test6, y_predc6))
print("Recall:",metrics.recall_score(y_test6, y_predc6))
print("F1:",metrics.f1_score(y_test6, y_predc6))

#KNN n = 3
#Model 1
#Model 2
klf = KNeighborsClassifier(n_neighbors=3)
klf.fit(X_train,y_train)
y_predk = klf.predict(X_test)

cnf_matrixk = metrics.confusion_matrix(y_test, y_predk)
print(cnf_matrixk)
sns.heatmap(cnf_matrixk,annot = True)
plt.figure()

print("KNN --------------")
print("Accuracy:",metrics.accuracy_score(y_test, y_predk))
print("Precision:",metrics.precision_score(y_test, y_predk))
print("Recall:",metrics.recall_score(y_test, y_predk))
print("F1:",metrics.f1_score(y_test, y_predk))
#Model 3
klf = KNeighborsClassifier(n_neighbors=3)
klf.fit(X_train3,y_train3)
y_predk3 = klf.predict(X_test3)

cnf_matrixk3 = metrics.confusion_matrix(y_test3, y_predk3)
print(cnf_matrixk3)
sns.heatmap(cnf_matrixk3,annot = True)
plt.figure()

print("KNN --------------")
print("Accuracy:",metrics.accuracy_score(y_test3, y_predk3))
print("Precision:",metrics.precision_score(y_test3, y_predk3))
print("Recall:",metrics.recall_score(y_test3, y_predk3))
print("F1:",metrics.f1_score(y_test3, y_predk3))
#Model 4
klf = KNeighborsClassifier(n_neighbors=3)
klf.fit(X_train4,y_train4)
y_predk4 = klf.predict(X_test4)

cnf_matrixk4 = metrics.confusion_matrix(y_test4, y_predk4)
print(cnf_matrixk4)
sns.heatmap(cnf_matrixk4,annot = True)
plt.figure()

print("KNN --------------")
print("Accuracy:",metrics.accuracy_score(y_test4, y_predk4))
print("Precision:",metrics.precision_score(y_test4, y_predk4))
print("Recall:",metrics.recall_score(y_test4, y_predk4))
print("F1:",metrics.f1_score(y_test4, y_predk4))
#model 5
klf = KNeighborsClassifier(n_neighbors=3)
klf.fit(X_train5,y_train5)
y_predk5 = klf.predict(X_test5)

cnf_matrixk5 = metrics.confusion_matrix(y_test5, y_predk5)
print(cnf_matrixk5)
sns.heatmap(cnf_matrixk5,annot = True)
plt.figure()

print("KNN --------------")
print("Accuracy:",metrics.accuracy_score(y_test5, y_predk5))
print("Precision:",metrics.precision_score(y_test5, y_predk5))
print("Recall:",metrics.recall_score(y_test5, y_predk5))
print("F1:",metrics.f1_score(y_test5, y_predk5))
#model 6
klf = KNeighborsClassifier(n_neighbors=3)
klf.fit(X_train6,y_train6)
y_predk6 = klf.predict(X_test6)

cnf_matrixk6 = metrics.confusion_matrix(y_test6, y_predk6)
print(cnf_matrixk6)
sns.heatmap(cnf_matrixk6,annot = True)
plt.figure()

print("KNN --------------")
print("Accuracy:",metrics.accuracy_score(y_test6, y_predk6))
print("Precision:",metrics.precision_score(y_test6, y_predk6))
print("Recall:",metrics.recall_score(y_test6, y_predk6))
print("F1:",metrics.f1_score(y_test6, y_predk6))

#Random forest
#Model 1
#Model 2
rlf = RandomForestClassifier()
rlf.fit(X_train,y_train)
y_predr = rlf.predict(X_test)

cnf_matrixk = metrics.confusion_matrix(y_test, y_predr)
print(cnf_matrixk)
sns.heatmap(cnf_matrixk,annot = True)
plt.figure()

print("Random Forest m2 --------------")
print("Accuracy:",metrics.accuracy_score(y_test, y_predr))
print("Precision:",metrics.precision_score(y_test, y_predr))
print("Recall:",metrics.recall_score(y_test, y_predr))
print("F1:",metrics.f1_score(y_test, y_predr))
#Model 3
rlf.fit(X_train3,y_train3)
y_predr = rlf.predict(X_test3)

cnf_matrixk = metrics.confusion_matrix(y_test3, y_predr)
print(cnf_matrixk)
sns.heatmap(cnf_matrixk,annot = True)
plt.figure()

print("Random Forest m3 --------------")
print("Accuracy:",metrics.accuracy_score(y_test3, y_predr))
print("Precision:",metrics.precision_score(y_test3, y_predr))
print("Recall:",metrics.recall_score(y_test3, y_predr))
print("F1:",metrics.f1_score(y_test3, y_predr))
#Model 4
rlf.fit(X_train4,y_train4)
y_predr = rlf.predict(X_test4)

cnf_matrixk = metrics.confusion_matrix(y_test4, y_predr)
print(cnf_matrixk)
sns.heatmap(cnf_matrixk,annot = True)
plt.figure()

print("Random Forest m4 --------------")
print("Accuracy:",metrics.accuracy_score(y_test4, y_predr))
print("Precision:",metrics.precision_score(y_test4, y_predr))
print("Recall:",metrics.recall_score(y_test4, y_predr))
print("F1:",metrics.f1_score(y_test4, y_predr))
#model 5
rlf.fit(X_train5,y_train5)
y_predr = rlf.predict(X_test5)

cnf_matrixk = metrics.confusion_matrix(y_test5, y_predr)
print(cnf_matrixk)
sns.heatmap(cnf_matrixk,annot = True)
plt.figure()

print("Random Forest m5 --------------")
print("Accuracy:",metrics.accuracy_score(y_test5, y_predr))
print("Precision:",metrics.precision_score(y_test5, y_predr))
print("Recall:",metrics.recall_score(y_test5, y_predr))
print("F1:",metrics.f1_score(y_test5, y_predr))
#model 6
rlf.fit(X_train6,y_train6)
y_predr = rlf.predict(X_test6)

cnf_matrixk = metrics.confusion_matrix(y_test6, y_predr)
print(cnf_matrixk)
sns.heatmap(cnf_matrixk,annot = True)
plt.figure()

print("Random Forest m6 --------------")
print("Accuracy:",metrics.accuracy_score(y_test6, y_predr))
print("Precision:",metrics.precision_score(y_test6, y_predr))
print("Recall:",metrics.recall_score(y_test6, y_predr))
print("F1:",metrics.f1_score(y_test6, y_predr))