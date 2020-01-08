#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

#Reaing the data set
df_data = pd.read_csv('UniversalBank.csv')

#Drop columns that won't be needed
df_data.drop(columns = ['ID','ZIP Code'],axis = 1,inplace = True)

#Checking for missing values
df_data.isnull().sum()

#Getting the features and dependent variable
x = df_data.drop(columns = ['Personal Loan'])
y = df_data['Personal Loan']

#Data visualization of features vs the dependent variable
#First some data correlation
sns.heatmap(x.corr(),annot = True,linewidths = .1)
sns.distplot(x['Age'],bins = x['Age'].max(),kde = False)
sns.jointplot(x = 'Personal Loan',y = 'Age',data = df_data)
sns.jointplot(x = 'Personal Loan',y = 'Experience',data = df_data)
sns.distplot(x['Experience'],bins = x['Experience'].max(),kde = False)
sns.jointplot(x = 'Personal Loan',y = 'Income',data = df_data)
sns.barplot(x = y,y = x['Family'],hue = x['Family'])
sns.barplot(x = y,y = x['Education'],hue = x['Education'])

#Data prepation
#Droping the colums with high correlation
x.drop(columns = ['Age','CCAvg'],axis = 1, inplace = True)
sns.heatmap(x.corr(),annot = True,linewidths = .1)

"""Didn't make feature encoding since the columns with categorical values
already have the same values, 0 ad 1 if i were to make feature encoding"""

#Data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#KNN model creation with a for loop to get various testing with different K values
error_rate = []
for i in range(1,51):
     knn_classifier = KNeighborsClassifier(n_neighbors = i)
     knn_classifier.fit(x_train,y_train)
     pred = knn_classifier.predict(x_test)
     error_rate.append(np.mean(pred != y_test))
     
plt.scatter(x = range(1,51),y = error_rate)
plt.title('Plotting the error rate')
plt.xlabel('KNN neighbors')
plt.ylabel('Error')
plt.show()

#With a logistic regression model, accuracy of 95 
lr_classifier = LogisticRegression()
lr_classifier.fit(x_train,y_train)

pred = lr_classifier.predict(x_test)
cm = confusion_matrix(y_test,pred)
report = classification_report(y_test,pred)

#Models evaluation comparison
#With 3 K(Best one with an accuracy of 96)
knn_classifier = KNeighborsClassifier(n_neighbors = 3)
knn_classifier.fit(x_train,y_train)
pred = knn_classifier.predict(x_test)
cm = confusion_matrix(y_test,pred)
report = classification_report(y_test,pred)

#With 50 K an accuracy of 94
knn_classifier = KNeighborsClassifier(n_neighbors = 50)
knn_classifier.fit(x_train,y_train)
pred = knn_classifier.predict(x_test)
cm = confusion_matrix(y_test,pred)
report = classification_report(y_test,pred)