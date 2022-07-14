# importing required values
#%matplotlib inline
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
import warnings
import os
import zipfile
import tkinter as tk
from tkinter.filedialog import askopenfilename
tk.Tk().withdraw()
warnings.filterwarnings("ignore")
banking_churn = pd.read_csv("https://raw.githubusercontent.com/Livuza/ADS-April-2021/main/Assignments/Assignment%202/banking_churn.csv")
#banking_churn.head()
#banking_churn.describe()
#banking_churn.shape
#banking_churn.shape[0]
#banking_churn.shape[1]
#banking_churn.columns
#st.write("Drop columns with no effect on whether customer will churn (RowNumber, CustomerId,Surname)")
data = banking_churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace = True)
#banking_churn.head()
#banking_churn.describe()
#banking_churn.columns
#st.subheader("Display Geography")
data = banking_churn["Geography"].unique()
#st.subheader("Count per Geography")
banking_churn["Geography"].value_counts()

#st.subheader("Effect of Geography on customer churn")
#st.write(sns.countplot(banking_churn["Exited"]))
#counts = banking_churn.groupby(['Geography', 'Exited']).Exited.count().unstack()
#counts.plot(kind='bar', stacked=True)
#st.pyplot()
#st.write(counts)

#st.write("Remove the categorical columns Geography and Gender")
#tempdata = banking_churn.drop(['Geography','Gender'], axis=1)
#st.write(tempdata.head(2))

#st.write("Create one hot encoded columns for Geography and Gender")
data = pd.get_dummies(banking_churn, drop_first=True)
#data.head()
#Geography = pd.get_dummies(banking_churn.Geography).iloc[:,0:]
#Gender = pd.get_dummies(banking_churn.Gender).iloc[:,0:]
#banking_churn = pd.concat([banking_churn,Geography,Gender], axis=1)
#st.write(banking_churn.head(2))

#st.subheader("Not Handling Imbalance")
#data["Exited"].value_counts()
#sns.countplot(data["Exited"])

X = data.drop(['Exited'], axis=1)
y = data['Exited']
#X.head()

#st.subheader("Handling Imbalanced Data with SMOTE - Simplelic Minority of Technics")
from imblearn.over_sampling import SMOTE
X_res,y_res = SMOTE().fit_resample(X,y)
#y_res.value_counts()

#st.subheader("Splitting SMOTE Dataset Into Training and Test Set")
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
#st.write("shape of train and test splits")
#train_x.shape, test_x.shape, train_y.shape, test_y.shape

#st.subheader("Feature Scaling")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)
#train_x

#st.subheader("Logistic Regression")
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(train_x, train_y)
y_pred1 = log.predict(test_x)
from sklearn.metrics import accuracy_score
accuracy_score(test_y, y_pred1)
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(test_y, y_pred1)
recall_score(test_y, y_pred1)
f1_score(test_y, y_pred1)

#st.subheader("SVC - Support Vector Classifier")
from sklearn import svm
svm = svm.SVC()
svm.fit(train_x, train_y)
y_pred2 = svm.predict(test_x)
accuracy_score(test_y, y_pred2)
precision_score(test_y, y_pred2)
recall_score(test_y, y_pred2)
f1_score(test_y, y_pred2)

#st.subheader("KNeighbors Classifier")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_x, train_y)
y_pred3 = knn.predict(test_x)
accuracy_score(test_y, y_pred3)
precision_score(test_y, y_pred3)
recall_score(test_y, y_pred3)
f1_score(test_y, y_pred3)

#st.subheader("Decision Tree Classifier")
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train_x, train_y)
y_pred4 = dt.predict(test_x)
accuracy_score(test_y, y_pred4)
precision_score(test_y, y_pred4)
recall_score(test_y, y_pred4)
f1_score(test_y, y_pred4)

#st.subheader("Random Forest Classifier")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(train_x, train_y)
y_pred5 = rf.predict(test_x)
accuracy_score(test_y, y_pred5)
precision_score(test_y, y_pred5)
recall_score(test_y, y_pred5)
f1_score(test_y, y_pred5)
#st.subheader("Gradient Boosting Classifier")
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(train_x, train_y)
y_pred6 = gb.predict(test_x)
accuracy_score(test_y, y_pred6)
precision_score(test_y, y_pred6)
recall_score(test_y, y_pred6)
f1_score(test_y, y_pred6)

final_data = pd.DataFrame({"Models": ["LR", "SVC", "KNN", "DT", "RF", "GB"],
                           "ACC":[accuracy_score(test_y, y_pred1),
                                  accuracy_score(test_y, y_pred2),
                                  accuracy_score(test_y, y_pred3),
                                  accuracy_score(test_y, y_pred4),
                                  accuracy_score(test_y, y_pred5),
                                  accuracy_score(test_y, y_pred6)]})
#final_data
#import seaborn as sns
#sns.barplot(final_data["Models"],final_data["ACC"])
#st.set_option('deprecation.showPyplotGlobalUse', False)
#pyplot()

final_data = pd.DataFrame({"models": ["LR", "SVC", "KNN", "DT", "RF", "GB"],
                           "PRE":[precision_score(test_y, y_pred1),
                                  precision_score(test_y, y_pred2),
                                  precision_score(test_y, y_pred3),
                                  precision_score(test_y, y_pred4),
                                  precision_score(test_y, y_pred5),
                                  precision_score(test_y, y_pred6)]})
#final_data

final_data = pd.DataFrame({"models": ["LR", "SVC", "KNN", "DT", "RF", "GB"],
                           "Recall":[recall_score(test_y, y_pred1),
                                  recall_score(test_y, y_pred2),
                                  recall_score(test_y, y_pred3),
                                  recall_score(test_y, y_pred4),
                                  recall_score(test_y, y_pred5),
                                  recall_score(test_y, y_pred6)]})
#final_data

final_data = pd.DataFrame({"models": ["LR", "SVC", "KNN", "DT", "RF", "GB"],
                           "F1":[f1_score(test_y, y_pred1),
                                  f1_score(test_y, y_pred2),
                                  f1_score(test_y, y_pred3),
                                  f1_score(test_y, y_pred4),
                                  f1_score(test_y, y_pred5),
                                  f1_score(test_y, y_pred6)]})
#final_data

#st.subheader("Save the Best Model - Random Forest Classifier")

X_res = sc.fit_transform(X_res)
model = rf.fit(X_res, y_res)

#data.columns

#model.predict([[619,42,2,0.0,0,0,0,101348.88,0,0,0]])
st.title("Banking Churn Prediction")
CreditScore = st.text_input("Credit Score")
Age = st.text_input("Age")
Tenure = st.text_input("Tenure")
Balance = st.text_input("Balance")
NumofProducts = st.text_input("Num of Products")
HasCrCard = st.text_input("Has Cr Card")
IsActiveMember = st.text_input("Is Active Member")
EstimatedSalary = st.text_input("Estimated Salary")
Geography = st.text_input("Geography")
Gender = st.text_input("Gender")
st.button("Predict")
