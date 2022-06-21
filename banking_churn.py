# importing required values
#%matplotlib inline
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
banking_churn = pd.read_csv("https://raw.githubusercontent.com/Livuza/ADS-April-2021/main/Assignments/Assignment%202/banking_churn.csv")
st.write(banking_churn.head())
st.write(banking_churn.describe())
st.write("Drop columns with no effect on whether customer will churn (RowNumber, CustomerId,Surname)")
banking_churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace = True)
st.write(banking_churn.head())
st.write(banking_churn.describe())
st.write("Count per Geography")
st.write(banking_churn["Geography"].value_counts())

st.write("Effect of Gender on customer churn")
counts = banking_churn.groupby(['Gender', 'Exited']).Exited.count().unstack()
counts.plot(kind='bar', stacked=True)
st.pyplot()
st.write(counts)

st.write("Effect of Geography on customer churn")
counts = banking_churn.groupby(['Geography', 'Exited']).Exited.count().unstack()
counts.plot(kind='bar', stacked=True)
st.pyplot()
st.write(counts)

st.write("Remove the categorical columns Geography and Gender")
tempdata = banking_churn.drop(['Geography','Gender'], axis=1)
st.write(tempdata.head(2))

st.write("Create one hot encoded columns for Geography and Gender")
Geography = pd.get_dummies(banking_churn.Geography).iloc[:,1:]
Gender = pd.get_dummies(banking_churn.Gender).iloc[:,1:]
banking_churn = pd.concat([tempdata,Geography,Gender], axis=1)
st.write(banking_churn.head(2))

train_X = banking_churn.drop(['Exited'], axis=1)
train_Y = banking_churn['Exited']
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)
st.write("shape of train and test splits")
train_x.shape, test_x.shape, train_y.shape, test_y.shape
st.subheader("Train and Evaluation of Machine Learning Models")
st.write("Training the model using Random Forest algorithm:")
from sklearn.ensemble import RandomForestClassifier as rfc

rfc_object = rfc(n_estimators=200, random_state=0)
rfc_object.fit(train_x, train_y)
predicted_labels = rfc_object.predict(test_x)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
st.write((classification_report(test_y, predicted_labels)))
st.write((confusion_matrix(test_y, predicted_labels)))
st.write((accuracy_score(test_y, predicted_labels)))
from sklearn.svm import SVC as svc

st.write("Train the model using support vector machines:")
svc_object = svc(kernel='rbf', degree=8)
svc_object.fit(train_x, train_y)
predicted_labels = svc_object.predict(test_x)
st.write((classification_report(test_y, predicted_labels)))
st.write((confusion_matrix(test_y, predicted_labels)))
st.write((accuracy_score(test_y, predicted_labels)))

st.write("Train the model using logistic regression:")
from sklearn.linear_model import LogisticRegression
lr_object = LogisticRegression()
lr_object.fit(train_x, train_y)
predicted_labels = lr_object.predict(test_x)
st.write((classification_report(test_y, predicted_labels)))
st.write((confusion_matrix(test_y, predicted_labels)))
st.write((accuracy_score(test_y, predicted_labels)))


st.sidebar.title("Customer Bank Churn App")
