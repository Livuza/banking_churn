# importing required values
#%matplotlib inline
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
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
# read the train data
train_data = pd.read_csv('https://raw.githubusercontent.com/Livuza/ADS-April-2021/main/Assignments/Assignment%202/banking_churn.csv')

# check for the null values
st.write("check for the null values")
st.write(train_data.isna().sum())
import category_encoders as ce
# create an object of the OneHotEncoder
OHE = ce.OneHotEncoder(cols=['Geography',
                             'Gender',],use_cat_names=True)
# encode the categorical variables
train_data = OHE.fit_transform(train_data)
#st.write(train_data)
from sklearn.preprocessing import StandardScaler
# create an object of the StandardScaler
scaler = StandardScaler()
# fit with the Geography
scaler.fit(np.array(train_data.Geography_Spain).reshape(-1,1))
# transform the data
train_data.Geography_Spain = scaler.transform(np.array(train_data.Geography_Spain).reshape(-1,1))

# importing libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# separate the independent and target variable
train_X = train_data.drop(columns=['Gender_Female','Gender_Male'])
train_Y = train_data['Gender_Male']

# randomly split the data
train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y,test_size=0.2,random_state=42)

# shape of train and test splits
train_x.shape, test_x.shape, train_y.shape, test_y.shape

# create an object of the LinearRegression Model
#model_LR = LinearRegression()

# fit the model with the training data
#model_LR.fit(train_x, train_y)

# predict the target on train and test data
#predict_train = model_LR.predict(train_x)
#predict_test  = model_LR.predict(test_x)

# Root Mean Squared Error on train and test date
#print('RMSE on train data: ', mean_squared_error(train_y, predict_train)**(0.5))
#print('RMSE on test data: ',  mean_squared_error(test_y, predict_test)**(0.5))

# create an object of the RandomForestRegressor
#model_RFR = RandomForestRegressor(max_depth=10)

# fit the model with the training data
#model_RFR.fit(train_x, train_y)

# predict the target on train and test data
#predict_train = model_RFR.predict(train_x)
#predict_test = model_RFR.predict(test_x)

# Root Mean Squared Error on train and test data
#print('RMSE on train data: ', mean_squared_error(train_y, predict_train)**(0.5))
#print('RMSE on test data: ',  mean_squared_error(test_y, predict_test)**(0.5))

# plot the 7 most important features
#plt.figure(figsize=(10,7))
#feat_importances = pd.Series(model_RFR.feature_importances_, index = train_x.columns)
#feat_importances.nlargest(7).plot(kind='barh');
