#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
We are going to predict the closing stock price of the day of a corporetion using the past 60 days stock prices.
the coperation will be facebook here.
'''


# In[1]:


#importing all the needed libraries
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import quandl
from sklearn.model_selection import train_test_split


# In[2]:


#fetching the data
df=quandl.get("WIKI/FB")
print(df.head())


# In[3]:


#get the adjusted price
df=df[['Adj. Close']]
print(df.head())


# In[10]:


# no of days we wnt to predict in the future(n)
forecast_out=30
#we will use the shift method to predict the price in the future
#creating the new column shifted n units up
df["Prediction"]=df[['Adj. Close']].shift(-forecast_out)
#printing the new dataset
print(df.head())
print(df.tail())


# In[13]:


# creating the independent dataset(X)
X=np.array(df.drop(['Prediction'],1))
#remove the ;ast n rows
X=X[:-forecast_out]
print(X)


# In[15]:


#creating  the dependent dataset(y)
# converting the dataframe to numpy array of all the values including the nan values
y=np.array(df["Prediction"])
# get all the y values except the last n rows
y=y[:-forecast_out]
print(y)


# In[18]:


#spliting the data in train and test set 20%
x_train, x_test, y_train, y_test=train_test_split(X,y, test_size=0.2)


# In[20]:


# create and train the SVM(REgression)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_rbf.fit(x_train,y_train)


# In[21]:


#Teting the model(R squared value)
svm_confidence=svr_rbf.score(x_test, y_test)
print("svm confidence: ",svm_confidence)


# In[22]:


# create and train the linear regression model
lr= LinearRegression()
lr.fit(x_train, y_train)


# In[24]:


#Teting the model(R squared value)
lr_confidence=lr.score(x_test, y_test)
print("lr confidence: ",lr_confidence)


# In[25]:


# setx_forecast= equals the last 30 rows from the adj. close
x_forecast= np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)


# In[27]:


# print the prediction for the next 30 days(linear regression)
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)
print("\n")
# print the prediction for the next 30 days(SVM)
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)


# In[ ]:




