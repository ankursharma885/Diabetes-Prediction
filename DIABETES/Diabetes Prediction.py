#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# DATA COLLECTION

# In[8]:


diabetes_dataset =pd.read_csv(r'C:\Users\omkar\OneDrive\Desktop\DIABETES\diabetes.csv')


# In[9]:


diabetes_dataset


# In[10]:


diabetes_dataset.head()


# In[11]:


diabetes_dataset.shape


# In[12]:


# statistical measures of the data


# In[13]:


diabetes_dataset.describe()


# In[15]:


diabetes_dataset.corr()


# In[18]:


diabetes_dataset['Outcome'].value_counts()


# 0 ---> Non diabetic
# 1 ---> diabetic

# In[19]:


diabetes_dataset.groupby('Outcome').mean()


# In[20]:


x=diabetes_dataset.drop(columns ='Outcome',axis=1)
y=diabetes_dataset['Outcome']


# In[21]:


print(x)


# In[22]:


print(y)


# In[46]:


scaler = StandardScaler()


# In[52]:


standardized_data = scaler.fit_transform(x)


# In[53]:


standardized_data


# In[54]:


X=standardized_data


# In[55]:


print(X)
print(y)


# In[56]:


X_train, X_test,y_train, y_test = train_test_split(X,y, test_size =0.2, stratify=y,random_state=2)


# In[60]:


print(X.shape, X_train.shape, X_test.shape)


# Training the model

# In[62]:


classifier = svm.SVC(kernel='linear')


# In[63]:


classifier.fit(X_train,y_train)


# Model Evaluation

# accuracy score

# In[68]:


# Accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction,y_train)


# In[69]:


print('Accuracy score of the training data:',training_data_accuracy)


# In[70]:


# accuracy score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction,y_test)


# In[71]:


print('Accuracy score of the training data:',test_data_accuracy)


# In[72]:


# MAKING A PREDICTIVE SYSTEM


# In[78]:


input_data = (10,139,80,0,0,27.1,1.441,57)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')


# In[ ]:




