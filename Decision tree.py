#!/usr/bin/env python
# coding: utf-8

# 
# # Decision tree for disease prediction
# 

# In[132]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_score,recall_score,f1_score
import warnings
warnings.filterwarnings('ignore')


# In[133]:


df = pd.read_csv(r'D:\Dowloads\Testing.csv',delimiter =",")


# In[134]:


df.isnull().sum()


# In[135]:


df.head(10)


# In[136]:


df.drop(["prognosis"],axis=1,inplace=True)
df.head(10)


# In[137]:


data = df.to_numpy()


# In[138]:


clf = DecisionTreeClassifier()


# In[139]:


X = data[:,:-1]

Y = data[:,-1]
X


# In[140]:


from sklearn import model_selection
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size = 0.25,random_state = 0)


# In[141]:


clf.fit(X_train,Y_train)


# In[142]:


Y_pred = clf.predict(X_test)


# In[143]:


print("Confusion matrix is:")
print(confusion_matrix(Y_test,Y_pred))
print("\nAccuracy score is:")
print(accuracy_score(Y_test,Y_pred))
print("\nClassification report is: ")
print(classification_report(Y_test,Y_pred))
print("\nPrecision score is:")
print(precision_score(Y_test,Y_pred))
print("\nRecall score is:")
print(recall_score(Y_test,Y_pred))
print("\nF1 score is:")
print(f1_score(Y_test,Y_pred))

