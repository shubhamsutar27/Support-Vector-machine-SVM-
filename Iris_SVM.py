#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[55]:


iris = pd.read_csv('Iris.csv')
iris.head()


# In[56]:


# Dropping id from above features
iris.drop('Id',axis=1,inplace=True)


# In[57]:


iris


# In[58]:


# Creating a pairplot to visualize the similarities and difference between the species.
sns.pairplot(data =iris,hue='Species',palette='Set1')


# In[59]:


sns.barplot(x="Species",y="SepalLengthCm",data=iris).set_title("Sepal Length of three species")


# In[60]:


sns.scatterplot(x=iris.SepalLengthCm,y=iris.SepalWidthCm,hue=iris.Species).set_title("Sepal length and Sepal width distribution of three flowers")


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


x = iris.iloc[:,:-1]
y = iris.iloc[:,4]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[63]:


y


# ## Training and Fitting the model

# In[64]:


from sklearn.svm import SVC
model=SVC()


# In[65]:


model.fit(x_train,y_train)


# ## Predicting from the trained model

# In[66]:


pred = model.predict(x_test)


# ## Model Evaluation 

# In[67]:


# Import the classification report and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix


# In[68]:


print(confusion_matrix(y_test,pred))


# In[69]:


print(classification_report(y_test, pred))

