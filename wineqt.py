#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\\Users\\Abhi\\documents\\readings')


# In[4]:


df=pd.read_csv('WineQT.csv')


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[17]:


# fill the missing values
for col,value in df.items():
    if col!='type':
       df[col]=df[col].fillna(df[col].mean())


# In[18]:


df.isnull().sum()


# # EDA

# In[30]:


# CREATE BOXPLOT 
fig,ax=plt.subplots(ncols=6,nrows=2,figsize=(20,10))
index=0
ax=ax.flatten()

for col,value in df.items():
    if col!='type':
     sns.boxplot(data=df,y=col,ax=ax[index])
     index +=1
plt.tight_layout(pad=0.5,w_pad=0.7,h_pad=5.0)


# In[15]:


sns.distplot(df['free sulfur dioxide'])


# In[21]:


sns.countplot(x=df['quality'])


# # Coorelation Matrix

# In[33]:


corr=df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()


# # Input Split

# In[41]:


import smote
import imblearn
from imblearn.over_sampling import SMOTE


# In[34]:


x=df.drop(columns=['density','quality'])
y=df['quality']


# # Class Imbalancement

# In[43]:


from imblearn.over_sampling import SMOTE
oversample =SMOTE()
# TRANSFORM THE DATA
x,y=oversample.fit_resample(x,y)


# In[45]:


y.value_counts()


# # Model Traning

# In[59]:


import sklearn
from sklearn import model_selection


# In[60]:


# classify function 
from sklearn.model_selction import cross_val_score, train_test_split
def classify(model,x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
    #train the model
    print("Accuracy:",model.score(x_test,y_test)*100)

    # cross-validation
    score=cross_val_score(model,x,y,cv=5)
    print("CV Score",np.mean(score)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




