#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install missingno')
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import pie, axis, show


# In[5]:


df = pd.read_csv(r"C:\Users\mahit\Downloads\archive\bigml_59c28831336c6604c800002a.csv")


# In[6]:


df.head(10)


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.dtypes


# In[10]:


df['area code'] = df['area code'].astype('object')
df['churn']=df['churn'].astype('int')
df['international plan'] = df['international plan'].astype('category')
df.describe(include=['O'])


# In[11]:


df.drop(["phone number"], axis = 1,)


# In[12]:


df.isnull().sum() #data cleaning


# In[13]:


df.churn.value_counts()


# In[14]:


a=sns.countplot(x='churn',data=df)


# In[15]:


a=sns.countplot(x='international plan',data=df)


# In[16]:


a=sns.countplot(x='voice mail plan',data=df)


# In[17]:


# In[13]:


churn_percentage = df["churn"].sum() * 100 / df["churn"].shape[0]
print("Churn percentage is %.3f%%." % churn_percentage)
#Churn percentage is 14.491%.


# In[18]:


plt.figure(figsize=(14,8))
sns.countplot(x='customer service calls',hue = 'churn',data=df,palette='Set1' )


# In[19]:


plt.figure(figsize=(14,8))
sns.countplot(x ='international plan',hue='churn',data = df,palette='Set2')


# In[20]:


sns.countplot(df['total day minutes'])


# In[21]:


features = df[['account length','international plan','voice mail plan','number vmail messages','total day calls','total day charge','total eve calls','total eve charge',
      'total night calls','total night charge','total intl calls','total intl charge','customer service calls']]

target = df[['churn']]

target.head()


# In[22]:




# In[14]:


plt.figure(figsize=(14,8))
sns.countplot(x ='international plan',hue='churn',data = df,palette='Set2')


# In[15]:


features = df[['account length','international plan','voice mail plan','number vmail messages','total day calls','total day charge','total eve calls','total eve charge',
      'total night calls','total night charge','total intl calls','total intl charge','customer service calls']]


# In[16]:


target = df[['churn']]


# In[17]:


target.head()


# In[18]:


features = pd.get_dummies(features,columns=['international plan','voice mail plan'],drop_first=True) ## creating dummy variables for categorical 


# In[19]:


features.head()


# In[33]:


Log_clf.intercept_


# In[34]:


Log_clf.coef_ 


# In[35]:


## prediction on Train data
Log_Train_Pred = Log_clf.predict(features_train)


# In[36]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[37]:


print(classification_report(target_train,Log_Train_Pred))


# In[38]:


Log_Test_Pred = Log_clf.predict(features_test)


# In[39]:


print(classification_report(target_test,Log_Test_Pred))


# In[40]:


from sklearn.svm import SVC
classifier =SVC(kernel='rbf',random_state=1, gamma='auto')
classifier.fit(features_train,target_train)


# In[41]:


svm_pred=classifier.predict(features_test)


# In[42]:


accuracy_score(svm_pred,target_test)


# In[43]:


from sklearn.ensemble import RandomForestClassifier


# In[44]:


RFclf = RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_leaf=5,class_weight='balanced',random_state=42)


# In[32]:


RFclf.fit(features_train,target_train)


# In[33]:


RFclf_Train_pred=RFclf.predict(features_train)


# In[34]:


RFclf_Train_pred


# In[ ]:





# In[35]:


RFclf_Test_pred=RFclf.predict(features_test)


# In[36]:


ac=accuracy_score(RFclf_Test_pred,target_test)*100


# In[37]:


print("Accuracy is :")
ac




# In[2]:


import tensorflow as tf #pip install tensorflow
from tensorflow import keras

#syntax =  keras.layers.Dense(output_dimension, input_shape, activation)

model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'), #Dense means all the neurons of present are connected to all the neurons in next layer
    keras.layers.Dense(15, activation='relu'), #no need to mention the input dimension,it will be taken from the previous layer
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy', #because we have binary(2) classes to predict
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100) #start slow with epochs and then increase the value


# In[ ]:




