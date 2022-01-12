#!/usr/bin/env python
# coding: utf-8

# Bank of LAiMM wants to predict who are possible defaulters for the consumer loans product.
# They have data about historic customer behavior based on what they have observed. Hence when
# they acquire new customers, they want to predict who is riskier and who is not. You are required
# to use the training dataset to identify patterns that predict “potential” defaulters. Name your
# Jupyter notebook SVM and report SVM_report.docx.

# (a) (20%) Plot the statics of training dataset. You may need to process training data before feeding
# them into model.
# This figure is an example, please plot the statistics of training data that are fed into the model.

# In[4]:


import os
os.getcwd()


# (b) (40%) Use Support Vector Machine to identify possible defaulters, please report training
# history, confusion matrix and accuracy. 

# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('training.csv')


# In[ ]:




