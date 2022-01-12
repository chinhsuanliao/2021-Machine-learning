#!/usr/bin/env python
# coding: utf-8

# 1. (20%) Name your Jupyter notebook cm2ftin_table and Python script cm2ftin _table.py.
# Write a Python program that prints out a table with centimeter 145, 150, 155, …, 210 in the first
# column and the corresponding feet and inches in the second column. Below is a sample output:
# cm ft in
# 145, 4.0, 9.086614114999996
# 150, 4.0, 11.055118049999997
# 155, 5.0, 1.0236219849999983
# 160, 5.0, 2.9921259199999994
# 165, 5.0, 4.960629854999993
# 170, 5.0, 6.9291337899999945
# 175, 5.0, 8.897637724999996
# 180, 5.0, 10.866141659999997
# 185, 6.0, 0.8346455949999978
# 190, 6.0, 2.803149529999999
# 195, 6.0, 4.771653465
# 200, 6.0, 6.740157400000001
# 205, 6.0, 8.708661335000002
# 210, 6.0, 10.677165270000003

# In[49]:


import math
print ("cm   ft   in ")
 
cm = 0 
ft = 0
inch = 0

for i in range(0, 14):
        cm = 145 + i*5 
        ft = cm * 0.032808399
        ft = float(math.floor(ft))
        inch = cm * 0.3937
        cm = str(cm) + ","
        ft = str(ft) + ","

        print (cm,ft,inch)


# (20%) Name your Jupyter notebook sin_approx and Python script sin_approx.py. Using a
# while loop to implement an approximation of sine function with polynomial:
# sin(𝑥) ≈ 𝑥 − 𝑥**3/3! + 𝑥**5/5!−𝑥**7/7!+ ⋯
# Compute the right-hand side for powers up to 𝑁 = 25. Below is a sample output:
# sin(1.6) approximation is 0.9995736030415051

# In[ ]:





# (20%) Name your Jupyter notebook dna_seq and Python script dna_seq.py. A DNA sequence
# contains four base letters A, C, G, T. Genomes of different species vary with respect to the
# proportion of Gs and Cs in their DNA as opposed to Ts and As.
# Write a function gc_content that returns the GC content of a given DNA sequence represented
# as a string. Use the following simple script and sample outputs to test your function:

# In[ ]:





# Bank of LAiMM wants to predict who are possible defaulters for the consumer loans product.
# They have data about historic customer behavior based on what they have observed. Hence when
# they acquire new customers, they want to predict who is riskier and who is not. You are required
# to use the training dataset to identify patterns that predict “potential” defaulters. Name your
# Jupyter notebook SVM and report SVM_report.docx.

# (a) (20%) Plot the statics of training dataset. You may need to process training data before feeding
# them into model.
# This figure is an example, please plot the statistics of training data that are fed into the model.

# In[ ]:





# (b) (40%) Use Support Vector Machine to identify possible defaulters, please report training
# history, confusion matrix and accuracy. 
