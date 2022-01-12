#!/usr/bin/env python
# coding: utf-8

# (20%) Name your Jupyter notebook sin_approx and Python script sin_approx.py. Using a
# while loop to implement an approximation of sine function with polynomial:
# sin(𝑥) ≈ 𝑥 − 𝑥**3/3! + 𝑥**5/5!−𝑥**7/7!+ ⋯
# Compute the right-hand side for powers up to 𝑁 = 25. 
# Below is a sample output:
# sin(1.6) approximation is 0.9995736030415051

# In[38]:


sum = 0
m = 1.6
i = 1   # 索引變數
while i<= 25:
    if i == 1 or i ==5 or i ==9 or i ==13 or i ==17 or i ==21 or i ==25:
        sum +=  m ** i / factorial(i)
    else:
        sum += -(m ** i) / factorial(i)
    i += 2   
print(sum)


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
