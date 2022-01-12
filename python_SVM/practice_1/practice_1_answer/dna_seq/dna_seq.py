#!/usr/bin/env python
# coding: utf-8

# (20%) Name your Jupyter notebook dna_seq and Python script dna_seq.py. A DNA sequence
# contains four base letters A, C, G, T. Genomes of different species vary with respect to the
# proportion of Gs and Cs in their DNA as opposed to Ts and As.
# Write a function gc_content that returns the GC content of a given DNA sequence represented
# as a string. Use the following simple script and sample outputs to test your function:

# In[23]:


def gc_content(seq):
    total = len(seq)
    c = seq.count("C")
    g = seq.count("G")
    gcsum =  (c + g)/ total
    print(gcsum)


# In[25]:


seq50 = "GGAACCTT"
gc_content(seq50)


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
