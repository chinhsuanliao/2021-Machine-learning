#!/usr/bin/env python
# coding: utf-8

# In[1]:


def gcd(m,n):
    while n:
        r = m%n
        m = n
        n = r
    return m


# In[ ]:




