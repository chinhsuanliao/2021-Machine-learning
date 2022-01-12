#!/usr/bin/env python
# coding: utf-8

# In[1]:


def gcd(m,n):
    while n:
        r = m%n
        m = n
        n = r
    return m


# In[5]:


class Fraction(object):
    
    #class 的第一個function一定要給self
    #__init__ initial state of the function
    def __init__(self,numerator,denominator):
        self.top = numerator
        self.bottom = denominator
        
    #  __str__  for print function.產生可以印出來的東西
    def __str__(self):
        return str(self.top)+ "/" + str(self.bottom)
    
    ##以下為Magic Methods for comparison，利用__str__ display##
    
    def __add__(self,other):
        top = self.top*other.bottom + self.bottom* other.top 
        bottom = other.bottom * self.bottom
        reduction= gcd(top,bottom)
        return Fraction(top//reduction,bottom//reduction)
    
    def __sub__(self,other):
        top = self.top*other.bottom - self.bottom* other.top
        bottom = other.bottom * self.bottom
        reduction= gcd(top,bottom)
        return Fraction(top//reduction,bottom//reduction)
    
    def __mul__(self,other):
        top = self.top * otherfraction.top
        bottom = self.bottom * other.bottom
        reduction = gcd(top, bottom)
        return Fraction(top//reduction, bottom//reduction)
    
    def __floordiv__(self,other):#self//other
        top = self.top * other.bottom
        bottom = self.bottom * other.top
        reduction = gcd(top, bottom)
        return Fraction(top//reduction, bottom//reduction)
    
    #利用交叉相乘法進行比較
    def __ne__(self, other):
        num1 = self.top* other.bottom
        num2 = other.top * self.bottom
        return num1 != num2
            
    def __eq__(self, other):
        num1 = self.top* other.bottom
        num2 = other.top * self.bottom
        return num1 != num2
    


# Class的功能:
# 1. 把功能分類切成一塊一塊:物件結構
# 2. 抽象化: 把function藏起來
# 
# Reference:
# 1.老師上課的影片
# 2.https://weilihmen.medium.com/%E9%97%9C%E6%96%BCpython%E7%9A%84%E9%A1%9E%E5%88%A5-class-%E5%9F%BA%E6%9C%AC%E7%AF%87-5468812c58f2

# In[ ]:





# In[3]:


5//2


# In[4]:


5/2


# In[ ]:




