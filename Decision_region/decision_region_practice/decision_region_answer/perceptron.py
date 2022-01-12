#!/usr/bin/env python
# coding: utf-8

# 1. (120%) Name your Jupyter notebook Perceptron.ipynb and Python script preceptron.py. As we mentioned in lecture last Friday, the learning rule can be expressed as
# 𝒘←𝒘+𝜂∙(𝑦−𝑑(𝑘))∙𝒙(𝑘)
# where
# 𝒘=[𝜃𝑤1𝑤2⋯𝑤𝑛] is the vector containing the threshold and weights;
# 𝒙(𝑘)=[−1𝑥1(𝑘)𝑥2(𝑘)⋯𝑥𝑛(𝑘)] is the 𝑘𝑡ℎ training sample;
# 𝑑(𝑘) is the desired value for the 𝑘𝑡ℎ training sample;
# 𝜂 is a constant that defines the learning rate of the Perceptron.

# (a) (80%)Please finish the fit function we provide in Perception.py file and test the code using the following code fragment

# In[2]:


import numpy as np
class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.

    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of
        examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values.

        Returns
        -------
        self : object
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []
        # W current = W previous- eta(y-d(k))x(k)
        j = 0
        while 1:
            
            if j == self.n_iter:
                break
                
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0:] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            
            j += 1
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

