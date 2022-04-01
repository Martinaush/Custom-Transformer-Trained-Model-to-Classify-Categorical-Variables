#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#########################################
# Custom Transformer

# It is used in the main pipeline
#########################################

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        pass
        
    def fit(self, X, y = None):
        
        return self
        
    def transform(self, X, y = None):
        
        print("""################################################\n
# Prepocessing................\n
# Dealing with Infinite and NAs values\n
################################################""")
        
        # NAs to zero
        print("There were %d NAs"%X.isna().sum().sum())
        X = X.fillna(0)
        
        # Checking infinite values
        print("There were %d cells whose value were infinty"%np.isinf(X).values.sum(), end = "\n\n")

        # Converting Inf values to NAs
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Now since they were infinity values, these NAs will be replaced with the double of maximum
        # of the same column
        X = X.apply(lambda row: row.replace(np.nan, 2*max(row)), axis=1)
        
        return X

