# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 22:29:27 2016

@author: vijay
"""

import numpy as np
import pandas as pd

res=np.load('results_adam.npy')

columns = ['ImageId','Label']
df = pd.DataFrame(index=range(0,len(res)), columns=columns)

for ind,pred in enumerate(res):
    
    df['ImageId'][ind]=ind+1
    df['Label'][ind]=pred
    

df.to_csv('results.csv',index=False)
print df