# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:32:46 2018

@author: parla
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
#%%
metu = pd.read_csv("Project (Application 1) (MetuSabanci Treebank).conll", delimiter=r"\s+", header = None)

#%%
# Detect the rows whose 2nd column are _.
metu_with_underscore = metu[1] != '_'
# Remove the corresponding rows
metu_without_underscore = metu[metu_with_underscore]
# Retrieve the 2nd and 4th columns
dataset = metu_without_underscore[[1,3]]
# Rename columns
dataset.columns = ['word', 'pos']

#%%
# Make a copy of original dataset
temp = dataset.copy()
# Add a column named x, and label as True if the current row at the word column is .
temp['x'] = temp['word'] == '.'
# Add a y column named y, and calculate the cumulative sum according to True labels in the x column.
temp['y'] = temp['x'].cumsum()
# Removes .'s from set.
temp = temp[temp['word'] != '.']
# Group y column and concatenate words which have the same count and add . to the end.
words = temp.groupby('y').apply(lambda x: ' '.join(x['word'].astype(str))+' .')
# Group y column and concatenate pos tags which have the same count.
postags = temp.groupby('y').apply(lambda x: ' '.join(x['pos'].astype(str)))
# Concatenate words and pos tags to form a new dataframe.
sentece_base_dataset = pd.concat([words, postags], axis=1)
# Rename columns
sentece_base_dataset.columns = ['sentence', 'postags']

#%%
X_train, X_test = train_test_split(sentece_base_dataset, test_size=0.1, random_state=1)
