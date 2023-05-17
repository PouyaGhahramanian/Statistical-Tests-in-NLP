
import numpy as np
import pandas as pd
import pickle

df1 = pd.read_csv('collocations1.csv')
df3 = pd.read_csv('collocations3.csv')

df1_scores = pd.read_csv('collocations1_scores.csv')

word1, word2 = 'head', 'clerk'
# Call the function for a specific bigram
print('-------------------------------------------------------')
print('\tScores for words: {} and {}'.format(word1, word2))
print('-------------------------------------------------------')
print(df1_scores.loc[(df1_scores['Word1'] == word1) & (df1_scores['Word2'] == word2)])

word1, word2 = 'great', 'man'
print('-------------------------------------------------------')
print('\tScores for words: {} and {}'.format(word1, word2))
print('-------------------------------------------------------')
print(df1_scores.loc[(df1_scores['Word1'] == word1) & (df1_scores['Word2'] == word2)])
