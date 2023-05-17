import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import pickle

# Read the dataframes from the saved CSV files in part 1
df1 = pd.read_csv('collocations1.csv')
df3 = pd.read_csv('collocations3.csv')

# Load word frequencies from disk
with open('word_frequencies.pkl', 'rb') as f: word_freqs = pickle.load(f)
# Get total number of words
N = word_freqs.N()

df1 = df1.sort_values(by='Bigram_Freq', ascending=False)
df3 = df3.sort_values(by='Bigram_Freq', ascending=False)

print('-------------------------------------------------------')
print('\tWindow Size 1 Frequencies')
print('-------------------------------------------------------')
print(df1.head(20))
print('-------------------------------------------------------')
print("\tWindow Size 3 Frequencies")
print('-------------------------------------------------------')
print(df3.head(20))

window_size = 1

def t_score(row):
    word1_freq = row['Word1_Freq']
    word2_freq = row['Word2_Freq']
    bigram_freq = row['Bigram_Freq']
    # expected_freq = (word1_freq * word2_freq) / (window_size * N)
    expected_freq = (word1_freq * word2_freq) / total_words
    t_score = (bigram_freq - expected_freq) / np.sqrt(bigram_freq)
    return t_score

def chi_square(row):
    word1_freq = row['Word1_Freq']
    word2_freq = row['Word2_Freq']
    bigram_freq = row['Bigram_Freq']
    # expected_freq = (word1_freq * word2_freq) / (window_size * N)
    expected_freq = (word1_freq * word2_freq) / total_words
    # return window_size * N * (bigram_freq - expected_freq)**2 / (word1_freq * word2_freq)
    return total_words * (bigram_freq - expected_freq)**2 / (word1_freq * word2_freq)

from scipy.stats import binom

def likelihood_ratio(row):
    # Constants
    # SMALL_PROB = 1e-10
    SMALL_PROB = np.finfo(float).eps

    # Calculate probabilities
    p1 = max(row['Bigram_Freq'] / row['Word1_Freq'], SMALL_PROB)
    # p2 = max(row['Word2_Freq'] / row['Bigram_Freq'], SMALL_PROB)
    # p2 = max(row['Word2_Freq'] / (window_size * N), SMALL_PROB)
    p2 = max(row['Word2_Freq'] / total_words, SMALL_PROB)

    # Calculate likelihoods for the null and alternative hypotheses
    L_null = binom.pmf(row['Bigram_Freq'], row['Word1_Freq'], p2)
    L_alt = binom.pmf(row['Bigram_Freq'], row['Word1_Freq'], p1)

    # Avoid division by zero
    if L_null == 0:
        L_null = SMALL_PROB

    # Calculate and return likelihood ratio test statistic
    return -2 * np.log(L_null / L_alt)

# from scipy import stats

# Add t-score, chi-square, and likelihood ratio columns to the DataFrames
window_size = 1
total_words = df1['Word1_Freq'].sum() + df1['Word2_Freq'].sum()
df1['t_score'] = df1.apply(t_score, axis=1)
df1['chi_square'] = df1.apply(chi_square, axis=1)
df1['likelihood_ratio'] = df1.apply(likelihood_ratio, axis=1)

window_size = 3
total_words = df3['Word1_Freq'].sum() + df3['Word2_Freq'].sum()
df3['t_score'] = df3.apply(t_score, axis=1)
df3['chi_square'] = df3.apply(chi_square, axis=1)
df3['likelihood_ratio'] = df3.apply(likelihood_ratio, axis=1)

# Convert scientific numbers format to numbers
df1['t_score'] = df1['t_score'].astype(float)
df1['chi_square'] = df1['chi_square'].astype(float)
df1['likelihood_ratio'] = df1['likelihood_ratio'].astype(float)
df3['t_score'] = df3['t_score'].astype(float)
df3['chi_square'] = df3['chi_square'].astype(float)
df3['likelihood_ratio'] = df3['likelihood_ratio'].astype(float)

pd.set_option('display.float_format', '{:.5f}'.format)

df1.to_csv('collocations1_scores.csv', index=False)
df3.to_csv('collocations3_scores.csv', index=False)

# Sort by scores and print the top 20 candidates
print('-------------------------------------------------------')
print('\tWindow Size 1 - top 20 t-scores')
print('-------------------------------------------------------')
print(df1.sort_values('t_score', ascending=False).head(20))
print('-------------------------------------------------------')
print('\tWindow Size 1 - top 20 chi-square')
print('-------------------------------------------------------')
print(df1.sort_values('chi_square', ascending=False).head(20))
print('-------------------------------------------------------')
print('\tWindow Size 1 - top 20 likelihood-ratio')
print('-------------------------------------------------------')
print(df1.sort_values('likelihood_ratio', ascending=False).head(20))

print('-------------------------------------------------------')
print('\tWindow Size 3 - top 20 t-scores')
print('-------------------------------------------------------')
print(df3.sort_values('t_score', ascending=False).head(20))
print('-------------------------------------------------------')
print('\tWindow Size 3 - top 20 chi-square')
print('-------------------------------------------------------')
print(df3.sort_values('chi_square', ascending=False).head(20))
print('-------------------------------------------------------')
print('\tWindow Size 3 - top 20 likelihood-ratio')
print('-------------------------------------------------------')
print(df3.sort_values('likelihood_ratio', ascending=False).head(20))
