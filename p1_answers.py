import numpy as np
import pandas as pd
import pickle

# Read the dataframes from the saved CSV files in part 1 without eliminating below 10 occurrences
df1 = pd.read_csv('collocations1_all.csv')
df3 = pd.read_csv('collocations3_all.csv')

# PART 1.b
# Load word frequencies from disk
with open('word_frequencies.pkl', 'rb') as f: word_freqs = pickle.load(f)
# Get total number of words
N = word_freqs.N()
print('-------------------------------------------------------')
print('Total number of words in the corpus: {}'.format(N))
print('-------------------------------------------------------')

# Helper methods to answer part 1 questions
def find_word_freq(freq_dic, word):
    freq = freq_dic[word]
    return freq

def find_bigram_freq(df, word1, word2):
    row = df.loc[(df['Word1'] == word1) & (df['Word2'] == word2)]
    if row.empty: return None
    return row['Bigram_Freq'].values[0]

# PART 1.d
words = ['that', 'the', 'abject', 'london', '.']
for word in words:
    print('Word: {} - Frequency: {}'.format(word, find_word_freq(word_freqs, word)))

word1 = 'magnificent'
word2 = 'capital'
print('Word1: {} - Word2: {} - Frequency: {}'.format(word1, word2, find_bigram_freq(df1, word1, word2)))

word1 = 'bright'
word2 = 'fire'
print('Word1: {} - Word2: {} - Frequency: {}'.format(word1, word2, find_bigram_freq(df3, word1, word2)))

word1 = 'mr.'
word2 = 'skimpole'
print('Word1: {} - Word2: {} - Frequency: {}'.format(word1, word2, find_bigram_freq(df1, word1, word2)))

word1 = 'spontaneous'
word2 = 'combustion'
print('Word1: {} - Word2: {} - Frequency: {}'.format(word1, word2, find_bigram_freq(df3, word1, word2)))
