
import numpy as np
import pandas as pd
import string
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist
import pickle

# Read Text Corpus
addr = 'Student Release/Fyodor Dostoyevski Processed.txt'
with open(addr, 'r') as file:
    text = file.read()

# Download Sources from NLTK
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Tokenize the corpus
tokens = nltk.word_tokenize(text)

# POS Tagging with NLTK
pos_tags = pos_tag(tokens, tagset='universal')

# Define Costum Lemmatizer
tag_dict = {'ADJ': wordnet.ADJ, 'NOUN': wordnet.NOUN, 'VERB': wordnet.VERB}
lemmatizer_wn = WordNetLemmatizer()
def lemmatize(token):
    word, tag = token[0], token[1]
    if tag in tag_dict: return lemmatizer_wn.lemmatize(word, tag_dict[tag])
    return word.lower()

# Lemmatize the tokens
tokens_lemmatized = [lemmatize(t) for t in pos_tags]

# Define a function to find collocations from lemmatized tokens (ds) and a window size
def find_collocations(ds, window_size = 1):
    # Find collocations with given window size
    finder = BigramCollocationFinder.from_words(ds, window_size)
    # Eliminate bigrams that occur less than 10 times
    finder.apply_freq_filter(10)
    # Eliminate bigrams that include stopwords or including any punctuation marks
    finder.apply_word_filter(lambda w: w in stop_words or not w.isalpha())
    # Eliminate all bigrams except those with POS tags NOUN-NOUN or ADJ-NOUN.
    finder.apply_ngram_filter(lambda w1, w2: (pos_tag([w1])[0][1], pos_tag([w2])[0][1]) not in [('NN', 'NN'), ('JJ', 'NN')])
    result = finder.ngram_fd.items()
    # result_10 = finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
    return result

# Define filters to find bigrams with
collocations1 = find_collocations(tokens_lemmatized, window_size = 2)
collocations3 = find_collocations(tokens_lemmatized, window_size = 4)

# Calculate word frequencies
word_freqs = FreqDist(tokens_lemmatized)

# Save word frequencies to a file
with open('word_frequencies.pkl', 'wb') as f:
    pickle.dump(word_freqs, f)

def get_collocations_dataframe(collocations, word_freqs):
    # Build DataFrame
    df = pd.DataFrame(columns=["Word1", "Word2", "Word1_Freq", "Word2_Freq", "Bigram_Freq"])

    # Fill DataFrame
    for bigram, bigram_freq in collocations:
        word1_freq = word_freqs[bigram[0]]
        word2_freq = word_freqs[bigram[1]]
        df = df.append({
            "Word1": bigram[0],
            "Word2": bigram[1],
            "Word1_Freq": word1_freq,
            "Word2_Freq": word2_freq,
            "Bigram_Freq": bigram_freq
        }, ignore_index=True)

    return df

df1 = get_collocations_dataframe(collocations1, word_freqs)
df3 = get_collocations_dataframe(collocations3, word_freqs)

# Save the DataFrames to disk
df1.to_csv('collocations1.csv', index=False)
df3.to_csv('collocations3.csv', index=False)

print("Collocations for window size 1:")
print(df1.head())
print("\nCollocations for window size 3:")
print(df3.head())
