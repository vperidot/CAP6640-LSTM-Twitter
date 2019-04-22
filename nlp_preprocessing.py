"""
Preprocessing for tweet data

@author Ity Bahadur, Victoria Proetsch
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re

def remove_pattern(text, pattern):
    """
    Removes a regex pattern from a string
    """
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, '', text)        
    return text

def clean_data(text):
    """
    Removes RTs, handles, hashtags, links, and most special characters 
    """
    # remove twitter Return handles (RT @xxx:)
    text = np.vectorize(remove_pattern)(text, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    text = np.vectorize(remove_pattern)(text, "@[\w]*")
    # remove URL links (httpxxx)
    text = np.vectorize(remove_pattern)(text, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    text = np.core.defchararray.replace(text, "[^ a-zA-Z#]", "")
    return text

def clean_data_text(text):
    """
    Performs same ops as clean data but returns a string instead of np array
    """
    # remove twitter Return handles (RT @xxx:)
    text = re.sub(r"RT @[\w]*:", "", text)
    # remove twitter handles (@xxx)
    text = re.sub(r"@[\w]*", "", text)
    # remove URL links (httpxxx)
    text = re.sub(r"https?://[A-Za-z0-9./]*", "", text)
    # remove special characters, numbers, punctuations (except for #)
    text = re.sub(r"[^ a-zA-Z#]", "", text)
    return text.lower()

def load_data_tfidf(filename):
    """
    Creates tf-idf features for each tweet
    """
    corpus = pd.read_csv(filename, delimiter='\t', header=None)
    corpus = clean_data(corpus)

    vectorizer = TfidfVectorizer(stop_words='english',
                                 analyzer='word',
                                 token_pattern=r'\w{2,}', #vetcorize 2-char word or more
                                 strip_accents='unicode')

    X = vectorizer.fit_transform(corpus[:,0])
    feature_names = vectorizer.get_feature_names()

    return X.toarray(), feature_names


def load_data_indexes(filename):
    """
    Transforms text to word-index lists (no feature extraction)
    """
    text_data = []
    vocab_count = {}
    with open(filename, 'r', encoding='utf8') as fin, open(filename[:-4]+'_tokens.txt', 'w') as fout:
        for line in fin:
            # clean data
            line = clean_data_text(line)
            # tokenize
            tokens = nltk.word_tokenize(line)
            # update vocab/frequency count
            for token in tokens:
                if token in vocab_count:
                    vocab_count.update({token:vocab_count[token]+1})
                else:
                    vocab_count.update({token:1})

            # add to the data array
            text_data.append(tokens)

    #create vocab word-index dictionary
    vocab = {
        'UNK' : 0,
        'START' : 1,
        'END' : 2
    }
    i=3
    for key, value in sorted(vocab_count.items(), key=lambda item: item[1], reverse=True):
        # don't include words in the vocab used less than three times
        if value > 3:
            #add to the vocab
            vocab.update({key:i})
            i+=1

    # transform text to indices
    data=[]
    for line in text_data:
        indices = [vocab[x] if x in vocab else 0 for x in line]
        indices = [1] + indices + [2]
        data.append(indices)

    return data[1:], vocab

def load_data_chars(filename):
    """
    Transforms text to character-index lists by ASCII code
    (no feature extraction)
    """

    # Build the char-index dictionary
    valid_chars = list('abcdefghijklmnopqrstuvwxyz #')
    vocab = {
        'UNK' : 0,
        'START' : 1,
        'END' : 2,
    }
    i = 3
    for c in valid_chars:
        vocab.update({c:i})
        i += 1

    # read the file
    char_data = []
    with open(filename, 'r', encoding='utf8') as fin, open(filename[:-4]+'_tokens.txt', 'w') as fout:
        for line in fin:
            # clean data
            line = clean_data_text(line)
            # tokenize by character
            tokens = [vocab[c] for c in line]
            tokens = [1] + tokens + [2]
            # add to the data array
            char_data.append(tokens)

    return char_data[1:], vocab

def vec_to_text(vec, index_word):
  """
  Turn a vector of indices into a string
  """
  words = []
  for i in vec:
    
    if type(i) != int:
      i = int(i)
    
    words.append(index_word[i])

  return ' '.join(words)

def create_sequence_next_pairs(train, seq_len):
    """
    Split arbitrary-length lists into pairs of (sequence, next)
    Where sequence is a list of length seq_len and next is the single next element

    @returns X, y where X is the list of sequences, y is the list of next elements
    """
    X  = []
    y = []

    for line in train:
      if line:
        if line.count(2)/len(line) <= 0.1: #if there are fewer than 10% unknowns 
          i=0
          while (i+seq_len < len(line)):
            X.append(line[i:i+seq_len])
            y.append(line[i+seq_len])
            i += 1

    return X, y

def train_val_test_split(X, y, train_pct=0.7, val_pct=0.1):
    """
    Split into training, validation, test sets by listed percentage
    (test_pct is 1 - train_pct - val_pct)
    """

    train_size = int(len(X) * train_pct)
    val_size = int(len(X) * val_pct)

    X_train = np.array(X[:train_size])
    y_train = np.array(y[:train_size])
    X_val = np.array(X[train_size:(train_size+val_size)])
    y_val = np.array(y[train_size:(train_size+val_size)])
    X_test = np.array(X[train_size+val_size:])
    y_test = np.array(y[train_size+val_size:])

    print('train, validate, test: {}, {}'.format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


#data, vocab = load_data_indexes('data/Trump_tweetdata.txt')
