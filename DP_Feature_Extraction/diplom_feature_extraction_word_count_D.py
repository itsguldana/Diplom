# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('df1.csv',  encoding='UTF-8')

import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import stop_words
#stop_words_russian = set(stopwords.words('russian'))
stop_words_russian = stop_words.get_stop_words('russian')

# Convert non-string values to empty string
df['text'] = df['text'].apply(lambda x: str(x) if type(x) != str else x)

# Remove stop words from the text column
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words_russian]))

# Define a function to convert text to lowercase
def convert_to_lower(text):
    return text.lower()

# Apply the function to the 'text' column of the DataFrame
df['text'] = df['text'].apply(convert_to_lower)

import re
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# Apply the function to the 'text' column of the DataFrame
df['text'] = df['text'].apply(remove_urls)

# Define a function to remove punctuation and extra spaces from a text string
def remove_punct_and_extra_space(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra space
    text = re.sub(' +', ' ', text)
    return text

# Apply the function to the 'text' column of the DataFrame
df['text'] = df['text'].apply(remove_punct_and_extra_space)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            
            tokens.append(token)
    if len(tokens) >= 1:
        return tokens
    return None

data = df.text.apply(lemmatize)
data = data.dropna()

from collections import Counter
all_text = ''

for el in data:
  for i in el:
    all_text += i
    all_text += ' '

# Разделим текст на слова
words = all_text.split()
for i in range(len(words)):
    words[i] = words[i].replace("\ufeff", " ")

# Подсчитаем количество уникальных слов и их частоту встречаемости
word_counts = Counter(words)

print(word_counts)

