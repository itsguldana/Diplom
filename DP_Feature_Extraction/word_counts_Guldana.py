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

# Tokenize the text in the 'text' column
df['text_tokenized'] = df['text'].apply(lambda x: nltk.word_tokenize(x))

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

import pymorphy2

# Инициализируем лемматизатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

# Определяем функцию для лемматизации текста
def lemmatize_text(text):
    # Разделяем текст на слова
    words = text.split()
    # Лемматизируем каждое слово
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    # Соединяем лемматизированные слова в строку
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# Применяем функцию лемматизации к столбцу 'text'
df['text_lemmatized'] = df['text'].apply(lemmatize_text)

# Определим функцию для подсчета количества слов
def count_words(text):
    # Разделяем текст на слова
    words = text.split()
    # Возвращаем количество слов
    return len(words)

# Применяем функцию count_words к столбцу 'text' и сохраняем результат в новый столбец 'word_count'
df['word_count'] = df['text_lemmatized'].apply(count_words)

from collections import Counter

all_text = ' '.join(df['text_lemmatized'])

# Разделим текст на слова
words = all_text.split()
for i in range(len(words)):
    words[i] = words[i].replace("\ufeff", "")


# Подсчитаем количество уникальных слов и их частоту встречаемости
word_counts = Counter(words)

print(word_counts)