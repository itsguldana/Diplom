import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import string
from stop_words import get_stop_words

class FeatureExtraction:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_csv(input_file, encoding='UTF-8')
        self.stop_words_russian = get_stop_words('russian')
        self.patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
        self.stopwords_ru = stopwords.words("russian")
        self.morph = MorphAnalyzer()

    def preprocess_text(self, text):
        # Convert non-string values to empty string
        text = str(text) if type(text) != str else text

        # Remove stop words from the text column
        text = ' '.join([word for word in text.split() if word.lower() not in self.stop_words_russian])

        # Convert text to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove punctuation and extra spaces
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(' +', ' ', text)

        return text

    def lemmatize_text(self, text):
        text = re.sub(self.patterns, ' ', text)
        tokens = []
        for token in text.split():
            if token and token not in self.stopwords_ru:
                token = token.strip()
                token = self.morph.normal_forms(token)[0]
                tokens.append(token)
        return tokens

    def extract_features(self):
        # Preprocess text
        self.df['text'] = self.df['text'].apply(self.preprocess_text)

        # Tokenize text
        self.df['text_tokenized'] = self.df['text'].apply(lambda x: nltk.word_tokenize(x))

        # Lemmatize text
        self.df['text_lemmatized'] = self.df['text'].apply(self.lemmatize_text)

        # Save feature array as .npy file
        features = np.array(self.df['text_lemmatized'].tolist(), dtype=object)
        np.save('features.npy', features)

if __name__ == '__main__':
    # Instantiate FeatureExtraction object and extract features
    fe = FeatureExtraction('df1.csv')
    fe.extract_features()