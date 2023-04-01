import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import re
import string
import stop_words


class FeatureExtraction:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath, encoding='UTF-8')
        self.stop_words_russian = stop_words.get_stop_words('russian')
        self.patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[]^`{|}~—-]+"
        self.stopwords_ru = stopwords.words("russian")
        self.morph = MorphAnalyzer()

    def _clean_text(self, text):
        text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(' +', ' ', text)
        return text

    def _lemmatize(self, doc):
        doc = re.sub(self.patterns, ' ', doc)
        tokens = []
        for token in doc.split():
            if token and token not in self.stopwords_ru:
                token = token.strip()
                token = self.morph.normal_forms(token)[0]
                tokens.append(token)
        if len(tokens) >= 1:
            return tokens
        return None

    def extract_features(self):
        self.df['text'] = self.df['text'].apply(self._clean_text)
        self.df['text_tokenized'] = self.df['text'].apply(lambda x: nltk.word_tokenize(x))
        data = self.df['text'].apply(self._lemmatize)
        self.df1 = pd.DataFrame(columns=['id', 'text', 'label'])
        self.df1['text'] = data
        self.df1['id'] = self.df['id']
        self.df1['label'] = self.df['label']
        # self.df = self.df.dropna()
        return self.df1
if __name__ == '__main__':
    # Instantiate FeatureExtraction object and extract features
    fe = FeatureExtraction()
    fe.extract_features()