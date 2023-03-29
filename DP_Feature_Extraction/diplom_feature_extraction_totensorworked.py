#!/usr/bin/env python
# coding: utf-8


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


# In[138]:


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


df3=pd.DataFrame(columns=['id','text','label'])
df3['text']=data
df3['id']=df['id']
df3['label']=df['label']
df3=df3.dropna()

df4=pd.DataFrame(columns=['id','text','label'])

df4['id']=df3['id']
df4['label']=df3['label']


from sklearn import preprocessing
import torch
from torch import nn


# In[201]:


le=preprocessing.LabelEncoder()


# In[215]:


count=0
maxlenth=len(df3)
for el in df3['text']:
    if count==maxlenth:break
    if not el:continue
    all_text=''
    
    for i in el:
        all_text = all_text+i
        all_text= all_text+' '
        
    
    df4['text'][count]=all_text
    count=count+1
df4


# In[216]:


df4.isna().sum()


# In[217]:


df4.tail(50)


# In[218]:


type(df4['text'])


# In[219]:


x=torch.as_tensor(le.fit_transform(df4['text']))


# In[220]:


y=torch.as_tensor(le.fit_transform(df4['label']))


# In[221]:


x.unsqueeze_(1)


# In[222]:


y.unsqueeze_(1)




if torch.cuda.is_available():
    dev=torch.device('cuda:0')
else: dev=torch.device('cpu:0')


# In[224]:


dev


# In[225]:


x=x.to(dev)


# In[226]:


y=y.to(dev)

print(x)
print(y)