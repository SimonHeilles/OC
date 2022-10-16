import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)
from collections import Counter
import tensorflow as tf
import sklearn

#Preprocessing
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

#Feature Extractions
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.models import Word2Vec
import tensorflow_hub as hub
from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertModel,
    DistilBertConfig,
)

#Predictions
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def preprocess(df_input):
    
    df = df_input.copy()

    # Pre-processing
    df['Body'] = df['Body'].apply(lambda x: BeautifulSoup(x).get_text())

    df['Tags'] = df['Tags'].str.split().str.join(" ")

    text_columns = df[['Title', 'Body']]

    for column in text_columns:
      df[column] = df[column].str.lower()

    for column in text_columns:
      spec_chars = ["!",'"',"#","%","&","'","(",")",
                  "*","+",",","-",".","/",":",";","<",
                  "=",">","?","@","[","\\","]","^","_",
                  "`","{","|","}","~","–", "$", "0", "1",
                  "2", "3", "4", "5", "6", "7", "8", "9"]

    for char in spec_chars:
        df[column] = df[column].str.replace(char, ' ')

    for column in text_columns:
      df[column] = df[column].str.split().str.join(" ")

    df2 = df.copy()
    cachedStopWords = stopwords.words("english")

    for column in text_columns:
      df2[column] = df2[column].apply(lambda x: [str(word) for word in word_tokenize(x) if not word in cachedStopWords])

    for column in text_columns:
      df2[column] = df2[column].apply(lambda x: ' '.join(x))

    # NB : no stemming, doesn't really increase the results

    # Preparing the list of tags
    df_cv = df2.copy()
    df_cv['TitleBody'] = df_cv['Title'] + ' ' + df_cv['Body']
        
    count_vect = CountVectorizer(max_features=1000, binary=True) # réutiliser notre modèle déjà fitté ?
    
    X_vectors = count_vect.transform(df_cv['TitleBody'])
    
    return X_vectors


def predict_tags(X_test, selected_model):
    # import du modèle pré-fitté et fine tuné
    pred=selected_model.predict(X_test1)
    
    
    return pred