from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora, models
import gensim
import pandas as pd
import re
import math

tokenizer = RegexpTokenizer(r'\w+')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# File paths
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

stops = set(stopwords.words("english"))

# list for tokenized documents in loop
texts = []

def LDA(row):
    # clean and tokenize document string
    Q1raw = str(row['question1']).lower().split()
    Q2raw = str(row['question2']).lower().split()

    tokens = Q1raw + Q2raw
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if i not in stops]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
    texts.append(stemmed_tokens)

LDA_data = df_train.apply(LDA, axis=1, raw=True)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# generate LDA model
ldamodel = Lda(corpus, num_topics=2, id2word=dictionary, passes=20)

print(ldamodel)