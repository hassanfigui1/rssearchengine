import pandas as pd
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


#1 reading dataset 
def Read_csv(file):
    movies = pd.read_csv(file)
    return movies
#Lower case function:
def to_Lower(df):
    df = [entry.lower() for entry in df]
    return df 
def tokenize_words(text):
    tokens = word_tokenize(text)
    return tokens
def steem_word(df):
    return [porter.stem(w) for w in tokenize_words(df)]

# def replace_punc(df):
#     df['title'] = df['title'].str.replace('+',' plus')
#     df['title'] = df['title'].str.replace('#',' sharp')
#     return df['title']

def RemovePunctuation(df,columns_list):
    punctuationList = []
    for i in string.punctuation:
        if(i!=','):
            punctuationList.append(i)
    for i in columns_list:
        for j in punctuationList:
            df[i] = df[i].str.replace(j,' ')
    return df
#  RemovePunctuation(lower_case(jobs))
# 2 - Cleaning jobs data set 
def drop_duplicates(df):
    columns_list = ['title','jobFunction','industry']
    df = df.fillna(' ', inplace=False)
    df['title'] = df['title'].str.replace('+',' plus')
    df['title'] = df['title'].str.replace('#',' sharp')

    df = df.drop_duplicates(subset=['title'])
    # for col in df:
    #     df[col] = df[col].apply(lambda x: str(x).replace('  ', ' '))
    #     df[col] = df[col].apply(lambda x: str(x).replace('-', ' '))
    #     df[col] = df[col].apply(lambda x: str(x).lower())
    #     df[col] = df[col].apply(lambda x: str(x).replace('&', ''))
    #     df[col] = df[col].apply(lambda x: str(x).replace('.', ''))
        
    return df

# 2 - remove stop words
def remove_StopWrods(data):
    stop = stopwords.words('english')
    data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return data
def word_tokenizer(data):
    data = data.apply(lambda x: word_tokenize(x))
    return data
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def getRecommendation(requete, df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df) 
    query_vec = vectorizer.transform([requete]) 
    results = cosine_similarity(X,query_vec).reshape((-1,)) 
    return results


