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
    print("Cleaning Text Features... ")
    df = df.fillna(' ', inplace=False)
    # df = df.drop_duplicates(subset=['title'])
    # for col in df:
    #     df[col] = df[col].apply(lambda x: str(x).replace('  ', ' '))
    #     df[col] = df[col].apply(lambda x: str(x).replace('-', ' '))
    #     df[col] = df[col].apply(lambda x: str(x).lower())
    #     df[col] = df[col].apply(lambda x: str(x).replace('&', ''))
    #     df[col] = df[col].apply(lambda x: str(x).replace('.', ''))
    return df

def df():
    df  = Read_csv("jobs_data.csv")
    df = drop_duplicates(df)
    columns_list = ['title','jobFunction','industry']
    df = RemovePunctuation(df,columns_list)
    df['clean_title'] = df['title']
    df['clean_title'] = to_Lower(df['clean_title'])
    df['clean_title']= remove_StopWrods(df['clean_title'])
    df['tokenized_title']=df['title'].apply(lambda x: tokenize(x))
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

