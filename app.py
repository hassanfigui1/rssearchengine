import csv
from time import time
import time
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
import tokenize
from operations import Read_csv, RemovePunctuation, drop_duplicates
from flask import Flask
from flask import  render_template,request,url_for,redirect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from csv import writer
from nltk.tokenize import word_tokenize

porter = SnowballStemmer("english")
from nltk.tokenize import word_tokenize

from flask_paginate import Pagination, get_page_args
app = Flask(__name__)
def get_myData(myData,offset=0,per_page=20):
    return myData[offset:offset + per_page]
def to_Lower(df):
    df = [entry.lower() for entry in df]
    return df 
def tokenize_words(text):
    tokens = word_tokenize(text)
    return tokens
def steem_word(df):
    return [porter.stem(w) for w in tokenize_words(df)]

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
	    requete = request.form["nm"]
	    return redirect(url_for("home", requete=requete))
    else:
        return render_template("index.html")

@app.route("/reviews/data",methods=["POST"])
def reviews():
    df = request.form.get("df")
    title = request.form.get("title")
    jobF = request.form.get("jobF")
    industry = request.form.get("industry")
    id = request.form.get("id")
    requete = request.form.get("requete")
    if request.form.get("rating"):
        rating = request.form.get("rating")
    else:
        rating =0
    List = [id,rating,requete]
    with open("reviews.csv", 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(List)
    return redirect(url_for("home",requete=requete))
@app.route("/details/<title>/<jobF>/<industry>/<df>/<id>/<requete>/")
def details(title,jobF,industry,df,id,requete):
    if request.method == "POST":
        requeteD = request.form["requete"]
        reviewD = request.form["reviewD"]
        idD = request.form["idD"]
        return render_template("details.html",title=title,jobF=jobF,industry=industry,df=df,id=id,requete=requete)
    else:
        print("hassanfig")
        return render_template("details.html",title=title,jobF=jobF,industry=industry,df=df,id=id,requete=requete)
@app.route("/<requete>")
def home(requete):
    start_time = time.time()
    #req1 here is used to send it through url in order to save later in a csv file
    req1 = requete
    df  = Read_csv("jobs_data.csv")
    # df = CleanData(df)
    df= drop_duplicates(df)
    df['clean_title'] = [entry.lower() for entry in df['title']]
    columns_list = ['title','jobFunction','industry']
    df = RemovePunctuation(df,columns_list)
    df['tokenized_word'] = df['clean_title'].apply(lambda x: tokenize_words(x))
    df['removed_stopwords']= df['tokenized_word'].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))
    df['steem_title'] = df['removed_stopwords'].apply(lambda x: steem_word(x))
    df['steem_title']= df['steem_title'].apply(lambda x: ' '.join([word for word in x]))
    
    vectorizer = TfidfVectorizer(decode_error='ignore', lowercase = True, min_df=2)
    X = vectorizer.fit_transform(df['steem_title'])

    query = word_tokenize(requete)
    query = [word for word in query if not word in stopwords.words()]
    req=[]
    res =""
    for word in query:
        res +=' '+porter.stem(word)
    query_vec = vectorizer.transform([res]) 
    results = cosine_similarity(X,query_vec).reshape((-1,)) 
    
    
    sizeOfsimilarities = results[ (results >= 0.5) & (results <=1) ].size
    myData=results.argsort()[sizeOfsimilarities:][::-1]
    
    page,per_page,offset = get_page_args(page_parameter="page",per_page_parameter="per_page")
    total = sizeOfsimilarities
    pagination_myData = get_myData(myData,offset=offset,per_page=per_page)
    
    pagination = Pagination(page=page,per_page=per_page,total=total,css_framework="bootstrap4")
    
    return render_template("result.html",myData=pagination_myData,
                           page = page,per_page=per_page,
                           pagination=pagination,df=df,res=req1,time=time.time()-start_time,nbOfResults=sizeOfsimilarities)

if __name__ == "__main__":
    app.run(debug=False)
