# Mohammed Bari
# CS 4650 HW 5

import pandas
import spacy
from newsapi import NewsApiClient
import pickle
import en_core_web_lg
from wordcloud import WordCloud
from string import punctuation
from collections import Counter


nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient(api_key='3b1062ec54d547d684e6cce12a35958b')
dados = []
keywords = []
tags = ['VERB', 'NOUN', 'PROPN']


def getAll(i):
    temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2022-03-05', to='2022-03-25',
                                  sort_by='relevancy', page=i)
    return temp


def get_keywords_eng(art):
    result = []
    page = nlp_eng(art.lower())
    for token in page:
        if token.text in nlp_eng.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in tags:
            result.append(token.text)
    return result


articles = list(map(getAll, range(1, 5)))

for i, article in enumerate(articles):
    for x in article['articles']:
        title = x['title']
        description = x['description']
        content = x['content']
        # date = x['date']
        dados.append({'title': title, 'desc': description, 'content': content})

df = pandas.DataFrame(dados)
df = df.dropna()
df.head()

for content in df.content.values:
    keywords.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])
df['keywords'] = keywords

filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))
filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = 'articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

cvArt = pickle.loads(open('articlesCOVID.pckl', 'rb').read())
df = pandas.DataFrame(cvArt)
df.to_csv(r'covidArticles.csv')

i = str(keywords)
cloud = WordCloud(width= 1000, height= 500).generate(i)
cloud.to_file("covidArticlesWordCloud.png")
