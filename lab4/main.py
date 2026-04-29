from bs4 import BeautifulSoup
import requests
import re
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import os.path
import numpy as np
import stanza


tokenizer = RegexpTokenizer(r'\w+')

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\lab4'
CSV_NAME = 'lab4.csv'
CSV_PATH = os.path.join(PROJECT_PATH, CSV_NAME)

URL_BBC = 'https://www.bbc.com/news'
URL_FOX = 'https://www.foxnews.com/'

URL_NAME_BBC = 'BBC News'
URL_NAME_FOX = 'FoxNews'

RAW_FILENAME = 'raw_text.csv'

DATA_PATH = os.path.join(PROJECT_PATH, 'data')
FAKE_PATH = os.path.join(DATA_PATH, 'Fake.csv')
TRUE_PATH = os.path.join(DATA_PATH, 'True.csv')

NEWS_PATH = os.path.join(DATA_PATH, 'news.csv')
CLEANED_NEWS_PATH = os.path.join(DATA_PATH, 'cleaned_news.csv')

def view_site(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    print(soup)
    return


def news_parser(url, filename):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    if url == URL_BBC:
        # h2 card-headline
        quotes = soup.find_all('h2')
    else:
        # h3 title
        quotes = soup.find_all('h3', class_='title')

    # to pandas df
    data = pd.DataFrame(columns=['Titles'])
    for quote in quotes:
        data.loc[len(data)] = quote.text
    print(data)
    data.to_csv(filename, index=True)

    return


def filter_data_1(filename):
    data = pd.read_csv(filename)

    # remove punctuation, numbers

    def clean_text(text):
        # посилання
        text = re.sub(r"https?://\S+|www\.\S+", '', text)
        # html теги
        text = re.sub(r"<.*?>", '', text)
        # пунктуація
        text = re.sub(r"[^\w\s]", ' ', text)
        # слова з цифрами
        text = re.sub(r"\w*\d\w*", '', text)
        # цифри
        text = re.sub(r'\d+', '', text)
        #  
        text = re.sub(r' ', ' ', text)
        # extra spaces
        text = " ".join(text.split())

        text = text.lower()
        return text

    data['title'] = data['title'].map(clean_text)
    data['text'] = data['text'].map(clean_text)

    return data


def create_folder(name):
    path = os.path.join(PROJECT_PATH, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def teach_model(print_res=False):
    model_folder = create_folder('model')

    # news file
    if not os.path.exists(NEWS_PATH):
        fake_news = pd.read_csv(FAKE_PATH)
        true_news = pd.read_csv(TRUE_PATH)

        fake_news['isFake'] = 1
        true_news['isFake'] = 0

        if print_res:
            print(fake_news.head())
            print(len(fake_news))
            print(true_news.head())
            print(len(true_news))

        news = pd.concat([fake_news, true_news])
        if print_res:
            print(news.head())
            print(news.tail())
            print(len(news))

        news.to_csv(NEWS_PATH, index=False)

    # cleaned news
    if not os.path.exists(CLEANED_NEWS_PATH):
        cleaned_news = filter_data_1(NEWS_PATH)
        cleaned_news = cleaned_news.drop(['subject', 'date'])
        cleaned_news.to_csv(CLEANED_NEWS_PATH, index=False)
    else:
        cleaned_news = pd.read_csv(CLEANED_NEWS_PATH)

    if print_res:
        print(cleaned_news.head())
        print(cleaned_news.tail())
        print(len(cleaned_news))




    return


def process_site(url_name):
    url = URL_BBC if url_name == URL_NAME_BBC else URL_FOX

    folder_path = create_folder(url_name)
    raw_filename = os.path.join(folder_path, RAW_FILENAME)
    news_parser(url, raw_filename)


if __name__ == '__main__':
    #view_site(URL_BBC)
    #process_site(URL_NAME_BBC)
    #process_site(URL_NAME_FOX)
    teach_model(True)
