from bs4 import BeautifulSoup
import requests
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
import os.path
import numpy as np
import stanza

# частотний аналіз
# частотні повторення тегів
# українською! прибрати переклад
# графіки

tokenizer = RegexpTokenizer(r'\w+')

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\lab3'
CSV_NAME = 'lab3.csv'
CSV_PATH = os.path.join(PROJECT_PATH, CSV_NAME)

URL_1 = 'https://suspilne.media/'
URL_2 = 'https://hromadske.ua/'

URL_NAME_1 = 'suspilne'
URL_NAME_2 = 'hromadske'

RAW_FILENAME = 'raw_text.txt'


def view_site(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    print(soup)
    return


def news_parser(url, filename):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    if url == URL_1:
        quotes = soup.find_all('span', class_='c-article-card__headline-inner')
    else:
        quotes = soup.find_all('h3')

    with open(filename, "w", encoding="utf-8") as output_file:
        print(f'----------------------- СТРІЧКА НОВИН {url} ---------------------------------')
        for quote in quotes:
            print(quote.text)
            output_file.write(quote.text)
            output_file.write('\n')
        print('-----------------------------------------------------------------------')

    return


def filter_data(filename, output_name):
    text_file = open(filename, "r", encoding="utf-8")
    data = text_file.read()

    # turns words into tokens and removes punctuation
    tokens = tokenizer.tokenize(data.lower())

    # remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
    text_raw = " ".join(stemmed_tokens)
    text_raw = re.sub(r'\d+', '', text_raw)

    text_raw = re.findall('[a-zA-Z]+', str(text_raw))

    with open(output_name, "w", encoding="utf-8") as output_file:
        for line in text_raw:
            #print(line)
            output_file.write(line+'\n')

    return


if __name__ == '__main__':
    print(f"1 - {URL_1}\n2 - {URL_2}")
    option = 0
    while True:
        if option == 1 or option == 2:
            break
        option = int(input("Select option: "))
    print()
    URL = URL_1 if option == 1 else URL_2
    URL_NAME = URL_NAME_1 if option == 1 else URL_NAME_2

    FOLDER_PATH = os.path.join(PROJECT_PATH, URL_NAME)
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)

    RAW_FILENAME_PATH = os.path.join(FOLDER_PATH, RAW_FILENAME)
    #news_parser(URL, RAW_FILENAME_PATH)
