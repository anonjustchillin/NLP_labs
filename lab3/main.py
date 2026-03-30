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

RAW_FILENAME = 'raw_text.csv'


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

    # to pandas df
    data = pd.DataFrame(columns=['Titles'])
    for quote in quotes:
        data.loc[len(data)] = quote.text
    print(data)
    data.to_csv(filename, index=True)

    return


def filter_data_1(filename, output_name):
    data = pd.read_csv(filename, index_col=0)
    print(data)

    # remove punctuation, numbers, eng words
    text_file = open(filename, "r", encoding="utf-8")
    data = text_file.read()

    def en_to_uk(text):
        translation = GoogleTranslator(source="en", target="uk").translate(text)
        return translation

    def clean_text(text):
        # переклад з англ на укр
        if re.search('[a-zA-Z]', text):
            text = en_to_uk(text)

        # посилання
        text = re.sub(r"https?://\S+|www\.\S+", '', text)
        # html теги
        text = re.sub(r"<.*?>", '', text)
        # пунктуація
        text = re.sub(r"[^\w\s]", '', text)
        # слова з цифрами
        text = re.sub(r"\w*\d\w*", '', text)
        # цифри
        text = re.sub(r'\d+', '', text)
        #  
        text = re.sub(r' ', ' ', text)

        return text




    return


def filter_data_2(filename, output_name):
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
    if not os.path.exists(RAW_FILENAME_PATH):
        news_parser(URL, RAW_FILENAME_PATH)

    CLEAN_1_FILENAME_PATH = os.path.join(FOLDER_PATH, "cleaned_text_1.csv")
    filter_data_1(RAW_FILENAME_PATH, CLEAN_1_FILENAME_PATH)


