from bs4 import BeautifulSoup
import requests
import csv
import re
import cloudscraper
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

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\lab2'
SEP = '|'

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }

URL_1 = 'https://rozetka.com.ua/ua/528525784/p528525784/comments/'
URL_2 = 'https://touch.com.ua/ua/item/apple-airpods-pro-2nd-gen-usb-c/'

CSV_FIELDS = ['Id', 'Comment']


def view_site(url):
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url)
    if response.status_code != 200:
        print(f"The request failed with an error {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'lxml')
    print(soup)

    return


def site_parser(url, filename):
    #response = requests.get(url, headers=headers)
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url)
    if response.status_code != 200:
        print(f"The request failed with an error {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'lxml')

    if url == URL_1:
        reviews = []
        quotes = soup.find_all('div', class_='comment__body-wrapper')
        for x in quotes:
            reviews.append(x.find('p').text)
    else:
        reviews = []
        quotes = soup.find_all('div', class_='impressions')
        for x in quotes:
            reviews.append(x.find('p').text)

    #quotes = soup.find_all('p')

    with open(filename, "w", encoding="utf-8") as output_file:
        csvwriter = csv.writer(output_file, delimiter=SEP)
        csvwriter.writerow(CSV_FIELDS)

        #print(f'----------------------- КОМЕНТАРІ {url} ---------------------------------')
        counter = 0
        for review in reviews:
            review = review.replace('\n', ' ')
            csvwriter.writerow([str(counter), review])
            counter += 1
        #print('-----------------------------------------------------------------------')

    return


def translate_data(df):
    def translate_line(r):
        line = r['Comment']
        translation = GoogleTranslator(source="uk", target="en").translate(line)
        return translation

    df['Comment'] = df.apply(translate_line, axis=1)
    return df


def clean_data(df):
    ps = PorterStemmer()

    def clean_row(r):
        tokens = r['Comment']
        # turns words into tokens and removes punctuation
        filtered_row = [word for word in tokens if word not in stop_words]
        stemmed_row = [ps.stem(word) for word in filtered_row]
        text_raw = " ".join(stemmed_row)
        text_raw = re.sub(r'\d+', '', text_raw)
        text_raw = re.findall('[a-zA-Z]+', str(text_raw))
        return text_raw

    df['Comment'] = df['Comment'].str.lower()
    df['Comment'] = df.apply(lambda x: tokenizer.tokenize(x['Comment']), axis=1)
    df['Comment'] = df.apply(clean_row, axis=1)

    return df


def count_words(filename):
    text_file = open(filename, "r", encoding="utf-8")
    data = text_file.read()
    data = data.split('\n')

    words_dict = dict()
    for word in data:
        if word in words_dict:
            words_dict[word] = words_dict[word] + 1
        else:
            words_dict[word] = 1

    words_dict = dict(sorted(words_dict.items(), key=lambda item: item[1], reverse=True))

    print(words_dict)

    return words_dict


def analyze_data(filename, url_name, print_process=False):
    raw_df = pd.read_csv(filename, sep=SEP, index_col=0)

    if print_process:
        print('RAW DATA')
        print(raw_df.head())
        print(raw_df.tail())
        print()
    # translate to eng
    # delete stopwords, numbers, symbols
    # tokenize + lemitization

    path = get_filename('translated_text_', url_name)
    if not os.path.exists(path):
        translated_df = translate_data(raw_df)
        translated_df.to_csv(path, sep=SEP)
    else:
        translated_df = pd.read_csv(path, sep=SEP, index_col=0)
    if print_process:
        print('TRANSLATED DATA')
        print(translated_df.head())
        print(translated_df.tail())
        print()

    path = get_filename('cleaned_text_', url_name)
    if not os.path.exists(path):
        cleaned_df = clean_data(translated_df)
        cleaned_df.to_csv(path, sep=SEP)
    else:
        cleaned_df = pd.read_csv(path, sep=SEP, index_col=0)
    if print_process:
        print('CLEANED DATA')
        print(cleaned_df.head())
        print(cleaned_df.tail())
        print()


def get_filename(name, url_name):
    filename = name + url_name + '.csv'
    filename = os.path.join(PROJECT_PATH, filename)
    return filename


if __name__ == '__main__':
    URL = URL_1
    if URL == URL_1:
        url_name = 'rozetka'
    else:
        url_name = 'touch'

    FILENAME = get_filename('raw_text_', url_name)

    if not os.path.exists(FILENAME):
        #view_site(URL)
        site_parser(URL, FILENAME)
    else:
        analyze_data(FILENAME, url_name, True)

