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
CSV_NAME = 'lab2.csv'
CSV_PATH = os.path.join(PROJECT_PATH, CSV_NAME)

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
        csvwriter = csv.writer(output_file, delimiter='|')
        csvwriter.writerow(CSV_FIELDS)

        #print(f'----------------------- КОМЕНТАРІ {url} ---------------------------------')
        counter = 0
        for review in reviews:
            csvwriter.writerow([str(counter), review])
            counter += 1
        #print('-----------------------------------------------------------------------')

    return


def translate_data(filename, output_name):
    text_file = open(filename, "r", encoding="utf-8")
    data = text_file.read()
    data = data.split('\n')

    with open(output_name, "w", encoding="utf-8") as output_file:
        for line in data:
            translation = GoogleTranslator(source="uk", target="en").translate(line)
            print(translation)
            translated_line = translation + '\n'
            output_file.write(translated_line)

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


if __name__ == '__main__':
    URL = URL_1
    if URL == URL_1:
        url_name = 'rozetka'
    else:
        url_name = 'touch'

    FILENAME = 'raw_text_' + url_name + '.csv'
    FILENAME = os.path.join(PROJECT_PATH, FILENAME)
    #view_site(URL)
    site_parser(URL, FILENAME)

