from bs4 import BeautifulSoup
import requests
import csv
import re
import cloudscraper
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt

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
        text = ' '.join(text_raw)
        return text

    df['Comment'] = df['Comment'].str.lower()
    df['Comment'] = df.apply(lambda x: tokenizer.tokenize(x['Comment']), axis=1)
    df['Comment'] = df.apply(clean_row, axis=1)

    return df


def sentiment_analysis(df):
    def get_sentiment(text):
        scores = analyzer.polarity_scores(text)
        sentiment = 1 if scores['pos'] > 0 else 0
        return sentiment

    analyzer = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Comment'].apply(get_sentiment)
    return df


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

    # sentiment analysis
    path = get_filename('sentiment_text_', url_name)
    if not os.path.exists(path):
        senti_df = sentiment_analysis(cleaned_df)
        senti_df.to_csv(path, sep=SEP)
    else:
        senti_df = pd.read_csv(path, sep=SEP, index_col=0)
    if print_process:
        print('SENTIMENT DATA')
        print(senti_df.head())
        print(senti_df.tail())
        print()

    path = get_filename('result_', url_name)
    if not os.path.exists(path):
        df = pd.concat([
                        senti_df['Sentiment'],
                        translated_df['Comment'],
                        raw_df['Comment']],
                       axis=1)
        df.columns = ['Sentiment',
                      'Translated_comment',
                      'Original_comment']
        df.to_csv(path, sep=SEP)

    return


def view_result(filename, url_name):
    df = pd.read_csv(filename, sep=SEP, index_col=0)
    print(df.head())
    print()
    print(df.tail())
    print()
    df['Sentiment'] = df['Sentiment'].astype('category')
    print(df['Sentiment'].value_counts())

    len_pos = len(df.loc[df['Sentiment'] == 1])
    len_neg = len(df.loc[df['Sentiment'] == 0])
    plt.bar(['negative', 'positive'], [len_neg, len_pos])
    plt.title(f'Pos/Neg review count for {url_name}')
    plt.show()

    return


def get_filename(name, url_name):
    filename = name + url_name + '.csv'
    folderpath = os.path.join(PROJECT_PATH, url_name)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    filename = os.path.join(folderpath, filename)
    return filename


if __name__ == '__main__':
    URL = URL_2
    if URL == URL_1:
        url_name = 'rozetka'
    else:
        url_name = 'touch'

    FILENAME = get_filename('raw_text_', url_name)

    if not os.path.exists(FILENAME):
        #view_site(URL)
        site_parser(URL, FILENAME)
    else:
        #analyze_data(FILENAME, url_name)
        RESULT_FILENAME = get_filename('result_', url_name)
        view_result(RESULT_FILENAME, url_name)

