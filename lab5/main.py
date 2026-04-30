from bs4 import BeautifulSoup
import requests
import re
from deep_translator import GoogleTranslator
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import os.path
import numpy as np

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\lab5'
CSV_NAME = 'lab5.csv'
CSV_PATH = os.path.join(PROJECT_PATH, CSV_NAME)

NAME_OLX = 'Olx'
NAME_AUTORIA = 'AutoRIA'
NAME_RST = 'RST'
NAME_AUTOMOTO = 'AUTOMOTO'

AUTO_TYPES = {0: 'позашляховики', 1: 'седани', 2: 'мінівени'}

URLS = {NAME_OLX: ['https://www.olx.ua/uk/transport/legkovye-avtomobili/?currency=UAH&search%5Bfilter_enum_car_body%5D%5B0%5D=off-road-vehicle',
                   'https://www.olx.ua/uk/transport/legkovye-avtomobili/?currency=UAH&search%5Bfilter_enum_car_body%5D%5B0%5D=sedan',
                   'https://www.olx.ua/uk/transport/legkovye-avtomobili/?currency=UAH&search%5Bfilter_enum_car_body%5D%5B0%5D=minibus'],
        NAME_AUTORIA: ['https://auto.ria.com/uk/search/?search_type=1&category=1&bodystyle[0]=5&abroad=0&customs_cleared=1&page=0&limit=20',
                       'https://auto.ria.com/uk/search/?search_type=1&category=1&bodystyle[0]=3&abroad=0&customs_cleared=1&page=0&limit=20',
                       'https://auto.ria.com/uk/search/?search_type=1&category=1&bodystyle[0]=8&abroad=0&customs_cleared=1&page=0&limit=20'],
        NAME_AUTOMOTO: ['https://automoto.ua/uk/car/Vnedorozhnik-Krossover',
                        'https://automoto.ua/uk/car/Sedan',
                        'https://automoto.ua/uk/car/Miniven']}

RAW_FILENAME = 'raw_text.csv'
CLEANED_FILENAME = 'cleaned_text.csv'


# Реалізуйте web-скрапінг будь-яких платформ / виробників з продажу автомобілів.
# Бажано обробити не менше 3-х джерел.
# Проведіть агрегацію пропозицій за категоріями:
#   - позашляховик;
#   - седан;
#   - мінівен.
# Результати збережіть у файлах.
#
# Провести порівняльний аналіз пропозицій між платформами.
#

def view_site(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    print(soup)
    return


def news_parser(url, filename):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    if url == URL_1:
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


def clean_text(text):
    def uk_to_en(text):
        translation = GoogleTranslator(source="uk", target="en").translate(text)
        return translation

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
    # переклад з укр на англ
    if not re.search('[a-zA-Z]', text):
        text = uk_to_en(text)
    # extra spaces
    text = " ".join(text.split())

    return text


def create_path(name):
    path = os.path.join(PROJECT_PATH, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    print('hi')
