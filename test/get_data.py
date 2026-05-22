from bs4 import BeautifulSoup
import requests
import cloudscraper
import pandas as pd
import os.path

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\\test'

URL_TRUSTED = 'https://suspilne.media/'
URL_TRUSTED_NAME = 'Суспільне'

URL_FAKE = 'https://resonance.ua/'
URL_FAKE_NAME = 'Резонанс'

def view_site(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    print('hi')
    print(soup)

    return

def news_parser(url, open_file=False):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    if url == URL_TRUSTED:
        url_name = URL_TRUSTED_NAME
        quotes = soup.find_all('span', class_='c-article-card__headline-inner')
    else:
        url_name = URL_FAKE_NAME
        quotes = soup.find_all('h2', class_="penci-entry-title entry-title grid-title")

    quotes = quotes[:5]

    filepath = os.path.join(PROJECT_PATH, 'news.csv')

    if open_file:
        data = pd.read_csv(filepath, index_col=0)
    else:
        data = pd.DataFrame(columns=['News', 'Site'])

    for quote in quotes:
        data.loc[len(data)] = [quote.text, url_name]

    data.to_csv(filepath, index=True)
    return

def create_df():
    #view_site(URL_FAKE)
    news_parser(URL_TRUSTED)
    news_parser(URL_FAKE, True)