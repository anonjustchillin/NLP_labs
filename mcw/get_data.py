from bs4 import BeautifulSoup
import csv
import re
import cloudscraper
import os
from deep_translator import GoogleTranslator
import pandas as pd

SEP = '|'
HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\\mcw'

class TextManipulation:
    def __init__(self, filename):
        self.filename = filename
        self.filepath = os.path.join(PROJECT_PATH, filename)
        self.cols = ['Id', 'Review', 'CleanedReview', 'Product', 'Rating']

    def add_sentiment(self, sentiment):
        df = pd.read_csv(self.filepath, index_col=0)
        df['Rating'] = sentiment
        df.to_csv(self.filepath)

    def parse_data(self, url, name, first=False):
        url += 'comments/'
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url)
        if response.status_code != 200:
            print(f"The request failed with an error {response.status_code}")
            exit()

        soup = BeautifulSoup(response.text, 'lxml')

        reviews = []
        quotes = soup.find_all('div', class_='comment__body-wrapper')
        for x in quotes:
            reviews.append(x.find('p').text)

        if first:
            filepath = os.path.join(PROJECT_PATH, 'original_'+self.filename)
            with open(filepath, "w", encoding="utf-8") as output_file:
                csvwriter = csv.writer(output_file, delimiter=SEP)
                csvwriter.writerow(['Id', 'Review', 'Product'])
                counter = 0
                for review in reviews:
                    review = review.replace('\n', ' ')
                    csvwriter.writerow([str(counter), review, name])
                    counter += 1
            raw_df = pd.read_csv(filepath, sep=SEP, index_col=0)
            raw_df.to_csv(self.filepath)
            self.clean_data()
        else:
            df = pd.read_csv(self.filepath, index_col=0)
            for review in reviews:
                review = review.replace('\n', ' ')
                df.loc[len(df)] = [review, name, '']
            df.to_csv(self.filepath)
            self.clean_data()
        return

    def clean_data(self):
        def clean_row(r):
            text = r['Review']
            def rus_to_uk(text):
                translation = GoogleTranslator(source="russian", target="ukrainian").translate(text)
                return translation

            # переклад з рос на українську
            text = rus_to_uk(text)
            # посилання
            text = re.sub(r"https?://\S+|www\.\S+", ' ', text)
            # html теги
            text = re.sub(r"<.*?>", ' ', text)
            text = ' '.join(text.split())
            text = text.lower()

            if len(text) <=3:
                return pd.NA

            return text

        df = pd.read_csv(self.filepath, index_col=0)
        df['CleanedReview'] = df.apply(clean_row, axis=1)
        df.dropna(inplace=True, ignore_index=True)
        df.to_csv(self.filepath)
