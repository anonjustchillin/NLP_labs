from bs4 import BeautifulSoup
import requests
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.svm import LinearSVC
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import os.path
import numpy as np
import stanza
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import pickle

seed_num = 100

stop_words = set(stopwords.words('english'))
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

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

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


def clean_text(text):
    # turns words into tokens and removes punctuation
    tokens = tokenizer.tokenize(text.lower())

    # remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
    text = " ".join(stemmed_tokens)

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

    return text


def analyze_text(filename):
    data = pd.read_csv(filename)
    # tf-idf

    # лексична дисперсія

    # розподіл довжини слів

    # біграмний аналіз

    return


def create_path(name):
    path = os.path.join(PROJECT_PATH, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def labels_barplot(data):
    # barplot distribution of True and Fake news
    true_false_dist = data['isFake'].value_counts()
    color_code = [['True','green'],['Fake','red']]
    plt.bar(true_false_dist.index, true_false_dist.values, color=[color_code[k][1] for k in true_false_dist.index])
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Distribution of True and Fake news')
    plt.xticks([0, 1], ['True', 'Fake'])
    plt.show()


def prepare_train_data(print_res=False):
    # news file
    if not os.path.exists(NEWS_PATH):
        fake_news = pd.read_csv(FAKE_PATH)
        true_news = pd.read_csv(TRUE_PATH)

        fake_news['isFake'] = 1
        true_news['isFake'] = 0

        if print_res:
            print(fake_news.head())
            print('fake_news length: ' + str(len(fake_news)))
            print(true_news.head())
            print(len(true_news))
            print('true_news length: ' + str(len(true_news)))

        news = pd.concat([fake_news, true_news])
        if print_res:
            print(news.head())
            print(news.tail())
            print(len(news))
            print('news length: ' + str(len(news)))

        news.to_csv(NEWS_PATH, index=False)

    # cleaned news
    if not os.path.exists(CLEANED_NEWS_PATH):
        cleaned_news = pd.read_csv(NEWS_PATH)

        cleaned_news['Text'] = cleaned_news['title'].copy()
        #cleaned_news['Text'] = cleaned_news[['title', 'text']].agg(' '.join, axis=1)
        cleaned_news = cleaned_news.drop(['title', 'text', 'subject', 'date'], axis=1)

        cleaned_news = cleaned_news.sample(frac=1, random_state=seed_num).reset_index(drop=True)

        cleaned_news['Text'] = cleaned_news['Text'].map(clean_text)

        cleaned_news.to_csv(CLEANED_NEWS_PATH, index=False)
    else:
        cleaned_news = pd.read_csv(CLEANED_NEWS_PATH)

    if print_res:
        print(cleaned_news.head())
        print(cleaned_news.tail())
        print('cleaned_news length: ' + str(len(cleaned_news)))

        # barplot distribution of True and Fake news
        labels_barplot(cleaned_news)

    return


def teach_model():
    cleaned_news = pd.read_csv(CLEANED_NEWS_PATH)

    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(cleaned_news['Text'],
                                                        cleaned_news['isFake'],
                                                        random_state=seed_num,
                                                        stratify=cleaned_news['isFake'])

    # train classifier
    vectorizer = TfidfVectorizer()
    svc = LinearSVC()
    pipe = Pipeline([('vectorizer', vectorizer), ('clf', svc)])
    clf = pipe.fit(X_train, Y_train)

    # evaluate accuracy
    train_score = clf.score(X_train, Y_train)
    test_score = clf.score(X_test, Y_test)

    print('Accuracy (training data): '+str(train_score))
    print('Accuracy (test data): '+str(test_score))

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(Y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])
    disp.plot(cmap=plt.cm.viridis)
    plt.title('Confusion Matrix')
    plt.show()

    joblib.dump(clf, MODEL_FILE)
    joblib.dump({'vocabulary_': clf['vectorizer'].vocabulary_, 'idf_': clf['vectorizer'].idf_},
                VECTORIZER_FILE)

    return


def view_vectorizer():
    imported_vec = joblib.load(VECTORIZER_FILE)

    vectorizer = TfidfVectorizer(vocabulary=imported_vec['vocabulary_'])
    vectorizer.idf_ = imported_vec['idf_']
    print()
    print('First 20 items in TfidfVectorizer vocabulary and idf')
    print(list(vectorizer.vocabulary_.items())[:20])
    print(vectorizer.idf_[:20])
    print()
    print('Last 20 items in TfidfVectorizer vocabulary and idf')
    print(list(vectorizer.vocabulary_.items())[-20:])
    print(vectorizer.idf_[-20:])
    print()

    return


def test_model(filename, output):
    news = pd.read_csv(filename, index_col=0)

    # prep text
    news['Text'] = news['Titles'].map(clean_text)

    # load model
    imported_vec = joblib.load(VECTORIZER_FILE)
    vectorizer = TfidfVectorizer(vocabulary=imported_vec['vocabulary_'])
    vectorizer.idf_ = imported_vec['idf_']

    clf = joblib.load(MODEL_FILE)
    clf.vectorizer = vectorizer

    # predict
    news['isFake'] = clf.predict(news['Text'])

    labels_barplot(news)

    news.to_csv(output, index=True)


def process_site(url_name):
    url = URL_BBC if url_name == URL_NAME_BBC else URL_FOX

    folder_path = create_path(url_name)
    raw_filename = os.path.join(folder_path, RAW_FILENAME)
    #news_parser(url, raw_filename)
    output_filename = os.path.join(folder_path, 'output.csv')
    test_model(raw_filename, output_filename)


if __name__ == '__main__':
    #view_site(URL_BBC)

    prepare_train_data(True)
    teach_model()
    if os.path.exists(VECTORIZER_FILE):
        view_vectorizer()
    process_site(URL_NAME_BBC)
    process_site(URL_NAME_FOX)
