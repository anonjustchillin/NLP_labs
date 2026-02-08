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

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\lab1'
CSV_NAME = 'lab1.csv'
CSV_PATH = os.path.join(PROJECT_PATH, CSV_NAME)

URL_1 = 'https://suspilne.media/'
URL_2 = 'https://hromadske.ua/'
URLs = [URL_1, URL_2]

DATE_FORMAT = "%Y-%m-%d %H:%M"


def news_parser(url, filename):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    #print(soup)

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


def create_time_series():
    cols = ['Datetime',
            'Top_5',
            'Frequency',
            'Freq_sum',
            'Comment']
    ts = pd.DataFrame(columns=cols)
    print('empty dataframe created')
    return ts


def update_time_series(ts, words_dict, curr_date_str):
    temp_ts = create_time_series()

    top5_keys = list(words_dict.keys())[0:5]
    top5_values = list(words_dict.values())[0:5]

    for i in range(0, 5):
        temp_ts.loc[len(temp_ts)] = [
                           curr_date_str,
                           top5_keys[i],
                           top5_values[i],
                           pd.NA,
                           pd.NA]
    temp_ts = temp_ts.assign(Freq_sum=temp_ts.Frequency.sum())

    ts = pd.concat([ts, temp_ts])
    print('dataframe updated')
    return ts


def create_word_cloud(filename, output_name):
    text_file = open(filename, "r", encoding="utf-8")
    data = text_file.read()

    wordcloud = WordCloud().generate(data)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    plt.savefig(output_name)


def plot_freq(ts, output_name):
    x_datetime = pd.to_datetime(ts.Datetime)
    x = x_datetime[::5]
    y = ts.Freq_sum[::5]

    left = datetime(2026, 2, 8, 20, 0)
    right = datetime(2026, 2, 8, 23, 0)

    plt.plot(x, y)
    plt.xticks(rotation=30, ha='right')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().set_xbound(left, right)
    plt.show()

    plt.savefig(output_name)


def get_text_date():
    curr_datetime = datetime.now()
    curr_date_str = curr_datetime.strftime(DATE_FORMAT)

    curr_year = curr_datetime.year
    curr_month = curr_datetime.month
    curr_day = curr_datetime.day
    curr_hour = curr_datetime.hour
    curr_minute = curr_datetime.minute

    curr_date = str(curr_year) + '-' + str(curr_month) + '-' + str(curr_day)
    curr_time = str(curr_hour) + ':' + str(curr_minute)
    curr_time_folder = str(curr_hour) + '-' + str(curr_minute)

    return curr_date, curr_date_str, curr_time_folder


if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        ts = create_time_series()
        ts.to_csv(CSV_NAME)
    else:
        ts = pd.read_csv(CSV_NAME, index_col=0)

    curr_date, curr_date_str, curr_time_folder = get_text_date()

    FOLDER_NAME = curr_date + '_' + curr_time_folder
    FOLDER_PATH = os.path.join(PROJECT_PATH, FOLDER_NAME)
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)

    for i in range(2):
        URL = URLs[i]
        print(f'URL: {URL}')

        FILENAME = 'raw_text_' + str(i) + '.txt'
        TRANSLATED_FILENAME = 'translated_text_' + str(i) + '.txt'
        FILTERED_FILENAME = 'filtered_text_' + str(i) + '.txt'

        FILENAME = os.path.join(FOLDER_PATH, FILENAME)
        TRANSLATED_FILENAME = os.path.join(FOLDER_PATH, TRANSLATED_FILENAME)
        FILTERED_FILENAME = os.path.join(FOLDER_PATH, FILTERED_FILENAME)

        news_parser(URL, FILENAME)
        translate_data(FILENAME, TRANSLATED_FILENAME)
        filter_data(TRANSLATED_FILENAME, FILTERED_FILENAME)
        words_dict = count_words(FILTERED_FILENAME)

        CLOUD_NAME = 'wordcloud_' + str(i) + '.jpg'
        CLOUD_PATH = os.path.join(FOLDER_PATH, CLOUD_NAME)
        create_word_cloud(FILTERED_FILENAME, CLOUD_PATH)

        ts = update_time_series(ts, words_dict, curr_date_str)
        print(ts)
        ts.to_csv('lab1.csv')

    PLOT_NAME = 'freq_plot.jpg'
    plot_freq(ts, PLOT_NAME)
