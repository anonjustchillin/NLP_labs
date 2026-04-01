from bs4 import BeautifulSoup
import requests
import re
from nltk.tokenize import RegexpTokenizer
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import os.path
import numpy as np
import stanza

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
    #print(data)

    # remove punctuation, numbers, translate eng words

    def en_to_uk(text):
        translation = GoogleTranslator(source="en", target="uk").translate(text)
        return translation

    def clean_text(r):
        text = r['Titles']
        # переклад з англ на укр
        if re.search('[a-zA-Z]', text):
            text = en_to_uk(text)
            if re.search('[a-zA-Z]', text):
                text = re.sub('[a-zA-Z]', ' ', text)

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

        text = text.lower()

        return text

    clean_data = data.apply(clean_text, axis=1)
    #print(clean_data)

    clean_data.to_csv(output_name, index=True)

    return


def filter_data_2(filename, output_name):
    # stanza.download('uk', processors='tokenize,mwt,pos,lemma')
    data = pd.read_csv(filename, index_col=0)

    nlp = stanza.Pipeline('uk', processors='tokenize,mwt,pos,lemma')

    # combine all titles and do the nlp thing
    text_arr =  data['0'].values.tolist()
    text = ". ".join(text_arr)
    #print(text)

    doc = nlp(text)
    #print(doc)

    lemmas = [word.lemma for t in doc.iter_tokens() for word in t.words]
    pos = [word.upos for t in doc.iter_tokens() for word in t.words]
    other = [word.feats for t in doc.iter_tokens() for word in t.words]
    word_lens = [word.end_char - word.start_char for t in doc.iter_tokens() for word in t.words]

    tokens_data = pd.DataFrame(
        {
            'Word': lemmas,
            'POS': pos,
            'Length': word_lens,
            'Note': other
        }
    )

    tokens_data.drop(tokens_data[tokens_data.Word == '.'].index, inplace=True)

    tokens_data.to_csv(output_name)

    return


def analyze_data(filename):
    # func for cool gradient
    def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
            cmap(np.linspace(min_val, max_val, n)))
        return new_cmap
    # word cloud
    def show_cloud(data):
        wordcloud = WordCloud().generate(data)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        return
    # freq barplot
    def show_freq_plot(df, title='Word Count', print_data=True):
        data = df['Word'].value_counts()
        if print_data: print(data.head(20))

        x = data.head(15).index.tolist()
        y = data.head(15).values.tolist()

        fig, ax = plt.subplots(figsize=[12, 8])
        bars = ax.bar(x, y)
        plt.grid(True, alpha=0.3)

        y_min, y_max = ax.get_ylim()
        grad = np.atleast_2d(np.linspace(0, 1, 256)).T
        ax = bars[0].axes
        lim = ax.get_xlim() + ax.get_ylim()
        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor("none")
            x, _ = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()

            c_map = truncate_colormap(plt.cm.plasma, min_val=0,
                                      max_val=(h - y_min) / (y_max - y_min))
            ax.imshow(grad, extent=[x, x + w, h, y_min], aspect="auto", zorder=0,
                      cmap=c_map)
        ax.axis(lim)

        plt.title(title)
        plt.xticks(rotation=20)
        plt.show()
        return

    # pos barplot
    def show_pos_plot(df, title='POS Count', print_data=True):
        data = df['POS'].value_counts()
        if print_data: print(data)

        x = data.index.tolist()
        y = data.values.tolist()

        fig, ax = plt.subplots(figsize=[12, 8])
        bars = ax.bar(x, y)
        plt.grid(True, alpha=0.3)

        y_min, y_max = ax.get_ylim()
        grad = np.atleast_2d(np.linspace(0, 1, 256)).T
        ax = bars[0].axes
        lim = ax.get_xlim() + ax.get_ylim()
        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor("none")
            x, _ = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()

            c_map = truncate_colormap(plt.cm.plasma, min_val=0,
                                      max_val=(h - y_min) / (y_max - y_min))
            ax.imshow(grad, extent=[x, x + w, h, y_min], aspect="auto", zorder=0,
                      cmap=c_map)
        ax.axis(lim)

        plt.title(title)
        plt.xticks(rotation=45)
        plt.show()
        return
    # word length hist
    def show_len_hist(df, title='Length Histogram', print_data=True):
        data = df['Length']
        if print_data:
            print(data.sort_values(ascending=False).head(20))
            print()
            print(data.sort_values(ascending=True).head(20))

        data.plot.hist()
        plt.title(title)
        plt.show()

        return

    data = pd.read_csv(filename, index_col=0)

    data_2 = data.copy()
    selected_pos = ['ADP', 'PART', 'DET', 'SCONJ', 'CCONJ']
    #print(data_2[data_2['POS'].isin(selected_pos)])
    data_2.drop(data_2[data_2['POS'].isin(selected_pos)].index, inplace=True)

    text_arr = data_2['Word'].values.tolist()
    text = " ".join(text_arr)
    show_cloud(text)

    show_freq_plot(data)
    print()
    show_freq_plot(data_2)
    print()
    show_pos_plot(data)
    print()
    show_pos_plot(data_2, print_data=False)
    print()
    show_len_hist(data)
    print()
    show_len_hist(data_2)

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
    if not os.path.exists(CLEAN_1_FILENAME_PATH):
        filter_data_1(RAW_FILENAME_PATH, CLEAN_1_FILENAME_PATH)

    CLEAN_2_FILENAME_PATH = os.path.join(FOLDER_PATH, "cleaned_text_2.csv")
    if not os.path.exists(CLEAN_2_FILENAME_PATH):
        filter_data_2(CLEAN_1_FILENAME_PATH, CLEAN_2_FILENAME_PATH)

    analyze_data(CLEAN_2_FILENAME_PATH)
