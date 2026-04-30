from bs4 import BeautifulSoup
import requests
import re
from deep_translator import GoogleTranslator
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import pandas as pd
import os.path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.decomposition import PCA
import stanza
import random
import warnings

nlp = stanza.Pipeline('uk', processors='tokenize,mwt,pos,lemma')
selected_pos = ['ADP', 'PART', 'DET', 'SCONJ', 'CCONJ']

warnings.filterwarnings(action='ignore')

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

def view_site(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    print(soup)
    return


def news_parser(url, name, filename):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    if name == NAME_OLX:
        # h4 css-wlcw7o
        quotes = soup.find_all('h4', class_='css-wlcw7o')
    elif name == NAME_AUTORIA:
        quotes = soup.find_all('div', class_='titleS')
    else:
        quotes = soup.find_all('div', class_='card-name')

    # to pandas df
    data = pd.DataFrame(columns=['Title'])
    for quote in quotes:
        data.loc[len(data)] = quote.text
    data.to_csv(filename, index=True)
    #print(data)
    #print()
    return


def clean_text(text):
    def lemmatize_and_stuff(text):
        doc = nlp(text)
        lemmas = [word.lemma for t in doc.iter_tokens() for word in t.words]
        pos = [word.upos for t in doc.iter_tokens() for word in t.words]
        tokens_data = pd.DataFrame(
            {
                'Word': lemmas,
                'POS': pos
            }
        )
        tokens_data.drop(tokens_data[tokens_data['POS'].isin(selected_pos)].index, inplace=True)

        return " ".join(tokens_data['Word'])

    def rus_to_uk(text):
        translation = GoogleTranslator(source="russian", target="ukrainian").translate(text)
        return translation

    # переклад з рос на українську
    text = rus_to_uk(text)

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

    text = lemmatize_and_stuff(text)
    # пунктуація (ще раз)
    text = re.sub(r"[^\w\s]", ' ', text)

    # extra spaces
    text = " ".join(text.split())

    return text.lower()


def similarity_analysis(text1, text2, text1_name, text2_name, N=range(10, 21, 5)):
    def list_to_str(arr):
        return ' '.join(arr)

    def plot_2d_vectors(vectors, words, n, title=''):
        plt.figure(figsize=(12, 5))
        plt.scatter(vectors['PC1'], vectors['PC2'])
        annotations = []
        for word in words:
            x_space = random.uniform(0.02, 0.09)
            y_space = random.uniform(0.02, 0.09)
            annotations.append(plt.text(vectors.loc[word, 'PC1'] + x_space,
                                        vectors.loc[word, 'PC2'] + y_space,
                                        word,
                                        fontsize=10))

        plt.title(f'2D Visualization of {n} word vectors' + title)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True, alpha=0.4)
        adjust_text(annotations, expand=(1.2, 2),
                    arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        plt.show()

    def plot_text_heatmap(df, title=''):
        plt.figure(figsize=(12, 5))
        sns.heatmap(df, annot=True, cmap="rocket")
        plt.title(f'Heatmap {text1_name} and {text2_name}'+title)
        plt.show()

    text1 = list_to_str(text1)
    text2 = list_to_str(text2)

    documents = [text1, text2]
    count_vectorizer = CountVectorizer(stop_words="english")
    sparse_matrix = count_vectorizer.fit_transform(documents)

    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix,
                      columns=count_vectorizer.get_feature_names_out(),
                      index=[text1_name, text2_name])

    eucld_dist = euclidean_distances(df,df)
    cos_similarity = cosine_similarity(df, df)
    plot_text_heatmap(eucld_dist, ' (euclidean distances)')
    plot_text_heatmap(cos_similarity, ' (cosine similarity)')

    pca = PCA(n_components=2)
    word_vectors = df.T
    reduced_vectors = pca.fit_transform(word_vectors)
    pca_df = pd.DataFrame(reduced_vectors,columns=['PC1', 'PC2'],index=word_vectors.index)
    word_freq = word_vectors.sum(axis=1)

    for n in N:
        top_words = word_freq.nlargest(n).index
        print(f'PCA dataframe {text1_name} vs {text2_name} (only top {n} words by frequency)')
        print(pca_df.loc[top_words])
        print()
        plot_2d_vectors(pca_df.loc[top_words], top_words, f' ({text1_name} vs {text2_name})')


if __name__ == '__main__':
    flag = False
    for name in URLS.keys():
        curr_folder = os.path.join(PROJECT_PATH, name)
        if os.path.exists(curr_folder):
            flag = True
            break

    if not flag:
        for name, urls in URLS.items():
            print(f'PARSING {name}...')
            curr_folder = os.path.join(PROJECT_PATH, name)
            os.makedirs(curr_folder)
            for i in range(3):
                file = os.path.join(curr_folder, name+'_'+str(i)+'.csv')
                news_parser(urls[i], name, file)
            print()

        print('MAKING A DATAFRAME')
        start_idx = 0
        df = pd.read_csv(os.path.join(os.path.join(PROJECT_PATH, NAME_OLX), NAME_OLX+'_'+str(start_idx)+'.csv'), index_col=0)
        df['Car_type'] = start_idx
        df['Site'] = NAME_OLX
        for name, urls in URLS.items():
            curr_folder = os.path.join(PROJECT_PATH, name)
            for i in range(3):
                if name == NAME_OLX and i == 0: continue
                file = os.path.join(curr_folder, name+'_'+str(i)+'.csv')
                temp_df = pd.read_csv(file, index_col=0)
                if name == NAME_AUTORIA:
                    temp_df = temp_df.iloc[:-3] # то не оголошення
                temp_df['Car_type'] = i
                temp_df['Site'] = name
                df = pd.concat([df, temp_df]).reset_index(drop=True)

        df.to_csv(RAW_FILENAME, index=True)

        print(df.describe())
        print(df.head())
        print(df.tail())
        print(df['Car_type'].value_counts())
        print(df['Site'].value_counts())
        print()
    else:
        df = pd.read_csv(RAW_FILENAME, index_col=0)

    print('CLEANING TEXT')
    print(f'Raw dataframe length: {len(df)}')
    if not os.path.exists(CLEANED_FILENAME):
        df['Title_cleaned'] = df['Title'].map(clean_text)
        df = df[['Title', 'Title_cleaned', 'Car_type', 'Site']]
    else:
        df = pd.read_csv(CLEANED_FILENAME, index_col=0)

    #print(df['Title_cleaned'].head())
    #print(df['Title_cleaned'].tail())

    is_nan = df.isnull().sum().any()
    print(f'Are there any NaN values? {is_nan}')
    if is_nan:
        df.dropna(inplace=True)
        print(f'Cleaned dataframe length: {len(df)}')
    print()
    df.to_csv(CLEANED_FILENAME, index=True)

    print('SIMILARITY ANALYSIS')
    car_0 = df.loc[df["Car_type"] == 0, 'Title_cleaned'].to_list()
    car_1 = df.loc[df["Car_type"] == 1, 'Title_cleaned'].to_list()
    car_2 = df.loc[df["Car_type"] == 2, 'Title_cleaned'].to_list()

    similarity_analysis(car_0, car_1, AUTO_TYPES[0], AUTO_TYPES[1])
    similarity_analysis(car_0, car_2, AUTO_TYPES[0], AUTO_TYPES[2])
    similarity_analysis(car_1, car_2, AUTO_TYPES[1], AUTO_TYPES[2])