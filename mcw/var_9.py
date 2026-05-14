import pandas as pd
import os
import re
from get_data import TextManipulation
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings

"""
Організуйте автоматизовану класифікацію відгуків на платформі електронної комерції.
"""
warnings.filterwarnings(action='ignore')

pattern = r"/([^/]+)/p\d+"

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\\mcw'

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def predict_sentiment(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Дуже Негативний",
                     1: "Негативний",
                     2: "Нейтральний",
                     3: "Позитивний",
                     4: "Дуже Позитивний"}
    return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]


def get_url():
    print('Введіть URL на сторінку товару')
    while True:
        url = input('URL: ')
        if 'rozetka.com.ua' in url:
            break
    name = re.search(pattern, url).group(1)
    print('Назва товару: ' + name)
    return url, name


if __name__ == '__main__':
    print('Класифікація відгуків на платформі Rozetka')

    first = True
    if os.path.exists(os.path.join(PROJECT_PATH, 'data.csv')):
        while True:
            choice = input('Переписати минулий набір відгуків? (Т/Н) ')
            if choice == 'Т':
                first = True
                break
            elif choice == 'Н':
                first = False
                break

    text_manipulation = TextManipulation('data.csv')

    if os.path.exists(os.path.join(PROJECT_PATH, 'data.csv')):
        while True:
            choice = input('Додати відгуки до набору даних? (Т/Н) ')
            if choice == 'Т':
                url, name = get_url()
                text_manipulation.parse_data(url, name, first)
                break
            elif choice == 'Н':
                break

    print()

    df = pd.read_csv(text_manipulation.filepath, index_col=0)
    sentiment = predict_sentiment(df['CleanedReview'].tolist())
    text_manipulation.add_sentiment(sentiment)

    print('Результати класифікації')
    df = pd.read_csv(text_manipulation.filepath, index_col=0)
    print(df.Rating.value_counts())
    for i in range(len(df)):
        print(f'{i}. {df.loc[i, "CleanedReview"]}')
        print(f'Оцінка: {df.loc[i, "Rating"]}')
        print()
