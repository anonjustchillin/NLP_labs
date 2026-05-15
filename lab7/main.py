from bs4 import BeautifulSoup
import requests
from gtts import gTTS
import speech_recognition as sr
from deep_translator import GoogleTranslator
import pandas as pd
import os.path

"""
Розробити програмний скрипт, що озвучує стрічку новин голосом обраного сайту. Функціонал скрипта має включати:
• парсинг сайту новин з понад 50 записів / повідомлень;
• фіксація результатів парсингу у файлі;
• озвучування результатів парсингу українською та англійською мовами, вибір мови озвучування здійснюється голосом.
"""

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\\lab7'

URL = 'https://suspilne.media/'
URL_NAME = 'suspilne'


def news_parser(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    quotes = soup.find_all('span', class_='c-article-card__headline-inner')

    # to pandas df
    data = pd.DataFrame(columns=['Titles_UK'])
    for quote in quotes:
        data.loc[len(data)] = quote.text
    filepath =os.path.join(PROJECT_PATH, 'data\\raw_data.csv')
    data.to_csv(filepath, index=True)
    return

def translate():
    def uk_to_en(r):
        text = r['Titles_UK']
        translation = GoogleTranslator(source="uk", target="en").translate(text)
        return translation

    filepath = os.path.join(PROJECT_PATH, 'data\\raw_data.csv')
    df = pd.read_csv(filepath, index_col=0)
    df['Titles_EN'] = df.apply(uk_to_en, axis=1)

    filepath = os.path.join(PROJECT_PATH, 'data\\data.csv')
    df.to_csv(filepath, index=False)
    return

def text_to_speech(lang='uk'):
    filepath = os.path.join(PROJECT_PATH, 'data')

    df = pd.read_csv(os.path.join(filepath, 'data.csv'))

    if lang == 'en':
        print('Text-to-Speech English...')
        filename='news_en.mp3'
        texts = df['Titles_EN'].tolist()
    else:
        print('Text-to-Speech Українською...')
        filename = 'news_uk.mp3'
        texts = df['Titles_UK'].tolist()

    audio_filepath = os.path.join(filepath, filename)
    with open(audio_filepath, 'wb') as fp:
        for text in texts:
            tts = gTTS(text=text, lang=lang)
            tts.write_to_fp(fp)

    return


def choose_lang():
    lang = 'uk-ua'

    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print('Connected to microphone...')
        r.adjust_for_ambient_noise(source, duration=1)
        print('Choose text-to-speech language: English / Українська')
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=7)
        except sr.WaitTimeoutError:
            print("No speech detected. Using default 'Українська'")
            return 'uk'
    try:
        audio_text = r.recognize_google(audio, language=lang).lower()
        print(f'You: {audio_text}')
        if "ukrainian" in audio_text or "українськ" in audio_text or "укр" in audio_text:
            return "uk"
        elif "english" in audio_text or "англійськ" in audio_text or "англ" in audio_text:
            return "en"
        else:
            print("Unknown language.")
    except:
        print("What? No speech detected.")

    return 'uk'


if __name__ == '__main__':
    print(f'Parsing site {URL_NAME}...')
    news_parser(URL)
    print('Translating news...')
    translate()
    lang = choose_lang()
    if lang is not None:
        text_to_speech(lang=lang)

