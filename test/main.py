from google import genai
from google.genai import types
import time
import pandas as pd
from gtts import gTTS
from pygame import mixer
from io import BytesIO
import os
from ddgs import DDGS
from get_data import create_df
from key import *
import warnings
warnings.filterwarnings('ignore')

api_key = API_KEY

client = genai.Client(api_key=api_key)
config=types.GenerateContentConfig(
        temperature=0.1
    )
model_name = "gemini-3.5-flash"

lang = 'uk'

PROJECT_PATH = 'D:\\uni\\3курс\\NLP\\NLP_labs\\test'

def voiceover(text):
    mixer.init()

    speak = gTTS(text=text, lang=lang, slow=False)
    print(text)
    mp3_fp = BytesIO()
    speak.write_to_fp(mp3_fp)

    mp3_fp.seek(0)
    mixer.music.load(mp3_fp, "mp3")
    mixer.music.play()
    while mixer.music.get_busy():
        pass
    return

def search_web(query, max_results=3):
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "Не знайдено інформації в інтернеті."

        search_context = ""
        for r in results:
            search_context += f"- Заголовок: {r['title']}\n  Фрагмент: {r['body']}\n\n"
        return search_context
    except Exception as e:
        return f"Помилка: {e}"

def analyze_news(news_text):
    web_context = search_web(news_text)

    prompt = f"""
        Ти — експерт з медіаграмотності, фактчекер, аналітик дезінформації.
        Твоє завдання — проаналізувати текст новини, спираючись на надані результати веб-пошуку.

        Зверни увагу на:
        1. Емоційне забарвлення, клікбейт та маніпуляції в самій новині.
        2. Чи підтверджуються факти з новини результатами пошуку, чи навпаки - є спростування.
        3. Логічні розриви у твердженнях.

        Надай відповідь строго у такому форматі:
        Вердикт: [Фейк / Ймовірний фейк / Ймовірна правда / Правда]
        Обґрунтування: [Коротке пояснення на 3-4 речення. Обов'язково вкажи, чи знайшлися докази/спростування в інтернеті].

        Текст новини:
        "{news_text}"

        Результати веб-пошуку (Контекст):
        {web_context}
        """
    time.sleep(2)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"Помилка: {e}"


def start_analysis():
    filepath = os.path.join(PROJECT_PATH, 'news.csv')
    df = pd.read_csv(filepath, index_col=0)

    text = "Починаю аналіз..."
    voiceover(text)

    df['LLM_result'] = df['News'].apply(analyze_news)

    text = "Готово!"
    voiceover(text)
    print()

    for i, row in df.iterrows():
        text = f"Новина {i+1}"
        voiceover(text)
        text = f"Джерело: {row['Site']}"
        voiceover(text)
        text = f"Текст: {row['News']}"
        voiceover(text)
        text = f"{row['LLM_result']}"
        voiceover(text)
        print()

    df.to_csv(filepath, index=True)


if __name__ == "__main__":
    print()
    text = 'Добрий день! Я зараз аналізуватиму надані у файлі news.csv новини на достовірність.'
    voiceover(text)
    create_df()
    start_analysis()