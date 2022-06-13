import demoji
import re
from nltk.corpus import stopwords
import unicodedata as uni
import spacy
from nltk.stem import WordNetLemmatizer


sp = spacy.load("en_core_web_sm")

def remove_url(text):
    text = re.sub(r"http\S+", "", text)
    return text


def handle_emoji(string):
    emojis = demoji.findall(string)

    for emoji in emojis:
        string = string.replace(emoji, " " + emojis[emoji].split(":")[0])

    return string

def word_tokenizer(text):
    text = text.lower()
    text = text.split()

    return text


en_stopwords = set(stopwords.words('english'))


def remove_stopwords(text):
    text = [word for word in text if word not in en_stopwords]
    return text

def lemmatization(text):

    # text = [sp(word).lemma_ for word in text]

    text = " ".join(text)
    token = sp(text)

    text = [word.lemma_ for word in token]
    return text


def preprocessing(text):
    
    text = remove_url(text) 
    text = uni.normalize('NFKD', text)
    text = handle_emoji(text)
    text = text.lower() 
    text = re.sub(r'[^\w\s]', '', text)
    text = word_tokenizer(text)
    # text = stemming(text)
    text = lemmatization(text)
    text = remove_stopwords(text)
    text = " ".join(text)

    return text
