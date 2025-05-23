
import spacy
import re

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def spacy_preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)
=======
# preprocess.py
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])

def spacy_preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])
