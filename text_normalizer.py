import re
import nltk
import spacy
import unicodedata
import unidecode

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer


tokenizer = ToktokTokenizer()
nltk.download('stopwords')
# python -m spacy download en_core_web_sm
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()


def stem_text(text:str, stemmer = PorterStemmer()):
    return " ".join([stemmer.stem(token) for token in tokenizer.tokenize(text)])


def lemmatize_text(text):
    tokens = nlp(text)
    return " ".join([token.lemma_ for token in tokens])


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    for key, value in contraction_mapping.items():
      text = text.replace(key, value)
    return text
    
def remove_accented_chars(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')

def remove_special_chars(text: str, remove_digits: bool=False):
    return ''.join([c for c in text if ((c.isalpha() if remove_digits else c.isalnum()) or c.isspace())])


def remove_stopwords(text: str, is_lower_case=True, stopwords=stopword_list):
    text = text.lower() if is_lower_case else text
    return " ".join([token for token in tokenizer.tokenize(text) if token not in stopwords])


def remove_extra_new_lines(text: str):
    return text.replace("\n", " ")


def remove_extra_whitespace(text: str):
    return re.sub(' +', ' ', text)
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    # if isinstance(corpus, list):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
          # Remove HTML
          if html_stripping:
              doc = remove_html_tags(doc)
              
          # Remove extra newlines
          doc = remove_extra_new_lines(doc)
          
          # Lowercase the text    
          if text_lower_case:
              doc = doc.lower()
          
          # Remove accented chars
          if accented_char_removal:
              doc = remove_accented_chars(doc)
              
          # Expand contractions    
          if contraction_expansion:
              doc = expand_contractions(doc)

          # Remove special chars and\or digits    
          if special_char_removal:
              doc = remove_special_chars(
                  doc,
                  remove_digits=remove_digits
              )  

          # Remove stopwords
          if stopword_removal:
              doc = remove_stopwords(
                  doc,
                  is_lower_case=text_lower_case,
                  stopwords=stopwords
              )
              
          # Lemmatize text
          if text_lemmatization:
              doc = lemmatize_text(doc)
              
          # Stemming text
          if text_stemming and not text_lemmatization:
              doc = stem_text(doc)
              
          # Remove extra whitespace
          doc = remove_extra_whitespace(doc)
          
          # Remove extra whitespace
          doc = remove_extra_whitespace(doc)
          doc = doc.strip()
              
          normalized_corpus.append(doc)
          
    return normalized_corpus

def normalize_data(
    data,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    if isinstance(data, list):
      return normalize_corpus(data, html_stripping, contraction_expansion,
                              accented_char_removal, text_lower_case, text_stemming,
                              text_lemmatization, special_char_removal,
                              remove_digits, stopword_removal, stopwords)
    elif isinstance(data, tuple):
      normalized_data = []
      for corpus in data:
        normalized_data.append(normalize_corpus(corpus, html_stripping, contraction_expansion,
                                                accented_char_removal, text_lower_case, text_stemming,
                                                text_lemmatization, special_char_removal,
                                                remove_digits, stopword_removal, stopwords))
      return normalized_data
    else:
      raise TypeError("data sholud be either a list or a tuple")
    