import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

def get_default_vectorizer(use_idf=True, norm='l2'):
    return TfidfVectorizer(lowercase=True,
                           stop_words='english',
                           ngram_range=(1,3),
                           max_features=10000,
                           use_idf=False,
                           smooth_idf=True,
                           token_pattern=r"(?u)\b\w\w\w\w+\b", # at least four alphanumeric characters
                           norm=norm,
                          )


_spacy_nlp = spacy.load('en_core_web_lg', disable=['textcat', 'parser', 'ner'])
def _preprocessor(text, ):
    return ' '.join([token.lemma_.lower() for token in _spacy_nlp(text) if token.lemma_.isalpha() and len(token.lemma_)>3])

def get_spacy_processed_vectorize(use_idf=True, norm='l2'):
    stopwords = {stopword for stopword in _spacy_nlp.Defaults.stop_words if stopword==_preprocessor(stopword)}
    return TfidfVectorizer(lowercase=True,
                           preprocessor=_preprocessor,
                           stop_words=stopwords,
                           ngram_range=(1,3),
                           max_features=10000,
                           use_idf=False,
                           smooth_idf=True,
                           norm=norm,
                          )