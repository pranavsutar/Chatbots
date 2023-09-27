import nltk, numpy as np


# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def bag_ofwords(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bg = np.zeros(len(all_words), dtype=np.float32)
    for i in range(len(all_words)):
        if all_words[i] in tokenized_sentence:
            bg[i] = 1.0
    return bg
