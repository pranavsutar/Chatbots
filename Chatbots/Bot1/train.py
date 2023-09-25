import json, nltk, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

intents = json.loads(open('intents.json').read())

print(str(intents)[:60])
print(type(intents))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', "'s", "'ve", "'re", "'ll", "'m", "'d", "'t", 'a', 'an', 'the', 'is', 'are', 'was', 'were', 'am', 'do', 'does', 'did', 'can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'have', 'has', 'had', 'of', 'at', 'to', 'in', 'on', 'for', 'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = tokenize(pattern)
        # words.extend(word_list)
        for word in word_list:
            if word not in ignore_letters:
                words.append(stem(word))
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(words)
for d in documents:
    print(d)

trainX = []
trainY = []

for doc in documents:
    bag = bag_ofwords(doc[0], words)
    trainX.append(bag)
    label = classes.index(doc[1])
    trainY.append(label)

trainX = np.array(trainX)
trainY = np.array(trainY)

print(trainX)
print(trainY)

