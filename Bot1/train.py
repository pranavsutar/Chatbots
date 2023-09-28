import json, nltk, numpy as np, random # , torch, torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

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

def getintents():
    return intents
# print(str(intents)[:60])
# print(type(intents))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', "'s", "'ve", "'re", "'ll", "'m", "'d", "'t", 'a', 'an', 'the', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'have', 'has', 'had', 'of', 'at', 'to', 'in', 'on', 'for', 'with', 'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'no', 'nor', 'not', 'so', 'than', 'too',  's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
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
# for d in documents:
#     print(d)

trainX = []
trainY = []

for doc in documents:
    bag = bag_ofwords(doc[0], words)
    trainX.append(bag)
    label = classes.index(doc[1])
    trainY.append(label)

trainX = np.array(trainX)
trainY = np.array(trainY)

# print(trainX)
# print(trainY)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(trainX, trainY)

import pickle
pickle.dump(model, open('modelnaiveb.pkl', 'wb'))

# SVM
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1, gamma='auto', probability=True)
model.fit(trainX, trainY)

pickle.dump(model, open('modelsvm.pkl', 'wb'))


# load model
# model = pickle.load(open('modelsvm.pkl', 'rb'))


# Now input a sentence and predict the class
# sentence = "How long does delivery take?"
# sentence = tokenize(sentence)
# sentence = bag_ofwords(sentence, words)
# sentence = np.array(sentence).reshape(1,-1)
# print(sentence)
# print(model.predict(sentence))
# # print(classes[])
# import random
# print(random.choice(intents['intents'][model.predict(sentence)[0]]['responses']))


# while True:
#     sentence = input('You: ')
#     sentence = tokenize(sentence)
#     sentence = bag_ofwords(sentence, words)
#     sentence = np.array(sentence).reshape(1,-1)
#     response = random.choice(intents['intents'][model.predict(sentence)[0]]['responses'])
#      # Check the probability of the predicted intent for naive bayes model
#     prob = np.max(model.predict_proba(sentence))*100
#     if prob < 15:
#         response = "I don't understand, try again"
#     print(f'Confidence: {prob:.2f}%')


#     print(f'Bot: {response}')