import pickle, numpy as np, random
from nltk_utils import bag_ofwords, tokenize
from train import intents, words
model = pickle.load(open('modelsvm.pkl', 'rb'))

while True:
    sentence = input('You: ')
    sentence = tokenize(sentence)
    sentence = bag_ofwords(sentence, words)
    sentence = np.array(sentence).reshape(1,-1)
    response = random.choice(intents['intents'][model.predict(sentence)[0]]['responses'])
    print(f'Bot: {response}')