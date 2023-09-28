import pickle, numpy as np, random
from nltk_utils import bag_ofwords, tokenize
from train import intents, words
model = pickle.load(open('modelsvm.pkl', 'rb'))
# model = pickle.load(open('modelnaiveb.pkl', 'rb'))

while True:
    sentence = input('You: ')
    if sentence.lower().strip() == 'quit':
        break
    sentence = tokenize(sentence)
    sentence = bag_ofwords(sentence, words)
    sentence = np.array(sentence).reshape(1,-1)
    response = random.choice(intents['intents'][model.predict(sentence)[0]]['responses'])
    # Check the probability of the predicted intent
    prob = np.max(model.predict_proba(sentence))*100
    if prob < 15:
        response = "I don't understand, try again"
    # print(f'Confidence: {prob:.2f}%')
    print(f'Bot: {response}')