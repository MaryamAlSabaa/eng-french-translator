import string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


with open('eng_tokenizer.pkl', 'rb') as f:
    eng_tokenizer = pickle.load(f)
with open('fr_tokenizer.pkl', 'rb') as f:
    fr_tokenizer = pickle.load(f)


en_max = 15
fr_max = 21


def clean_sentence(sentence):
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)
    translator = str.maketrans('', '', string.punctuation)
    return sentence.translate(translator).lower()

def translate_sentence(model, sentence):
        y_id_to_word = {value: key for key, value in fr_tokenizer.word_index.items()} #model will give numbers of french words indicies, so change from incdicies to words
        y_id_to_word[0] = "<PAD>" #means not part of vocab

        sentence =eng_tokenizer.texts_to_sequences([sentence]) #encode

        sentence = pad_sequences(sentence, maxlen=en_max, padding="post") #pad
        predictions = model.predict(sentence) #predict

        translated_sentence = " ".join([y_id_to_word[np.argmax(x)] for x in predictions[0]]) #taking the max propability using argmax
        return translated_sentence