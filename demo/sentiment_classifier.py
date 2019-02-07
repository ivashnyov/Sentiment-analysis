from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf


def import_model(filename):
    global model
    model = load_model(filename)
    global graph
    graph = tf.get_default_graph()
    return model


class SentimentClassifier(object):
    def __init__(self):
        self.model = import_model('imdb_model.h5')
        self.word_to_index = imdb.get_word_index()
        self.max_review_length = 2000
        self.classes_dict = {0: "negative", 1: "positive", -1: "prediction error"}

    @staticmethod
    def get_tonality(probability):
        if probability > 0.65:
            return "Good"
        if probability < 0.35:
            return "Bad"
        else:
            return "Neutral"

    def predict_text(self, text):
        try:
            words = text.split()
            review = []
            for word in words:
                if word not in self.word_to_index:
                    review.append(2)
                else:
                    review.append(self.word_to_index[word] + 3)
            review = sequence.pad_sequences([review], truncating='pre', padding='pre', maxlen=self.max_review_length)
            with graph.as_default():
                prediction = self.model.predict(review)
            return prediction[0][0]
        except Exception as e:
            print(e)
            return -1

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        return prediction
