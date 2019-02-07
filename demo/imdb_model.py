from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Flatten
from keras.datasets import imdb
top_words = 100000

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_review_length = 2000
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

print('Build model...')
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())

print('Train...')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy: %.2f%%" % (scores[1]*100))

model.save('imdb_model.h5')