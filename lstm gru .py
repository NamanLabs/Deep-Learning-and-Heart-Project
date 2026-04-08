import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

vocab_size = 10000
max_length = 100

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    GRU(64),
    Dense(1, activation='sigmoid')
])

gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

EPOCHS = 5
BATCH_SIZE = 128

lstm_history = lstm_model.fit(X_train, y_train,
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              validation_data=(X_test, y_test),
                              verbose=1)

gru_history = gru_model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            verbose=1)

def plot_rnn_comparison(hist_lstm, hist_gru):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist_lstm.history['val_accuracy'], label='LSTM Val Accuracy', color='blue', linewidth=2)
    plt.plot(hist_gru.history['val_accuracy'], label='GRU Val Accuracy', color='orange', linestyle='--')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_lstm.history['val_loss'], label='LSTM Val Loss', color='blue', linewidth=2)
    plt.plot(hist_gru.history['val_loss'], label='GRU Val Loss', color='orange', linestyle='--')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_rnn_comparison(lstm_history, gru_history)