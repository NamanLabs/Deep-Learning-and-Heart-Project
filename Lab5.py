import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

baseline_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

baseline_model.compile(optimizer='adam', 
                       loss='sparse_categorical_crossentropy', 
                       metrics=['accuracy'])

reg_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

reg_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

EPOCHS = 15
BATCH_SIZE = 128

baseline_history = baseline_model.fit(X_train, y_train, 
                                      epochs=EPOCHS, 
                                      batch_size=BATCH_SIZE, 
                                      validation_data=(X_test, y_test), 
                                      verbose=1)

reg_history = reg_model.fit(X_train, y_train, 
                            epochs=EPOCHS, 
                            batch_size=BATCH_SIZE, 
                            validation_data=(X_test, y_test), 
                            verbose=1)

def plot_comparison(history1, history2, title):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['val_loss'], label='Baseline Val Loss', linestyle='--')
    plt.plot(history2.history['val_loss'], label='Regularized Val Loss', linewidth=2)
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['val_accuracy'], label='Baseline Val Accuracy', linestyle='--')
    plt.plot(history2.history['val_accuracy'], label='Regularized Val Accuracy', linewidth=2)
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_comparison(baseline_history, reg_history, "Baseline vs Regularized Model")