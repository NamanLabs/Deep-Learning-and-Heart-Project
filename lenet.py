#Lenet
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

rows, cols = 28, 28

# Reshape to 4D
x_train = x_train.reshape(-1, rows, cols, 1)
x_test = x_test.reshape(-1, rows, cols, 1)

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

input_shape = (rows, cols, 1)

# Build LeNet-5 Model
def build_lenet(input_shape):
    model = tf.keras.Sequential()

    # C1
    model.add(tf.keras.layers.Conv2D(6, (5,5), activation='tanh', input_shape=input_shape))
    # S2
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

    # C3
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation='tanh'))
    # S4
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

    # Flatten
    model.add(tf.keras.layers.Flatten())

    # C5
    model.add(tf.keras.layers.Dense(120, activation='tanh'))
    # F6
    model.add(tf.keras.layers.Dense(84, activation='tanh'))

    # Output
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

lenet = build_lenet(input_shape)

# Compile
lenet.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
epochs = 5
history = lenet.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=128,
    verbose=1
)

# Evaluate
loss, acc = lenet.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Prediction on single image
image_index = 8888
plt.imshow(x_test[image_index].reshape(28,28), cmap='gray')
plt.axis('off')

pred = lenet.predict(x_test[image_index].reshape(1,28,28,1))
print("Predicted Label:", pred.argmax())










