import tensorflow as tf
import time

# Load the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape data to include channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a more complex neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Record the start time
start_time = time.time()

# Specify the device to use for training and evaluation
with tf.device('/device:GPU:0'):
    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64)

    # Evaluate the model
    evaluation = model.evaluate(x_test, y_test, verbose=2)

# Record the end time
end_time = time.time()

# Calculate and print the total training time
total_time = end_time - start_time
print("Total training time: {:.2f} seconds".format(total_time))
print("Test loss, Test accuracy:", evaluation)
