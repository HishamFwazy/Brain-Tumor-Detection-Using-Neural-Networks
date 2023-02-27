import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X1 = tf.constant([2, 3, 4, 5, 6, 7])
X2 = tf.constant([2, 3, 4, 5, 6, 7])
yTrain = tf.constant([4, 6, 8, 10, 12, 14])

input1 = keras.Input(shape=(1,))
input2 = keras.Input(shape=(1,))

x = layers.concatenate([input1, input2])
x = layers.Dense(8, activation='relu')(x)
outputs = layers.Dense(2)(x)
mlp = keras.Model(input = [input1, input2], output = outputs)

mlp.summary()

mlp.compile(loss='mean_squared_error',
            optimizer='adam', metrics=['accuracy'])

mlp.fit([X1, X2], yTrain, batch_size=1, epochs=10, validation_split=0.2,
        shuffle=True)

mlp.evaluate([X1, X2], yTrain)