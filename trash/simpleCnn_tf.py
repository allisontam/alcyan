import numpy as np
import tensorflow as tf
from tensorflow import keras


num_classes = 10 

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(keras.layers.Conv2D(8, (5, 5), input_shape=x_train.shape[1:]))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(8, (5, 5)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(20))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(num_classes))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
# model.save_weights('simpleCnn.h5')

