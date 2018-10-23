import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import keras
from tensorflow.image import rgb_to_grayscale

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = rgb_to_grayscale(x_train)
# x_test = rgb_to_grayscale(x_test)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print x_train.shape[1:]

model = Sequential()
model.add(Conv2D(8, (5, 5), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(8, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
model.save_weights('simpleCnn.h5')
