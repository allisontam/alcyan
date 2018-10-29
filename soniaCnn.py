from keras import backend as K
from keras.datasets import cifar10
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np

num_classes = 10 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(8, (5, 5), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(8, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pol_size=(2, 2)))

model.add(Flatten())
# model.add(Dense(20))
# model.add(Activation('relu'))
model.add(SoniaLayer(20))
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
model.save_weights('soniaCnn.h5')


# adapted from https://keras.io/layers/writing-your-own-keras-layers/
class SoniaLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SoniaLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(SoniaLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        W = self.get_weights()
        


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)












