import numpy as np
import tensorflow as tf
from tensorflow import keras 


# adapted from https://www.tensorflow.org/tutorials/eager/custom_layers
class SoniaLayer(keras.layers.Layer):

    def __init__(self, output_dim):
        super(SoniaLayer, self).__init__()
        self.output_dim = output_dim
        self.l_rate = 0.5
 
    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", 
                                    shape=[input_shape[-1].value,
                                        self.output_dim])

    def call(self, input):	# TODO my understanding is that this is the feed-forward part, so this is what a finished net would do to the input 
        # return tf.nn.relu(tf.matmul(input,self.kernel)) control for custom layer by trying Dense
        A = self.kernel - tf.stack([input for __ in range(self.output_dim)], axis=-1)
        B = tf.square(A)
        C = tf.reduce_sum(B, axis=1)	# sum across all i, i.e. across the input dimension. WHY is that dimension 0 and not dimension 1???
        D = tf.sqrt(C)
#         return tf.reshape(tf.nn.tanh(D), [1, self.output_dim])	# TODO this ranges from [-1, 1] I think but in the paper it says it should be [0, 1] (page 76). What's up?. Also, tf.keras.layers.Layer doesn't automatically have an activation attribute (or support for the kwarg). Can I just stick it in so that this activation can be applied in a more tf-ish way, or should I leave it hardcoded? Will it be a part of the training this way?
        return tf.nn.tanh(D)

num_classes = 10 

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(keras.layers.Conv2D(8, (5, 5), input_shape=x_train.shape[1:])) # , activation='relu')
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(8, (5, 5))) # activation='relu'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(20)) # , activation='relu')
# model.add(keras.layers.Activation('relu'))
model.add(SoniaLayer(20))
print model.output_shape, 'after sonia'
model.add(keras.layers.Dense(num_classes)) # , activation='softmax')
model.add(keras.layers.Activation('softmax'))
print model.output_shape, 'after dense'

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print tf.get_default_graph()
model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
model.save_weights('soniaCnn.h5')


