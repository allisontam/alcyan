import numpy as np
import tensorflow as tf
from tensorflow.python import keras 

num_classes = 10 

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(keras.Conv2D(8, (5, 5), input_shape=x_train.shape[1:]), activation='relu')
model.add(keras.Conv2D(8, (5, 5), activation='relu')
model.add(keras.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.Flatten())
# model.add(keras.Dense(20), activation='relu')
model.add(SoniaLayer(20))
model.add(keras.Dense(num_classes), activation='softmax')

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
class SoniaLayer_old(Layer):

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


# adapted from https://www.tensorflow.org/tutorials/eager/custom_layers
class SoniaLayer(keras.layers.Layer):

    def __init__(self, output_dim):
        super(SoniaLayer, self).__init__()
        self.output_dim = output_dim 
    
    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", 
                                    shape=[input_shape[-1].value, 
                                           self.num_outputs])
    	
    def call(self, input):	# TODO my understanding is that this is the feed-forward part, so this is what a finished net would do to the input 
        A = self.kernel - tf.stack([input for __ in range(output_dim)])
        B = tf.square(A)
        C = tf.reduce_sum(B, axis=1)	# sum across all i, i.e. across the input dimension
        D = tf.sqrt(C)
        return tf.nn.tanh(D)	# TODO this ranges from [-1, 1] I think but in the paper it says it should be [0, 1] (page 76). What's up?. Also, tf.keras.layers.Layer doesn't automatically have an activation attribute (or support for the kwarg). Can I just stick it in so that this activation can be applied in a more tf-ish way, or should I leave it hardcoded? Will it be a part of the training this way?









