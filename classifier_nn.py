import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy
import pickle
def create_classifier():
    pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)

    pickle_in = open("Y.pickle","rb")
    Y = pickle.load(pickle_in)

    X = X/255.0
    print(X.shape)
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Dense(3, activation='softmax'))

    Y = tf.keras.utils.to_categorical(Y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X, Y, epochs=10,  batch_size = 8,)
    return model


model = create_classifier()
model.save('my_model')

#model = tf.keras.models.load_model('my_model')

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)
pickle_in = open("Y_test.pickle","rb")
Y_test = pickle.load(pickle_in)


Y_test = tf.keras.utils.to_categorical(Y_test)
_, accuracy = model.evaluate(X_test, Y_test, batch_size=1)
print(accuracy)



