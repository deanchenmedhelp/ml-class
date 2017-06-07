# using a validation set properly

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

from keras.layers import Dropout
from keras.utils import np_utils

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]

y_test = np_utils.to_categorical(y_test)

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))

# adding a 2nd layer
# relu is like a rectifier activation function
model.add(Dense(30, activation='relu'))

# Dense gives a fully connected NN layer (fully connected
# means each node in the layer is connected to every node 
# in the previous layer)
# softmax activation divides the probability 0..1 for
# the output categories
model.add(Dense(num_classes, activation='softmax'))


# the categorical crossentropy loss function (error function)
# seeks to make one output response most pronounced amongst the rest
# Adam is a gradient descent optimization algorithm in tensorflow
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test))


# predict
pred = model.predict(X_test)

digit = X_test[3]

str = ""
for i in range(digit.shape[0]):
    for j in range(digit.shape[1]):
        if digit[i][j] == 0:
            str += " "
        elif digit[i][j] < 128:
            str += "."
        else:
            str += "X"
    str += "\n"

print(str)
print(pred[3])

