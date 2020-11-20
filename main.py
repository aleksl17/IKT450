# This is the main file
from input import img_input
from convolution import define_conv2d
from classifier import define_classifier
from inversion_counting import define_puzzler

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#do data stuff here
x_train, x_train_shuffle, y_train_classify, y_train_shuffle, x_test, x_test_shuffle, y_test_classify, y_test_shuffle = img_input(500)


y_train_test = []
for y in y_train_shuffle:
    temp = []
    for i in range(9):
        for j in range(9):
            if(y[i*9 + j] == 1):
                temp.append(j/9)
    y_train_test.append(temp)

y_test_test = []
for y in y_test_shuffle:
    temp = []
    for i in range(9):
        for j in range(9):
            if(y[i*9 + j] == 1):
                temp.append(j/9)
    y_test_test.append(temp)


#get sub networks
conv_model = define_conv2d()
clas_model = define_classifier()
puz_model = define_puzzler()

#define networks
#puzzle model
puzzle_model = Sequential()
puzzle_model.add(conv_model)
puzzle_model.add(puz_model)
puzzle_model.compile(loss='mse', optimizer=Adam(lr=0.01), metrics=['accuracy'])
print("Puzzle Model:")
puzzle_model.summary()

classifier_model = Sequential()
classifier_model.add(conv_model)
classifier_model.add(clas_model)
opt = Adam(lr=0.0002, beta_1=0.5)
classifier_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print("Classifier Model:")
classifier_model.summary()

#train puzzle here
print(y_test_shuffle)
puzzle_model.fit(x_train_shuffle/255, np.asarray(y_train_test), epochs=1000, shuffle=True)

y = puzzle_model.predict(x_test_shuffle/255)

print("0")
print(y[0])
print(y_test_test[0])
print("1")
print(y[1])
print(y_test_test[1])

#train classifier here


