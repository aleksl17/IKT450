# This is the main file
from input import img_input
from convolution import define_conv2d
from classifier import define_classifier
from inversion_counting import define_puzzler
from eval import compare

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD

#globals
just_classify = True
epochs = 1000
sample_size = 5000

#do data stuff here
x_train, x_train_shuffle, y_train_classify, y_train_shuffle, x_test, x_test_shuffle, y_test_classify, y_test_shuffle = img_input(sample_size)


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
puzzle_model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
print("Puzzle Model:")
puzzle_model.summary()

classifier_model = Sequential()
classifier_model.add(conv_model)
classifier_model.add(clas_model)
opt = Adam(lr=0.001, beta_1=0.5)
classifier_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print("Classifier Model:")
classifier_model.summary()

#train puzzle here
if not(just_classify):
    history = puzzle_model.fit(x_train_shuffle/255, np.asarray(y_train_test), epochs=epochs, shuffle=True)

    #Predictions
    pred = puzzle_model.predict(x_test_shuffle/255)

    correct = 0
    fail = 0
    for i in range(len(pred)):
        if compare(pred[i], y_test_test[i]):
            correct += 1
        else: fail += 1


    print("Correct:",correct)
    print("Fail:",fail)
    print("Accuracy:",correct/(fail+correct))


    plt.plot(history.history['accuracy'])
    plt.show()
#train classifier here

print(y_train_classify)


history = classifier_model.fit(x_train/255, y_train_classify, epochs=epochs, shuffle=True)

pred = classifier_model.predict(x_test/255)

correct = 0
fail = 0
for i in range(len(pred)):
    if pred[i] == y_test_classify[i]:
        correct += 1
    else:
        fail += 1

print("Correct:", correct)
print("Fail:", fail)
print("Accuracy:", correct / (fail + correct))

plt.plot(history.history['accuracy'])
plt.show()
# #train classifier here
