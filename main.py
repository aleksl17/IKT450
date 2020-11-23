# This is the main file
from input import img_input

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from convolution import define_conv2d
from classifier import define_classifier
from puzzle_solver import define_puzzler

#do data stuff here
x_train, x_train_shuffle, y_train_classify, y_train_shuffle, x_test, x_test_shuffle, y_test_classify, y_test_shuffle = img_input(5000)


#get sub networks
conv_model = define_conv2d()
clas_model = define_classifier()
puz_model = define_puzzler()

#define networks
#puzzle model
puzzle_model = Sequential()
puzzle_model.add(conv_model)
puzzle_model.add(puz_model)
puzzle_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
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
puzzle_model.fit(x_train_shuffle/255, y_train_shuffle/255, epochs=1000, shuffle=True)

y = puzzle_model.predict(x_test_shuffle/255)
pred = y[0]
pred_eval = []
for i in range(9):
    n = 0.0
    index = 0
    for j in range(9):
        if pred[9*i+j] > n:
            n = pred[9*i+j]
            index = j
    pred_eval.append(index)

y_eval = []
for i in range(9):
    n = 0.0
    index = 0
    for j in range(9):
        if y_test_shuffle[0][9*i+j] > n:
            n = y_test_shuffle[0][9*i+j]
            index = j
    y_eval.append(index)


print(pred_eval)
print(y_eval)

#train classifier here


