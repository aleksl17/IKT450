# This is the main file
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from convolution import define_conv2d
from classifier import define_classifier
from puzzle_solver import define_puzzler

#do data stuff here


#get sub networks
conv_model = define_conv2d()
clas_model = define_classifier()
puz_model = define_puzzler()

#define networks
#puzzle model
puzzle_model = Sequential()
puzzle_model.add(conv_model)
puzzle_model.add(puz_model)
opt = Adam(lr=0.0002, beta_1=0.5)
puzzle_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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

#train classifier here


