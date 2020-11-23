# This is the puzzle_solver file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow import split
from tensorflow import keras

def define_puzzler():
    #define nets
    input_shape = (36)
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Dense(36, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(18, activation='sigmoid'))
    model.add(Dense(9, activation='sigmoid'))

    model.summary()

    return model








