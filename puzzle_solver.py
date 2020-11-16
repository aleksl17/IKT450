# This is the puzzle_solver file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Add
from tensorflow import split
from tensorflow import keras

def define_puzzler():
    #define nets
    inputs = keras.Input(36)
    x = Dense(58, activation="relu")(inputs)
    x = Dense(81,activation="sigmoid")(x)

    split_tensor = split(x, num_or_size_splits=9,axis=-1)

    #Sub networks for each puzzle piece
    sub0 = Dense(9, activation="softmax")(split_tensor[0])
    sub1 = Dense(9, activation="softmax")(split_tensor[1])
    sub2 = Dense(9, activation="softmax")(split_tensor[2])
    sub3 = Dense(9, activation="softmax")(split_tensor[3])
    sub4 = Dense(9, activation="softmax")(split_tensor[4])
    sub5 = Dense(9, activation="softmax")(split_tensor[5])
    sub6 = Dense(9, activation="softmax")(split_tensor[6])
    sub7 = Dense(9, activation="softmax")(split_tensor[7])
    sub8 = Dense(9, activation="softmax")(split_tensor[8])

    #Merge outputs
    outputs  = Add()([sub0,sub1,sub2,sub3,sub4,sub5,sub6,sub7,sub8])

    #Create model from the above
    model = keras.Model(inputs, outputs, name="puzzler")
    model.summary()

    return model








