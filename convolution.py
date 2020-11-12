# This is the 2d conv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam

#parameters:
input_shape = (30, 30, 3)

def define_conv2d():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.1))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(2,2), padding='same', activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(36, activation='sigmoid'))

    #Present the model
    print("Convolutional Network 2D:")
    model.summary()

    return model


m = define_conv2d()
