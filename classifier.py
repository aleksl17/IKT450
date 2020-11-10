from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# List all the categories for reference
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#batch_size = 36
#img_width = 30
#img_height = 30

# Input from convolution: 36 bits
input_shape = (36, 1)

# Create model
model = Sequential()
model.add(Input(shape=input_shape))

# tune hyperparameter 'units' in dense layer
# to get a higher accuracy, keep the one
# that yields lowest loss value. Same goes for dropout

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(len(class_names), activation='softmax'))

model.summary()
