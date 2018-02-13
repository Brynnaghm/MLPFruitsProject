import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from data_providers import FruitsDataProvider

seed = 10102016 
rng = np.random.RandomState(seed)
batch_size = 100
train_data = FruitsDataProvider('train', batch_size=batch_size, rng=rng, one_hot=True)
valid_data = FruitsDataProvider('valid', batch_size=batch_size, rng=rng, one_hot=True)
test_data = FruitsDataProvider('test', batch_size=batch_size, rng=rng, one_hot=True)

trainInputs = train_data.inputs
trainTargetsFlat = train_data.targets
trainTargets = np.zeros((trainTargetsFlat.shape[0], 52))
trainTargets[range(trainTargetsFlat.shape[0]), trainTargetsFlat] = 1

validInputs = valid_data.inputs
validTargetsFlat = valid_data.targets
validTargets = np.zeros((validTargetsFlat.shape[0], 52))
validTargets[range(validTargetsFlat.shape[0]), validTargetsFlat] = 1

testInputs = test_data.inputs
testTargetsFlat = test_data.targets
testTargets = np.zeros((testTargetsFlat.shape[0], 52))
testTargets[range(testTargetsFlat.shape[0]), testTargetsFlat] = 1

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(52, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

logger = CSVLogger('training.log', separator=',', append=True)

checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, 
                                save_weights_only=False, mode='auto', period=1)

tensorBoard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, 
                            write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


model.fit(trainInputs, trainTargets, batch_size=batch_size, epochs=40, 
          validation_data=(validInputs, validTargets), callbacks=[logger, checkpointer, tensorBoard])

