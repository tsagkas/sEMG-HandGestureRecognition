import tensorflow as tf

import numpy as np
import math
import os

from PIL import Image
import time

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input, Dropout
from tensorflow.python.keras.layers import Reshape, AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Activation

from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adadelta

from keras.utils import np_utils

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import BatchNormalization
from keras.models import load_model

from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import itertools
import matplotlib.pyplot as plt
 
import scipy.io as sio

NAME = "Train_Myo_Tsagkas_v2_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('MyoConfusionMatrix')

# Loading Images
path_train  = '/train_set'
path_test   = '/test_set'
path_val    = '/val_set'

train_list  = os.listdir(path_train)
test_list   = os.listdir(path_test)
val_list    = os.listdir(path_val)

train_list.sort()
test_list.sort()
val_list.sort()

num_training_samples    = size(train_list)
num_testing_samples     = size(test_list)
num_validation_samples  = size(val_list)

matrix_train    = array([array(sio.loadmat(path_train + '/' + img)['image']).flatten() for img in train_list], 'f')
matrix_test     = array([array(sio.loadmat(path_test  + '/' + img)['image']).flatten() for img in test_list],  'f')
matrix_val      = array([array(sio.loadmat(path_val   + '/' + img)['image']).flatten() for img in val_list],   'f')

# Labeling
label_train = np.ones((num_training_samples,), dtype = int)

label_train[0      : 10848] = 0     #exercise 1  - E1 - index flexion 
label_train[10848  : 21696] = 1     #exercise 2  - E1 - ring flexion
label_train[21696  : 32544] = 2     #exercise 3  - E1 - thumb flexion

label_train[32544  : 43392] = 3     #exercise 4  - E2 - thumb up
label_train[43392  : 54240] = 4     #exercise 5  - E2 - abduction of all fingers
label_train[54240  : 65088] = 5     #exercise 6  - E2 - fingers flexed together in fist
label_train[65088  : 75936] = 6     #exercise 7  - E2 - fixed hook grasp
label_train[75936  : 86784] = 7     #exercise 8  - E2 - ring grasp

label_train[86784  :  97632] = 8    #exercise 9  - E3 - medium wrap
label_train[97632  : 108480] = 9    #exercise 10 - E3 - ring grasp
label_train[108480 : 119328] = 10   #exercise 11 - E3 - prismatic four finger grasp
label_train[119328 : 130176] = 11   #exercise 12 - E3 - writing tripod grasp

label_test = np.ones((num_testing_samples,), dtype = int)

label_test[0      :   3616] = 0   #exercise 1
label_test[3616   :   7232] = 1   #exercise 2 
label_test[7232   :  10848] = 2   #exercise 3

label_test[10848  :  14464] = 3   #exercise 4
label_test[14464  :  18080] = 4   #exercise 5
label_test[18080  :  21696] = 5   #exercise 6 
label_test[21696  :  25312] = 6   #exercise 7 
label_test[25312  :  28928] = 7   #exercise 8 

label_test[28928  :  32544] = 8   #exercise 9 
label_test[32544  :  36160] = 9   #exercise 10
label_test[36160  :  39776] = 10  #exercise 11 
label_test[39776  :  43392] = 11  #exercise 12 

label_val = np.ones((num_validation_samples,), dtype = int)

label_val[0      :   3616] = 0   #exercise 1
label_val[3616   :   7232] = 1   #exercise 2 
label_val[7232   :  10848] = 2   #exercise 3

label_val[10848  :  14464] = 3   #exercise 4
label_val[14464  :  18080] = 4   #exercise 5
label_val[18080  :  21696] = 5   #exercise 6 
label_val[21696  :  25312] = 6   #exercise 7 
label_val[25312  :  28928] = 7   #exercise 8 

label_val[28928  :  32544] = 8   #exercise 9 
label_val[32544  :  36160] = 9   #exercise 10
label_val[36160  :  39776] = 10  #exercise 11 
label_val[39776  :  43392] = 11  #exercise 12 

# Training set.
X_train, y_train = shuffle(matrix_train, label_train, random_state = 3)
# .. to images again! (For convolution)
img_rows = 8
img_cols = 15

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

X_train = np.transpose(X_train, (0,2,3,1))
# .. categorical labeling
num_classes = 12

Y_train= np_utils.to_categorical(y_train, num_classes)

# Test set.
X_test = matrix_test.reshape(matrix_test.shape[0], 1, img_rows, img_cols)

X_test = np.transpose(X_test, (0,2,3,1))
# .. categorical labeling
num_classes = 12

Y_test= np_utils.to_categorical(label_test, num_classes)
# Validation set.

X_val = matrix_val.reshape(matrix_val.shape[0], 1, img_rows, img_cols)

X_val = np.transpose(X_val, (0,2,3,1))
# .. categorical labeling
num_classes = 12

Y_val= np_utils.to_categorical(label_val, num_classes)

# Model             
model = Sequential()

# Stage 1
model.add(Conv2D(kernel_size=(3,4), strides=1, filters=32, padding='same',data_format = 'channels_last', name='layer_conv1', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))

# Stage 2
model.add(Conv2D(kernel_size=(3,3), strides=1, filters=32, padding='same', name='layer_conv2'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(pool_size = 2, strides=1))

# Stage 3
model.add(Conv2D(kernel_size=(2,1), strides=1, filters=32, padding='same', name='layer_conv4'))
model.add(Activation('relu'))

# Stage 4
model.add(Conv2D(kernel_size=(1,3), strides=(1,2), filters=64, padding='same', name='layer_conv5'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(pool_size = 2, strides=(2,2)))

# Stage 5
model.add(Conv2D(kernel_size=(1,2), strides=1, filters=64, padding='same', name='layer_conv7'))
model.add(Activation('relu'))

# Stage 6
model.add(Conv2D(kernel_size=(2,2), strides=1, filters=128, padding='same', name='layer_conv8'))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Flatten())

# Stage 7
model.add(Dense(units = 512))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Stage 8
model.add(Dense(units = 128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Stage 9
model.add(Dense(units = 64))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(12))
model.add(Activation('softmax'))

model.summary()

# Optimizer
sgd = SGD(lr=0.01, decay=1e-6,  momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit
model.fit(x = X_train, y = Y_train, validation_data=(X_val, Y_val), epochs = 50, batch_size = 1024, verbose = 1, callbacks = [tensorboard])

# Evaluation
result = model.evaluate(x=X_test,y=Y_test)

for name, value in zip(model.metrics_names, result):
    print(name, value)
    
model.save('Myo_Armband_Model_Demo.h5')

# Confusion Matrix
rounded_predictions = model.predict_classes(X_test, batch_size = 1024, verbose = 0)
conf_matrix = confusion_matrix(Y_test, rounded_predictions)

cm_plot_labels = ['index finger flexion', 'ring finger flexion', 'thumb extension', 'thumb up', 'index-middle extension','abduction of all fingers', 'fist', 'pointing index', 'bottle grasp', 'ring grasp', 'prismatic four finger grasp', 'writing tripod grasp']

plot_confusion_matrix(conf_matrix, cm_plot_labels, title = 'Confusion Matrix')