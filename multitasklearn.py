import csv
import os
import pandas as pd
from os import listdir
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras import models
from keras import layers
from keras import optimizers
import random 
from keras.preprocessing.image import ImageDataGenerator
from random import randrange
import cv2
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from keras import initializers
from sys import *
import os 
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)
my_init = initializers.glorot_uniform(seed=42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
K.tensorflow_backend._get_available_gpus()

cv_size=126
testsize=126

W=224
X_train = []
y_train = []
y_train1 = []
Path='C:/Users/rudre/Downloads/Height_DB_face_det'
heights = pd.read_csv('C:/Users/rudre/Downloads/Height_DB_face_det/annotation_height_f_.csv',encoding='latin-1')
print('Read train images')
for index, row in heights.iterrows():
    image_path = os.path.join(Path,'images', str((row['img'])) + '.jpg')
    img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR), (W,W) ).astype(np.float32)
    #img = (cv2.imread(image_path, cv2.IMREAD_COLOR), (W, W) )
    img = preprocess_input(img)
    X_train.append(img)
    y_train.append( [ row['WEIGHT'] ] )
    y_train1.append( [ row['HEIGHT'] ] )
  

y_train = np.array(y_train)
y_train = y_train.astype('float32')

m = y_train.mean()
s = y_train.std()
print ('Train mean, sd:', m, s )
y_train -= m
y_train /= s
print('Train shape:', y_train.shape)
print(y_train.shape[0], 'train samples')

y_train1 = np.array(y_train1)
y_train1 = y_train1.astype('float32')

m = y_train1.mean()
s = y_train1.std()
print ('Train mean, sd:', m, s )
y_train1 -= m
y_train1 /= s
print('Train shape:', y_train1.shape)
print(y_train1.shape[0], 'train samples')

X_train = np.array(X_train)
X_train = X_train.astype('float32')

m = X_train.mean()
s = X_train.std()
print ('Train mean, sd:', m, s )
X_train -= m
X_train /= s
print('Train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

#X_train /= 255  

#X, Y1, Y2 = make_x_y(X_train, y_train, y_train1)
#X_train, X_test, Y_train, Y_test, Y_train1, Y_test1 = utils.prepare_train_test(X_train,y_train,y_train1,W)
input_shape = (W, W, 3)

X_train, X_test, Y_train, Y_test, Y_train1, Y_test1 = train_test_split(X_train, y_train, y_train1, test_size=cv_size, random_state=42)
dd=0.0
#base_model = ResNet50(weights='imagenet', include_top=False)
#freeze all the layers
#for layer in base_model.layers[:]:
 #  layer.trainable = False

#base_model.layers[0].trainable=False
visible = Input(shape=input_shape)
#x = base_model(visible)
x = Conv2D(10, kernel_size=3, activation='elu', padding='same', kernel_initializer=my_init)(visible)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(20, kernel_size=3, activation='elu', padding='same', kernel_initializer=my_init)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(30, kernel_size=3, activation='elu', padding='same', kernel_initializer=my_init)(x)
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(256, activation='relu')(x)
x = Dropout(dd)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(dd)(x)
# start passing that fully connected block output to all the different model heads
y1 = Dense(128, activation='relu')(x)
y1 = Dropout(dd)(y1)
y1 = Dense(64, activation='relu')(y1)
y1 = Dropout(dd)(y1)
    
y2 = Dense(128, activation='relu')(x)
y2 = Dropout(dd)(y2)
y2 = Dense(64, activation='relu')(y2)
y2 = Dropout(dd)(y2)

y1 = Dense(Y_train.shape[1], activation='linear',name= 'weight')(y1)
y2 = Dense(Y_train1.shape[1], activation='linear',name= 'height')(y2)

model = Model(inputs=visible, outputs=[y1,y2])

model.compile(loss=['mse', 'mse'],
              optimizer=Adam(clipnorm = 1.), metrics = {'weight': 'accuracy','height': 'accuracy'}, loss_weights = [1, 1])

print(model.summary())
model.fit(X_train, [Y_train, Y_train1],
          validation_data = (X_test, [Y_test, Y_test1]),
          batch_size = 64,
          epochs = 20,
          verbose = True,
          shuffle = False)

predictions_valid = model.predict(X_test, batch_size=80, verbose=1)
compare = pd.DataFrame(data={'original_weight':Y_test.reshape((cv_size,)),
             'prediction_weight':predictions_valid[0].reshape((cv_size,)),
             'original_height':Y_test1.reshape((cv_size,)),
             'prediction_height':predictions_valid[1].reshape((cv_size,))})
compare.to_csv('compare_features_cross_validate.csv')
pred = model.predict(X_test)[0]
y_pred = [np.argmax(p) for p in pred]
y_true = [np.argmax(p) for p in Y_test]
print (classification_report(y_true, y_pred))

directory = 'C:/Users/rudre/Downloads/Height_DB_face_det/Test'
#csv = open("test.csv", "w")
#columnTitleRow = "name, weight, height\n"
#csv.write(columnTitleRow)

X_test=[]
#image_id=[]

for name in listdir(directory):
        # load an image from file
    filename = directory + '/' + name
    image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
    image = img_to_array(image)
        # reshape data for the model
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
    image = preprocess_input(image)
        # get features
    
    X_test.append(image)
    
    #feature = model.predict(image, verbose=0)

    #print(feature)
        # get image id
    #image_id = name.split('.')[0]

    #image_id.append(image_id)
    #row = str(image_id) + "," + str((feature[0])) + "," + str((feature[1])) + "\n"
    
    #csv.write(row)

X_test = np.array(X_test)
X_test = X_test.astype('float32')

m = X_test.mean()
s = X_test.std()
print ('Train mean, sd:', m, s )
X_test -= m
X_test /= s
print('Train shape:', X_test.shape)
print(X_test.shape[0], 'train samples')

predictions_valid = model.predict(X_test, batch_size=80, verbose=1)
compare = pd.DataFrame(data={'prediction_weight':predictions_valid[0].reshape((testsize,)),
             'prediction_height':predictions_valid[1].reshape((testsize,))})

compare.to_csv('compare_features_test.csv')

