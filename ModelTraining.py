import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

train = scipy.io.loadmat('train_32x32.mat')
test = scipy.io.loadmat('test_32x32.mat')
train_neg=np.load('train_neg32x32.npy')
test_neg=np.load('test_neg32x32.npy')

train_X=np.concatenate((train['X'],train_neg),axis=3)
test_X=np.concatenate((test['X'],test_neg),axis=3)
n=train_neg.shape[3]
train_y=np.vstack((train['y'],np.zeros((n,1))))
n=test_neg.shape[3]
test_y=np.vstack((test['y'],np.zeros((n,1))))

r=np.arange(train_X.shape[3])
np.random.shuffle(r)
train_X=train_X[:,:,:,r]
train_y=train_y[r,:]

train_X=np.moveaxis(train_X,-1,0)
test_X=np.moveaxis(test_X,-1,0)

train_y=np.eye(11)[train_y.flatten().astype('int')]
test_y=np.eye(11)[test_y.flatten().astype('int')]

model=VGG16(include_top=False,input_shape=(32,32,3))
flat=Flatten()(model.layers[-1].output)
dense1=Dense(4096, activation='relu')(flat)
dense2=Dense(4096, activation='relu')(dense1)
output=Dense(11, activation='softmax')(dense2)
model=tf.keras.Model(inputs=model.inputs, outputs=output)

ES=callbacks.EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True)
optimizer = optimizers.Adam(learning_rate=1e-4,amsgrad=True)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(train_X,train_y,epochs=1,batch_size=64,shuffle=True, validation_split=0.2,callbacks=[ES])

model.save("model5.h5")