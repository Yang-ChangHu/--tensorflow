import tensorflow as tf
import numpy as np
from sklearn import datasets

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

X_train=datasets.load_iris().data
y_train=datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(X_train)

np.random.seed(116)
np.random.shuffle(y_train)


class IrisModel(Model):
    def __init__(self):
        super(IrisModel,self).__init__()
        self.d1=Dense(3,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())
    def call(self,x):
        y=self.d1(x)
        return y
model=IrisModel()




# model=tf.keras.models.Sequential([
#     tf.keras.layers.Dense(3,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())
# ])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1)
              ,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
              ,metrics=['sparse_categorical_accuracy'])
model.fit(X_train,y_train,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)
model.summary()