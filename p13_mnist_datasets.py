import tensorflow as tf
import  matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator        #data enforcing
from PIL import Image
import numpy as np
import os


def generateds(path,txt):
    f=open(txt,'r')
    contents=f.readlines()
    f.close()
    x,y_=[],[]
    for content in contents:
        value=content.split()
        img_path=path+value[0]
        img=Image.open(img_path)
        img=np.array(img.convert('L'))
        img=img/255.
        x.append(img)
        y_.append(value[1])
        print('loading:'+content)
    x=np.array(x)
    y_=np.array(y_)
    y_=y_.astype(np.int64)
    return x,y_

checkpoint_save_path='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist1224.ckpt'

train_path='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist_image_label/mnist_train_jpg_60000/'
train_txt='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist_image_label/mnist_train_jpg_60000.txt'
x_train_savepath='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist_image_label/mnist_x_train.npy'
y_train_savepath='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist_image_label/mnist_y_train.npy'


test_path='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist_image_label/mnist_test_jpg_10000/'
test_txt='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist_image_label/mnist_test_jpg_10000.txt'
x_test_savepath='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist_image_label/mnist_x_test.npy'
y_test_savepath='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist_image_label/mnist_y_test.npy'

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print('--------------------load datasets---------------')
    x_train_save=np.load(x_train_savepath)
    y_train=np.load(y_train_savepath)

    x_test_save=np.load(x_test_savepath)
    y_test=np.load(y_test_savepath)

    x_train=np.reshape(x_train_save,(len(x_train_save),28,28))
    x_test=np.reshape(x_test_save,(len(x_test_save),28,28))
else:
    print('--------------generate datasets------------------')
    x_train,y_train=generateds(train_path,train_txt)
    x_test,y_test=generateds(test_path,test_txt)

    print('--------------save datasets---------------------')
    x_train_save=np.reshape(x_train,(len(x_train),-1))
    x_test_save=np.reshape(x_test,(len(x_test),-1))
    np.save(x_train_savepath,x_train_save)
    np.save(y_train_savepath,y_train)
    np.save(x_test_savepath,x_test_save)
    np.save(y_test_savepath,y_test)






cp_callback=keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path
    ,save_weights_only=True
    ,save_best_only=True
    )

model=keras.models.Sequential([keras.layers.Flatten()
    ,keras.layers.Dense(128,activation='relu'
                        #,kernel_regularizer=keras.regularizers.l2()
                        )
    ,keras.layers.Dense(10,activation='softmax')
                               ])

# class MnistModel(Model):
#     def __init__(self):
#         super(MnistModel,self).__init__()
#         self.flatten=Flatten()
#         self.d1=Dense(128,activation='relu')
#         self.d2=Dense(10,activation='softmax')
#
#     def call(self,x):
#         x=self.flatten(x)
#         x=self.d1(x)
#         y=self.d2(x)
#         return y
#
# model=MnistModel()
model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])
if os.path.exists(checkpoint_save_path+'.index'):
    print('-----------loading the model------------')
    model.load_weights(checkpoint_save_path)
history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),shuffle=True,validation_freq=1
                  ,callbacks=[cp_callback])
model.summary()

# np.set_printoptions(threshold=np.inf)
# print(model.trainable_variables)
# file=open('./weights.txt','w')
# for v in model.trainable_variables:
#     file.write(str(v.name)+'\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()
acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc,label='training acc')
plt.plot(val_acc,label='val acc')
plt.title('train and val acc')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='loss')
plt.plot(val_loss,label='val loss')
plt.title('loss')
plt.legend()

plt.show()