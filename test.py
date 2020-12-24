from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path='/home/ych/DL_std/pku_std/class4/MNIST_FC/mnist1224.ckpt'

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten()
    ,tf.keras.layers.Dense(128,activation='relu')
    ,tf.keras.layers.Dense(10,activation='softmax')
])
model.load_weights(model_save_path)
preNum=int(input('input the number of test pictures:'))
for i in range(preNum):
    image_path=input('the path of test picture:')
    img=Image.open(image_path)
    img=img.resize((28,28),Image.ANTIALIAS)
    img_arr=np.array(img.convert('L'))

    img_arr=255-img_arr

    img_arr=img_arr/255.
    X_predict=img_arr[tf.newaxis,...]
    result=model.predict(X_predict)
    pred=tf.argmax(result,axis=1)
    print('\n')

    tf.print(pred)