# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:50:16 2019

@author: cbx
"""

import numpy as np
from scipy import misc
import glob,keras
import imageio
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers import Conv2D,Conv2DTranspose, GlobalAveragePooling2D
from keras.layers import Input,Dense,Reshape,Lambda,Subtract
from keras.preprocessing import image
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization,Activation
imgs = glob.glob('C:\\Users\\cbx\\Desktop\\新建文件夹\\image\\samples1\\*.jpg')
np.random.shuffle(imgs)

height,width = imageio.imread(imgs[0]).shape[:2]
#center_height = int((height - width) / 2)
img_dim = 128
z_dim = 512


def imread(f):
    x = imageio.imread(f)
    #x = x[center_height:center_height+width, :]
    x = image.img_to_array(x)
    x = misc.imresize(x, (img_dim, img_dim))#sp.misc.imresize(img, [img_size_1, img_size_2])请注意img一定需要是numpy数组哦。
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32) / 255 * 2 - 1


def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.concatenate(X)
                yield X,None
                X = []
                #X = X / 127.5 - 1.
    


x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in
x = Conv2D(z_dim//16, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim//8, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim//4, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim//2, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = GlobalAveragePooling2D()(x)

encoder = Model(x_in, x)
encoder.summary()
map_size = K.int_shape(encoder.layers[-2].output)[1:-1]#以整数Tuple或None的形式返回张量shape

# 解码层，也就是生成器部分
z_in = Input(shape=K.int_shape(x)[1:])
z = z_in
z = Dense(np.prod(map_size)*z_dim)(z)
z = Reshape(map_size + (z_dim,))(z)
z = Conv2DTranspose(z_dim//2, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim//4, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim//8, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim//16, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(3, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = Activation('tanh')(z)

decoder = Model(z_in, z)
decoder.summary()

class ScaleShift(keras.layers.Layer):
    
    #*args表示任何多个无名参数，它是一个tuple；**kwargs表示关键字参数，它是一个dict。并且同时使用*args和**kwargs时，
    #必须*args参数列要在**kwargs前，
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    # 重参数技巧
    def call(self, inputs):
        z, shift, log_scale = inputs
        z = K.exp(log_scale) * z + shift
        logdet = -K.sum(K.mean(log_scale, 0))
        self.add_loss(logdet)
        return z
# 算p(Z|X)的均值和方差
z_shift = Dense(z_dim)(x)

z_log_scale = Dense(z_dim)(x)
# 重参数层，相当于给输入加入噪声
u = Lambda(lambda z: K.random_normal(shape=K.shape(z)))(z_shift)
z = ScaleShift()([u, z_shift, z_log_scale])

x_recon = decoder(z)
x_out = Subtract()([x_in, x_recon])
# xent_loss是重构loss，z_loss是KL loss
recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_out)[1:])
z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
vae_loss = recon_loss + z_loss

vae = Model(x_in, x_out)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(1e-4))


def sample(path):
   
    x_recon = decoder.predict(np.random.randn(1, *K.int_shape(x)[1:]))
    digit = x_recon[0]
            
    imageio.imwrite(path, digit)


class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('samples'):
            os.mkdir('samples')
    def on_epoch_end(self, epoch, logs=None):
        path = 'samples/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_encoder.weights')


evaluator = Evaluate()

vae.fit_generator(data_generator(),
                  epochs=1000,
                  steps_per_epoch=1000,
                  callbacks=[evaluator])