# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:10:28 2019

@author: cbx
"""

##用vae标签做回归问题
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
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
import cv2
from keras.preprocessing.image import img_to_array
##vae生成图片
imgs = glob.glob('F:\\inverse\\matlab\\image\\1\\*.jpg')
np.random.shuffle(imgs)
height,width = imageio.imread(imgs[0]).shape[:2]
img_dim = 128
z_dim = 512
test_imgs = glob.glob('F:\\inverse\\matlab\\image\\1\\*.jpg')

def imread(f):
    x = imageio.imread(f)
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
        path = 'samples/test_%s.jpg' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_encoder.weights')


evaluator = Evaluate()

vae.fit_generator(data_generator(),
                  epochs=100,
                  steps_per_epoch=300,
                  callbacks=[evaluator])

##提取图片数据以及条件
image_paths = test_imgs
S = []
W = []

for each in image_paths:
     
    image1 = imageio.imread(each)
    image1 = cv2.resize(image1,(128,128))
    image1 = img_to_array(image1)/255.0
    image1 = np.expand_dims(image1,axis=0)
    result = encoder.predict(image1)
    S.append(result)
    label = int(os.path.split(each)[1].split(".")[0])
    W.append(label)

result = np.array(S)
result1 = result.reshape(2000,512)
W = np.array(W)

##对数据用自编码器进行降维  
m_in = Input(shape=(512,))
m = m_in
s= Dense(256,activation='relu')(m)
s= Dense(128,activation='relu')(s)
s= Dense(64,activation='relu')(s)
s= Dense(16,activation='relu')(s)
s= Dense(4,activation='relu')(s)
s= Dense(4,activation='relu')(s)
s1= Dense(2,activation='relu')(s)
encoder1 = Model(m_in ,s1)
encoder1.summary()

u_in = s1
u = Dense(2,activation='relu')(u_in)
u= Dense(4,activation='relu')(u)
u= Dense(4,activation='relu')(u)
u= Dense(16,activation='relu')(u)
u= Dense(64,activation='relu')(u)
u= Dense(128,activation='relu')(u)
u= Dense(256,activation='relu')(u)
u1 = Dense(512,activation=None)(u)

    
U = Model(m_in , u1)
U.compile(optimizer='adam', loss='mse')  
U.fit(result1,result1,shuffle=True,epochs=100, batch_size=100)
U.summary()

##回归
x_train_regression =  encoder1.predict(result1)
ss = StandardScaler()
#标准化数据，均值为0 方差为1
x_train_regression  = ss.fit_transform(x_train_regression)
x_train_regression =  encoder1.predict(result1)               
f_train = W

ET = ExtraTreesRegressor()
ET.fit(x_train_regression ,f_train)
f_pred_et = ET.predict(x_train_regression)
print('r2_score_et:',r2_score(f_pred_et,f_train))
print('mean_squared_error:',mean_squared_error(f_pred_et,f_train))

svr = SVR(kernel = 'poly')
svr.fit(x_train_regression ,f_train)
f_pred_svr = svr.predict(x_train_regression)
print('r2_score_svr:',r2_score(f_pred_svr,f_train))
print('mean_squared_error:',mean_squared_error(f_pred_svr,f_train))
