# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:46:03 2019

@author: cbx
"""

import scipy.io as sio
import numpy as np
from keras.layers import Input, Dense,BatchNormalization
from keras.models import Model
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
from keras import optimizers
load_path = 'C:\\Users\\cbx\\Desktop\\IGA\\matlab1.mat'
load_data = sio.loadmat(load_path)
data = load_data['data_bentpipe_bp']
data_d = load_data['data_d']
from decimal import Decimal
#control_points = data[:30]

training_data = []
for i in range(10000):
    training_data.append({'control_points':data[i],'u':data_d[i]})#data_d
np.random.shuffle(training_data)
#round(training_data,6)
#training_data1 = Decimal(training_data)

x_Train = training_data[:9500]
x_Test = training_data[9500:10000]
# 压缩特征维度至n维  
encoding_dim = 16
  
# this is our input placeholder 
x_train = []
x_test = []
f_train = []
f_test = []

for i in range(len(x_Train)):
    temp =x_Train[i]['u']
    temp2 = x_Train[i]['control_points']
    x_train.append(temp)
    f_train.append(temp2)
x_train = np.array(x_train)
#x_train1 = x_train*1000000000
f_train = np.array(f_train)
f_train = f_train  

for i in range(len(x_Test)):
    temp =x_Test[i]['u']
    temp2 = x_Test[i]['control_points']
    x_test.append(temp)
    f_test.append(temp2)
x_test = np.array(x_test)
#x_test = x_test*1000000000
f_test = np.array(f_test)
f_test  = f_test 


input_data = Input(shape=(858,))  #858
  
# 编码层  
encoded = Dense(256, activation=None)(input_data)
#encoded = BatchNormalization()(encoded)  
encoded = Dense(128, activation ='relu')(encoded)
#encoded = BatchNormalization()(encoded) 
encoded = Dense(64, activation ='relu')(encoded)
#encoded = BatchNormalization()(encoded) 
encoded = Dense(32, activation ='relu')(encoded)
#encoded = BatchNormalization()(encoded)  
encoder_output = Dense(encoding_dim, activation = 'relu')(encoded)  
  
# 解码层  
decoded = Dense(32, activation=  'relu')(encoder_output)
#decoded  = BatchNormalization()(decoded )  
decoded = Dense(64, activation='relu')(decoded)
#decoded  = BatchNormalization()(decoded ) 
decoded = Dense(128, activation='relu')(decoded)
#decoded  = BatchNormalization()(decoded )  
decoded = Dense(256, activation=None)(decoded)
#decoded  = BatchNormalization()(decoded)    
decoder_output = Dense(858, activation=None)(decoded)  
  
# 构建自编码模型  
autoencoder = Model(inputs=input_data, outputs=decoder_output)  
  
# 构建编码模型  
encoder = Model(inputs=input_data, outputs=encoder_output)  

# compile autoencoder  
#sgd  = optimizers.RMSprop(lr=0.08, rho=0.9, epsilon=None, decay=0.0)
sgd  = optimizers.SGD(lr=0.5, momentum=0.1, decay=0.0, nesterov=False)
#sgd =optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
autoencoder.compile(optimizer=sgd, loss='mse')  
 
autoencoder.summary()
encoder.summary()

autoencoder.fit(x_train, x_train, epochs=100, batch_size=128, 
                shuffle=True)
 
encoded_x_test = encoder.predict(x_test)#编码器降维变量
decoded_x_test = autoencoder.predict(x_test)#解码器预测

print("r2_score:",r2_score(decoded_x_test,x_test))

x_train_regression = encoder.predict(x_train)
x_test_regression = encoded_x_test
#np.savetxt('降维训练变量4.txt',x_train_regression)
#np.savetxt('降维测试变量4.txt',x_test_regression)
#建立F与降维变量的回归关系
ss = StandardScaler()#标准化数据，均值为0 方差为1
x_train_regression  = ss.fit_transform(x_train_regression)
x_test_regression = ss.transform(x_test_regression)
ssf = StandardScaler()
f_train = ssf.fit_transform(f_train)
ET = ExtraTreesRegressor()
ET.fit(x_train_regression ,f_train)
f_pred_et = ET.predict(x_test_regression)
f_pred_et = ssf.inverse_transform(f_pred_et)
print('r2_score_et:',r2_score(f_pred_et,f_test))
print('mean_squared_error:',mean_squared_error(f_pred_et,f_test))