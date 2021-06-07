

# example of a cnn for image classification
from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import pandas as pd
import os
#from tensorflow.keras.layers.experimental import preprocessing


#loss_funktions=['KLD','MAE','MAPE','MSE','MSLE','binary_crossentropy','categorical_crossentropy','categorical_hinge','cosine_similarity','hinge','kullback_leibler_divergence','logcosh','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error','mean_squared_logarithmic_error','poisson','squared_hinge']#,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber'
loss_funktions= ['sparse_categorical_crossentropy'] # Probabilistic lossfunktion (Bayesian) lambda y, p_y: -p_y.log_prob(y)   #

scaler = preprocessing.StandardScaler()

def get_actuall_time_stamp():
    dateTimeObj = datetime.now()
    timestamp_from_NN = dateTimeObj.strftime("%d_%b_%Y_%H_%M")
    return timestamp_from_NN

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1_Test"
path_test_sequenz = r"C:\Users\dauserml\Documents\2020_09_25\Testsequenz_1"
time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(path+"\\all_Files.csv",delimiter=';')
time = time_xyz_antennen_Signal_Komplex_all_Files[0:,0]

X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,8:40]#[:,4:]#
Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,40]-1#[:,1:4]# #40 => Klassifikationen

time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz = np.loadtxt(path_test_sequenz+"\\all_Files.csv",delimiter=';')
time_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,0]

print("X_beginn_shape =",X_1.shape)
print("Y_beginn_shape =",Y_1.shape)

X_train, X_test, Y_train, Y_test= train_test_split(X_1, Y_1, test_size=0.2,random_state=42,shuffle=False)
normalizer_x = scaler
X_train_normalize= normalizer_x.fit_transform(X_train)
X_test_normalize=normalizer_x.transform(X_test)
X_train_time_series_normalize = pd.DataFrame(X_train_normalize)
X_test_time_series_normalize = pd.DataFrame(X_test_normalize)

Y_train = pd.DataFrame(Y_train)
Y_test = pd.DataFrame(Y_test)

X_train_time_series, Y_train_time_series = create_dataset(X_train_time_series_normalize,Y_train,time_steps=8)
X_test_time_series, Y_test_time_series = create_dataset(X_test_time_series_normalize,Y_test,time_steps=8)

print("X_Time_Series_generator_shape =",X_train_time_series.shape)
print("Y_Time_Series_generator_shape =",Y_train_time_series.shape)

model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),
    #tf.keras.layers.Dropout(0.3),
    #tf.keras.layers.Dense(128,activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.LSTM(64,activation='tanh',return_sequences=True),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.LSTM(128,activation='tanh',recurrent_activation='sigmoid',input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),#,return_sequences=True),
    #tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv1D(256,3, activation='relu'),
    #tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(120, activation='relu', kernel_initializer='random_normal'),
    tf.keras.layers.Dense(6,activation='Softmax')#,activation='linear')
])
model.summary()
time_stamp_from_NN =get_actuall_time_stamp()
doku = []
for loss in range(0,len(loss_funktions)):

    print('\n\n' + loss_funktions[loss])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),metrics=['accuracy'])

    history = model.fit(X_train_time_series, Y_train_time_series, epochs=2, verbose=2,shuffle=False, validation_split=0.15)#,batch_size=512)  # ,validation_data=(X_test, Y_test)) #x=generator

    path_NN = (path + '\\Neuronale_Netze_Time_Series_Generator_nur_mit_X')
    if (False == os.path.isdir(path_NN)):
        os.mkdir(path_NN)
    path_NN_tm = (path_NN+'\\NN_from_'+time_stamp_from_NN)
    if(False==os.path.isdir(path_NN_tm)):
        os.mkdir(path_NN_tm)
    path_NN_tm_loss = path_NN_tm+'\\'+loss_funktions[loss]
    #os.mkdir(path_NN_tm_loss)
    #model.save(path_NN_tm_loss)

    #print(model.evaluate(X_test_time_series,Y_test_time_series))
    Y_pred_time_series =model.predict(X_test_time_series)#[:4000]

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(Y_pred_time_series[:, 0])
    plt.plot(Y_test_time_series[:, 0])
    plt.legend(['Y Pred','Y True'])

    #result_train =
    #print('\n------------------------------------------------------------------------------------------------\n')
    #print('\n\n' + loss_funktions[loss])
    #print('Loss and accuracy Training:',history.history['loss'][-1],history.history['accuracy'][-1])
    #print('Loss and accuracy val_Training:',history.history['val_loss'][-1],history.history['val_accuracy'][-1])
    #result = model.evaluate(x=np.expand_dims(X_test_time_series, axis=0),y=np.expand_dims(Y_test_time_series, axis=0))

    #print("loss (test-set):", result)
    #print('loss test_sequenz:', model.evaluate(x=np.expand_dims(X_1_test_sequenz_normlize, axis=0),y=np.expand_dims(Y_1_test_sequenz_normlize, axis=0)))
    #print('\n------------------------------------------------------------------------------------------------\n')


    #doku.append('------------------------------------------------------------------------------------------------')
    #doku.append( loss_funktions[loss] + ': (2000 iteration,learning_rate=0.007,Adam,3xConv1D[120] mit relu, batchsize = [128,680]')
    #doku.append('\nLoss and accuracy Training: '+str(history.history['loss'][-1])+str(history.history['accuracy'][-1]))
    #doku.append('\nLoss and accuracy val_Training: '+str(history.history['val_loss'][-1])+str(history.history['val_accuracy'][-1]))
    #doku.append("\nloss (test-set): " +str(result))
    #doku.append('loss test_sequenz:'+str(model.evaluate(x=np.expand_dims(X_1_test_sequenz_normlize, axis=0),y=np.expand_dims(Y_1_test_sequenz_normlize, axis=0))))
    #doku.append('\n------------------------------------------------------------------------------------------------\n')

file = open(path + '\\Neuronale_Netze_Time_Series_Generator_nur_mit_X'+ '\\NN_from_'+time_stamp_from_NN+ '\\Doku.txt', 'w')
for i in range(0,len(doku)):
    file.write(str(doku[i]))
file.close()

plt.show()