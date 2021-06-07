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

#loss_funktions=['KLD','MAE','MAPE','MSE','MSLE','binary_crossentropy','categorical_crossentropy','categorical_hinge','cosine_similarity','hinge','kullback_leibler_divergence','logcosh','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error','mean_squared_logarithmic_error','poisson','squared_hinge']#,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber'
loss_funktions=['mse']

scaler = preprocessing.MinMaxScaler()

def get_actuall_time_stamp():
    dateTimeObj = datetime.now()
    timestamp_from_NN = dateTimeObj.strftime("%d_%b_%Y_%H_%M")
    return timestamp_from_NN

def first_NN(X_1,Y_1):
    X_train, X_test, Y_train, Y_test= train_test_split(X_1, Y_1, test_size=0.2,random_state=42,shuffle=True)
    normalizer = scaler
    normalizer.fit(X_train)
    X_train= normalizer.transform(X_train)
    X_test=normalizer.transform(X_test)
    normalizer.fit(Y_train)
    Y_train= normalizer.transform(Y_train)
    Y_test_normalization=normalizer.transform(Y_test)
    print('\nX_train shape:',X_train.shape)
    print('\nY_train shape:',Y_train.shape)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(120,activation='relu',input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(120,activation='relu',kernel_initializer='random_normal'),
        tf.keras.layers.Dense(120,activation='relu',kernel_initializer='random_normal'),
        tf.keras.layers.Dense(Y_train.shape[1])
    ])
    model.summary()
    model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.007))
    history = model.fit(X_train,Y_train, epochs=2000,verbose=2,validation_split=0.2,batch_size=128)#,validation_data=(X_test, Y_test))
    print('\n------------------------------------------------\nTrain MSE:',model.evaluate(X_train,Y_train,verbose=0))
    print('Train MSE:',model.evaluate(X_test,Y_test_normalization,verbose=0),'\n------------------------------------------------')
    Y_Pred = model.predict(X_test)
    Y_Pred = normalizer.inverse_transform(Y_Pred)
    plt.figure(3)
    plt.subplot(3,1,1)
    plt.plot(Y_Pred[:,0])
    plt.plot(Y_test[:,0])
    plt.ylabel('mm')
    plt.xlabel('n')
    plt.title('X-Position')
    plt.legend((['X Prediction','X Position']))
    plt.subplot(3,1,2)
    plt.plot(Y_Pred[:,1])
    plt.plot(Y_test[:,1])
    plt.ylabel('mm')
    plt.xlabel('n')
    plt.title('Y-Position')
    plt.legend((['Y Prediction','Y Position']))
    plt.subplot(3,1,3)
    plt.plot(Y_Pred[:,2])
    plt.plot(Y_test[:,2])
    plt.ylabel('mm')
    plt.xlabel('n')
    plt.title('Z-Position')
    plt.legend((['Z Prediction','Z Position']))
    plt.show()
    plt.figure(4)
    plt.plot(Y_Pred-Y_test)

def batch_generator(batch_size, sequence_length,num_x_signals,num_y_signals,num_train,x_train_scaled,y_train_scaled):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]

        yield (x_batch, y_batch)

def plot_comparison(x_train_scaled,y_train,x_test_scaled,y_test,y_scaler,start_idx,value_plot, length=100, train=True):
    """
    Plot the predicted and true output-signals.

    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """

    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test

    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    Output_name=['X_postion','Y_position','Z_position']

    # For each output-signal.
    for signal in range(0,len(y_train[0,:])):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]

        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        if train:
            plt.figure(signal+value_plot*3+1,figsize=(15, 5))
            plt.suptitle(Output_name[signal]+' Trainingset')
        else:
            plt.figure(signal+value_plot*3+1, figsize=(15, 5))
            plt.suptitle(Output_name[signal]+' Testset')

        # Plot and compare the two signals.
        plt.subplot(2,1,1)
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        plt.xlabel("n")
        plt.ylabel('mm')
        plt.subplot(2,1,2)
        plt.plot(signal_true-signal_pred)
        plt.xlabel("n")
        plt.ylabel('mm')

        # Plot labels etc.
        #plt.legend()
    #plt.show()

path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1"
path_test_sequenz = r"C:\Users\dauserml\Documents\2020_09_25\Testsequenz_1"
time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(path+"\\all_Files.csv",delimiter=';')
time = time_xyz_antennen_Signal_Komplex_all_Files[:,0]

X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,4:]
Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,1:4]

time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz = np.loadtxt(path_test_sequenz+"\\all_Files.csv",delimiter=';')
time_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,0]

X_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,4:]
Y_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,1:4]

print("X_shape =",X_1.shape)
print("y_shape =",Y_1.shape)

#first_NN(X_1,Y_1)
normalizer_test_sequenz_x = scaler
X_1_test_sequenz_normlize=normalizer_test_sequenz_x.fit_transform(X_1_test_sequenz)
normalizer_test_sequenz_y = scaler
Y_1_test_sequenz_normlize=normalizer_test_sequenz_y.fit_transform(Y_1_test_sequenz)


X_train, X_test, Y_train, Y_test= train_test_split(X_1, Y_1, test_size=0.2,random_state=42,shuffle=False)
normalizer_x = scaler
X_train_normalize= normalizer_x.fit_transform(X_train)
X_test_normalize=normalizer_x.transform(X_test)

normalizer_y = scaler
Y_train_normalize= normalizer_y.fit_transform(Y_train)
Y_test_normalize=normalizer_y.transform(Y_test)

generator = batch_generator(batch_size=(40960),sequence_length=(10),num_x_signals=X_train.shape[1],num_y_signals=Y_train.shape[1],num_train=len(X_train),x_train_scaled=X_train_normalize,y_train_scaled=Y_train_normalize)
X_Batch,Y_Batch=next(generator)
generator_test = batch_generator(batch_size=(40960),sequence_length=(10),num_x_signals=X_test.shape[1],num_y_signals=Y_test.shape[1],num_train=len(X_test),x_train_scaled=X_test_normalize,y_train_scaled=Y_test_normalize)
X_Batch_test,Y_Batch_test=next(generator)

print("X_Batch_shape",X_Batch.shape)
print("Y_Batch_shape",Y_Batch.shape)

#X_batch_data = tf.data.Dataset.from_tensor_slices((X_1,Y_1))
#new = X_batch_data.batch(32)
#X = pd.DataFrame(X_1,columns=["frame1_real","frame1_imag","frame2_real","frame2_imag","frame3_real","frame3_imag","frame4_real","frame4_imag","frame5_real","frame5_imag","frame6_real","frame6_imag","frame7_real","frame7_imag","frame8_real","frame8_imag","main1_real","main1_imag","main2_real","main2_imag","main3_real","main3_imag","main4_real","main4_imag","main5_real","main5_imag","main6_real","main6_imag","main7_real","main7_imag","main8_real","main8_imag",])
#Y = pd.DataFrame(Y_1,columns=['X_postion','Y_position','Z_position'])

model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64,3, activation='relu',padding="same", input_shape=(None,X_Batch.shape[2])),
        #tf.keras.layers.Dropout(0.3),
        #tf.keras.layers.Dense(64,activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.LSTM(256,activation='tanh',return_sequences=True),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.LSTM(256,activation='tanh',recurrent_activation='sigmoid',input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),#,return_sequences=True),
        #tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv1D(64,3, activation='relu',padding="same"),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv1D(64, 3, activation='relu',padding="same"),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.AveragePooling1D(),
        #tf.keras.layers.Dropout(0.3),
        #tf.keras.layers.Dense(120, activation='relu', kernel_initializer='random_normal'),
        tf.keras.layers.Dense(units=Y_Batch.shape[2])#,activation='linear')
    ])
model.summary()
time_stamp_from_NN =get_actuall_time_stamp()
doku = []
for loss in range(0,len(loss_funktions)):

    print('\n\n' + loss_funktions[loss])
    model.compile(loss=loss_funktions[loss],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['MSE'])

    history = model.fit(X_Batch, Y_Batch, epochs=50, verbose=2,batch_size=256,validation_data=(X_Batch_test, Y_Batch_test)) #x=generator


    path_NN = (path + '\\Neuronale_Netze_Batch_Generator')
    if (False == os.path.isdir(path_NN)):
        os.mkdir(path_NN)
    path_NN_tm = (path_NN+'\\NN_from_'+time_stamp_from_NN)
    if(False==os.path.isdir(path_NN_tm)):
        os.mkdir(path_NN_tm)
    path_NN_tm_loss = path_NN_tm+'\\'+loss_funktions[loss]
    #os.mkdir(path_NN_tm_loss)
    model.save(path_NN_tm_loss)


    #result_train =
    print('\n------------------------------------------------------------------------------------------------\n')
    print('\n\n' + loss_funktions[loss])
    print('Loss and accuracy Training:',history.history['loss'][-1],history.history['accuracy'][-1])
    print('Loss and accuracy val_Training:',history.history['val_loss'][-1],history.history['val_accuracy'][-1])
    result = model.evaluate(x=np.expand_dims(X_test_normalize, axis=0),y=np.expand_dims(Y_test_normalize, axis=0))

    print("loss (test-set):", result)
    print('loss test_sequenz:', model.evaluate(x=np.expand_dims(X_1_test_sequenz_normlize, axis=0),y=np.expand_dims(Y_1_test_sequenz_normlize, axis=0)))
    print('\n------------------------------------------------------------------------------------------------\n')


    #doku.append('------------------------------------------------------------------------------------------------')
    doku.append( loss_funktions[loss] + ': (2000 iteration,learning_rate=0.007,Adam,3xConv1D[120] mit relu, batchsize = [128,680]')
    doku.append('\nLoss and accuracy Training: '+str(history.history['loss'][-1])+str(history.history['accuracy'][-1]))
    doku.append('\nLoss and accuracy val_Training: '+str(history.history['val_loss'][-1])+str(history.history['val_accuracy'][-1]))
    doku.append("\nloss (test-set): " +str(result))
    doku.append('loss test_sequenz:'+str(model.evaluate(x=np.expand_dims(X_1_test_sequenz_normlize, axis=0),y=np.expand_dims(Y_1_test_sequenz_normlize, axis=0))))
    doku.append('\n------------------------------------------------------------------------------------------------\n')

file = open(path + '\\Neuronale_Netze_Batch_Generator'+ '\\NN_from_'+time_stamp_from_NN+ '\\Doku.txt', 'w')
for i in range(0,len(doku)):
    file.write(str(doku[i]))
file.close()

plot_comparison(start_idx=0, length=len(X_1), train=True,x_train_scaled=X_train_normalize,y_train=Y_train,x_test_scaled=X_test_normalize,y_test=Y_test,y_scaler=normalizer_y,value_plot=0)
plot_comparison(start_idx=0, length=len(X_1), train=False,x_train_scaled=X_train_normalize,y_train=Y_train,x_test_scaled=X_test_normalize,y_test=Y_test,y_scaler=normalizer_y,value_plot=1)
plot_comparison(start_idx=0, length=len(X_1), train=True,x_train_scaled=X_1_test_sequenz_normlize,y_train=Y_1_test_sequenz,x_test_scaled=X_1_test_sequenz_normlize,y_test=Y_1_test_sequenz,y_scaler=normalizer_test_sequenz_y,value_plot=2)

plt.show()

#torch.save(model, ("C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\model_2"))
