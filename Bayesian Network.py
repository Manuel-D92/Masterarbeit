from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import tensorflow_probability as tfp
tfd = tfp.distributions
import six
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable

#loss_funktions=['MSE','KLD','MAE','MSLE','binary_crossentropy','categorical_crossentropy','categorical_hinge','cosine_similarity','hinge','kullback_leibler_divergence','logcosh','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error','mean_squared_logarithmic_error','poisson','squared_hinge','MAPE']#,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber'

## Split Options
split_test_size = 0.2
split_random_state=42
split_shuffle = False

# Time Steps (create_dataset)
creat_dataset_time_steps = 16

## Loss Options
#loss_funktions=  ['mean_squared_error'] #Probabilistic lossfunktion (Bayesian)lambda y, p_y: -p_y.log_prob(y)
loss_funktions = lambda y, p_y: -p_y.log_prob(y)

## Scaler Options
scaler = preprocessing.Normalizer()


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

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

#def create_dataset(X, y, time_steps=1):
#    Xs, ys = [], []
#    overlap = int(time_steps/2)
#    for i in range(int(len(X)/(time_steps/2)) - overlap):
#        v = X.iloc[(i*overlap):((i*overlap)+ time_steps)].values
#        Xs.append(v)
#        ys.append(y.iloc[i*overlap + time_steps])
#    return np.array(Xs), np.array(ys)

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

def shape_the_Signal(X_Signal_number_train,X_train_time_series_normalize,Y_train_time_series_normalize,i):
    count = 0
    X_train_time_series = []
    Y_train_time_series = []
    for j in range(int(X_Signal_number_train[0]),int(X_Signal_number_train[-1] + 1)):
        of = 0 + count
        count = count + np.count_nonzero(X_Signal_number_train[:] == j)
        to = count
        if(creat_dataset_time_steps*(2**i)<np.count_nonzero(X_Signal_number_train[:] == j)):
            X_train_time_series_tm, Y_train_time_series_tm = create_dataset(X_train_time_series_normalize[of:to][:],
                                                                            Y_train_time_series_normalize[of:to][:],
                                                                            time_steps=creat_dataset_time_steps * (2 ** i))
            if(j==int(X_Signal_number_train[0])):
                X_train_time_series = X_train_time_series_tm
                Y_train_time_series = Y_train_time_series_tm
            else:
                X_train_time_series= np.vstack((X_train_time_series,X_train_time_series_tm))
                Y_train_time_series = np.vstack((Y_train_time_series, Y_train_time_series_tm))
    return X_train_time_series,Y_train_time_series

class MetricWrapper(tf.keras.metrics.Mean):
    def __init__(self, fn, name="my_metric", dtype=None, **kwargs):
        super(MetricWrapper, self).__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super(MetricWrapper, self).update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
        base_config = super(MetricWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1_Test"
path_test_sequenz = r"C:\Users\dauserml\Documents\2020_09_25\Testsequenz_1"
time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(path+"\\all_Files_Z_400_mm_Mit_seperater_Signal_trennung.csv",delimiter=';')
time = time_xyz_antennen_Signal_Komplex_all_Files[:,1]
X_Signal_number=time_xyz_antennen_Signal_Komplex_all_Files[:,0]
X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,9:41]#[:,9:]#
Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,2:5]#[:,2:5]#

##Look up table Daten
#ver = pd.read_csv(r'C:\Users\dauserml\Desktop\goalref_simulation-master\gui_multipleTabs\tables\holzRegal_119KHz_1.csv', skiprows=17,sep=';')
#ver = np.array(ver)
#X_1 = ver[:,6:25]
#Y_1 = ver[:,:3]

time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz = np.loadtxt(path_test_sequenz+"\\all_Files.csv",delimiter=';')
time_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,0]

X_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,7:40]
Y_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,1:4]

print("X_beginn_shape =",X_1.shape)
print("Y_beginn_shape =",Y_1.shape)

#first_NN(X_1,Y_1)
normalizer_test_sequenz_x = scaler
X_1_test_sequenz_normlize=normalizer_test_sequenz_x.fit_transform(X_1_test_sequenz)
normalizer_test_sequenz_y = scaler
Y_1_test_sequenz_normlize=normalizer_test_sequenz_y.fit_transform(Y_1_test_sequenz)

#X_1 = pd.DataFrame(X_1,columns=["frame1_real","frame1_imag","frame2_real","frame2_imag","frame3_real","frame3_imag","frame4_real","frame4_imag","frame5_real","frame5_imag","frame6_real","frame6_imag","frame7_real","frame7_imag","frame8_real","frame8_imag","main1_real","main1_imag","main2_real","main2_imag","main3_real","main3_imag","main4_real","main4_imag","main5_real","main5_imag","main6_real","main6_imag","main7_real","main7_imag","main8_real","main8_imag",])
#Y_1 = pd.DataFrame(Y_1,columns=['X_postion'])#,'Y_position','Z_position'])

X_train, X_test, Y_train, Y_test = train_test_split(X_1, Y_1, test_size=split_test_size,random_state=split_random_state,shuffle=split_shuffle)
X_Signal_number_train,X_Signal_number_test = train_test_split(X_Signal_number, test_size=split_test_size,random_state=split_random_state,shuffle=split_shuffle)
normalizer_x = scaler
X_train_normalize= normalizer_x.fit_transform(X_train)
X_test_normalize=normalizer_x.transform(X_test)
X_train_time_series_normalize = pd.DataFrame(X_train_normalize)
X_test_time_series_normalize = pd.DataFrame(X_test_normalize)

normalizer_y = scaler
Y_train_normalize= normalizer_y.fit_transform(Y_train)
Y_test_normalize=normalizer_y.transform(Y_test)
Y_train_time_series_normalize=pd.DataFrame(Y_train_normalize)
Y_test_time_series_normalize=pd.DataFrame(Y_test_normalize)

for i in range(1,2):
    print("Time Steps: "+str(creat_dataset_time_steps*(2**(i-1))))

    X_train_time_series,Y_train_time_series= shape_the_Signal(X_Signal_number_train=X_Signal_number_train,
                                                              X_train_time_series_normalize=X_train_time_series_normalize,
                                                              Y_train_time_series_normalize=Y_train_time_series_normalize,i=(i-1))
    X_test_time_series, Y_test_time_series = shape_the_Signal(X_Signal_number_train=X_Signal_number_test,
                                                              X_train_time_series_normalize=X_test_time_series_normalize,
                                                              Y_train_time_series_normalize=Y_test_time_series_normalize,i=(i-1))

    #X_train_time_series, Y_train_time_series = create_dataset(X_train_time_series_normalize,Y_train_time_series_normalize,time_steps=creat_dataset_time_steps*(2**(i-1)))
    #X_test_time_series, Y_test_time_series = create_dataset(X_test_time_series_normalize,Y_test_time_series_normalize,time_steps=creat_dataset_time_steps*(2**(i-1)))

    print("X_Time_Series_generator_shape =",X_train_time_series.shape)
    print("Y_Time_Series_generator_shape =",Y_train_time_series.shape)


    model = tf.keras.Sequential([#
        tfp.layers.Convolution1DFlipout(32,3,activation='relu',padding='same', input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),
        tfp.layers.Convolution1DFlipout(64,3,activation='relu',padding='same'),
        tf.keras.layers.Flatten(),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
        #tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(Y_train_time_series.shape[1]),activation=None),
        #tfp.layers.MultivariateNormalTriL( Y_train_time_series.shape[1],activity_regularizer = tfp.layers.KLDivergenceRegularizer( tfd.Independent(tfd.Normal(loc= tf.zeros(3, dtype=tf.float32),scale=1.0),reinterpreted_batch_ndims=1),weight=1/128))
        #tf.keras.layers.Conv1D(32,3, activation='relu',padding="same", input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),

        #tf.keras.layers.Dropout(0.3),
        #tf.keras.layers.Dense(64,activation='sigmoid'),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.LSTM(256,activation='tanh',return_sequences=True),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.LSTM(64,activation='tanh',recurrent_activation='sigmoid',input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),#,return_sequences=True),
        #tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.Conv1D(32,3, activation='relu',padding="same"),
        #tf.keras.layers.AveragePooling1D(2),
        #tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
        #tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
        #tf.keras.layers.AveragePooling1D(2),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.AveragePooling1D(2),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.AveragePooling1D(2),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
        #tf.keras.layers.AveragePooling1D(2),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Conv1D(32,9, activation='relu'),#padding="same"),
        #tf.keras.layers.AveragePooling1D(),
        #tf.keras.layers.Dropout(0.3),
        #tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(units=Y_train_time_series.shape[1])#,activation='linear')
    ])
    model.summary()
    time_stamp_from_NN =get_actuall_time_stamp()
    doku = []
    for loss in range(0,1):#len(loss_funktions)):

        #print('\n\n' + loss_funktions[loss])
        model.compile(loss=MetricWrapper(loss_funktions),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['MSE'])
        #,decay=1e-6,nesterov=True,momentum=0.9
        history = model.fit(X_train_time_series, Y_train_time_series, epochs=20,shuffle=True, verbose=2,batch_size=128,validation_data=(X_test_time_series, Y_test_time_series)) #x=generator

        path_NN = (path + '\\Neuronale_Netze_Time_Series_Generator_nur_mit_X')
        if (False == os.path.isdir(path_NN)):
            os.mkdir(path_NN)
        path_NN_tm = (path_NN+'\\NN_from_'+time_stamp_from_NN)
        if(False==os.path.isdir(path_NN_tm)):
            os.mkdir(path_NN_tm)
        #path_NN_tm_loss = path_NN_tm+'\\'+loss_funktions[loss]
        #os.mkdir(path_NN_tm_loss)
        #model.save(path_NN_tm_loss)

        #print(model.evaluate(X_test_time_series,Y_test_time_series))
        Y_pred_time_series =model.predict(X_test_time_series)#[:4000]
        #Y_pred_time_series = normalizer_y.inverse_transform(Y_pred_time_series)
        #Y_test_time_series = normalizer_y.inverse_transform(Y_test_time_series)#[:4000]
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.title('X Position')
        plt.plot(Y_pred_time_series[:, 0])
        plt.plot(Y_test_time_series[:, 0])
        plt.legend(["X_pred","X_true"])
        plt.subplot(2,1,2)
        plt.plot(np.sqrt((Y_pred_time_series[:, 0]-Y_test_time_series[:, 0])**2))
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.title('Y Position')
        plt.plot(Y_pred_time_series[:, 1])
        plt.plot(Y_test_time_series[:, 1])
        plt.legend(["Y_pred", "Y_true"])
        plt.subplot(2, 1, 2)
        plt.plot(np.sqrt((Y_pred_time_series[:, 1] - Y_test_time_series[:, 1])**2))
        plt.figure(3)
        plt.subplot(2, 1, 1)
        plt.title('Z Position')
        plt.plot(Y_pred_time_series[:, 2])
        plt.plot(Y_test_time_series[:, 2])
        plt.legend(["Z_pred", "Z_true"])
        plt.subplot(2, 1, 2)
        plt.plot(np.sqrt((Y_pred_time_series[:, 2] - Y_test_time_series[:, 2])**2))

        Y_pred_train_time_series = model.predict(X_train_time_series)  # [:4000]
        #Y_pred_train_time_series = normalizer_y.inverse_transform(Y_pred_train_time_series)
        #Y_train_time_series = normalizer_y.inverse_transform(Y_train_time_series)  # [:4000]
        plt.figure(4)
        plt.subplot(2, 1, 1)
        plt.title('X Position train')
        plt.plot(Y_pred_train_time_series[:, 0])
        plt.plot(Y_train_time_series[:, 0])
        plt.legend(["X_pred", "X_true"])
        plt.subplot(2, 1, 2)
        plt.plot(np.sqrt((Y_pred_train_time_series[:, 0] - Y_train_time_series[:, 0])**2))
        plt.figure(5)
        plt.subplot(2, 1, 1)
        plt.title('Y Position train')
        plt.plot(Y_pred_train_time_series[:, 1])
        plt.plot(Y_train_time_series[:, 1])
        plt.legend(["Y_pred", "Y_true"])
        plt.subplot(2, 1, 2)
        plt.plot(np.sqrt((Y_pred_train_time_series[:, 1] - Y_train_time_series[:, 1])**2))
        plt.figure(6)
        plt.subplot(2, 1, 1)
        plt.title('Z Position train')
        plt.plot(Y_pred_train_time_series[:, 2])
        plt.plot(Y_train_time_series[:, 2])
        plt.legend(["Z_pred", "Z_true"])
        plt.subplot(2, 1, 2)
        plt.plot(np.sqrt((Y_pred_train_time_series[:, 2] - Y_train_time_series[:, 2])**2))


        #result_train =
        print('\n------------------------------------------------------------------------------------------------\n')
        #print('\n\n' + loss_funktions[loss])
        print('Loss and MSE Training:',history.history['loss'][-1],history.history['MSE'][-1])
        print('Loss and MSE val_Training:',history.history['val_loss'][-1],history.history['MSE'][-1])
        result = model.evaluate(X_test_time_series,Y_test_time_series,batch_size=128)
        #print("loss (test-set):", str(result))
        #print('loss test_sequenz:', str(model.evaluate(x=np.expand_dims(X_test_time_series, axis=0),y=np.expand_dims(Y_test_time_series, axis=0))))
        print('\n------------------------------------------------------------------------------------------------\n')
        doku.append('------------------------------------------------------------------------------------------------')
        doku.append("\nX_Time_Series_generator_shape ="+str(X_train_time_series.shape)+"\nY_Time_Series_generator_shape ="+str(Y_train_time_series.shape))
        doku.append("\nTime Steps: "+str(creat_dataset_time_steps*(2**(i-1))))
        doku.append("\nScaler: "+str(scaler))
        for i in range (0,len(model._layers)):
            doku.append('\n'+str(model._layers[i].output))
        #doku.append('\n'+loss_funktions[loss] + ': (50 iteration,learning_rate=0.007,Adam, mit relu, batch_size = 128')
        doku.append('\nLoss and MSE Training: '+str(history.history['loss'][-1])+str(history.history['MSE'][-1]))
        doku.append('\nLoss and MSE val_Training: '+str(history.history['val_loss'][-1])+str(history.history['val_MSE'][-1]))
        #doku.append("\nloss (test-set): " +str(result))
        #doku.append('loss test_sequenz:'+str(model.evaluate(x=np.expand_dims(X_test_time_series, axis=0),y=np.expand_dims(Y_test_time_series, axis=0))))
        doku.append('\n------------------------------------------------------------------------------------------------\n')

    file = open(path + '\\Neuronale_Netze_Time_Series_Generator_nur_mit_X'+ '\\NN_from_'+time_stamp_from_NN+ '\\Doku.txt', 'w')
    for i in range(0,len(doku)):
        file.write(str(doku[i]))
    file.close()

plt.show()