from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import time as time_1
import pandas as pd
import os
import pickle
import random
from tcn import TCN
#from tcn import TCN   # Testing "tox" pip install tox

#from tensorflow.python.client import device_lib
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_divece_placement=True))

#loss_funktions=['MSE','KLD','MAE','MSLE','binary_crossentropy','categorical_crossentropy','categorical_hinge','cosine_similarity','hinge','kullback_leibler_divergence','logcosh','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error','mean_squared_logarithmic_error','poisson','squared_hinge','MAPE']#,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'

class config_Test_Tensorflow():
    # self.a = outside.
    # Time Steps (create_dataset)
    creat_dataset_time_steps = 16
    time_steps_geo_folge = 1  # geometrische-Folge(zweier Potenzen) for time steps, example: (time_steps=16 andtime_steps_geo_folge=2)->  1
    create_dataset_overlap = 15  # example time_steps=16(0-15); Overlap from Feature to Feature example: 0= no overlap, 15= 0-15,1-16
    create_dataset_offset = 7 # example time_steps=16(0-15)  the middle for y is = offset 7-8 (from 0-15) #0 first value, 15 last value

    #### Trainingsdata into Trainingssequence (Only No Cut Files!!) (with shape)
    sequence_training = True
    sequence_training_cut_x = [-200,1700]
    sequence_training_cut_y = [-200,1400]
    sequence_training_cut_z = [-400,400]

    #### Testdata into Trainingssequence
    sequence_test = True
    sequence_test_cut_x = [0, 1500]
    sequence_test_cut_y = [0, 1200]
    sequence_test_cut_z = [-200, 200]

    # Path
    path = r"C:\Users\Dauser\Desktop\dauserml_Messungen_2020_07_22\2020_11_16"
    path_test_sequenz = r"C:\Users\Dauser\Desktop\dauserml_Messungen_2020_07_22\Messung_1_Test"
    filename_training = "\\2020_11_16_files_No_cut_with_interp_40.csv"  # "\\all_Files_Z_400_mm_Mit_seperater_Signal_trennung.csv"
    filename_test_sequenz = "\\all_Files.csv"
    save_dir = '\\VGG_Time_Series'
    # self.save_dir='\\Neuronale_Netze_Time_Series_Generator_nur_mit_X'

    # Trainings Daten
    all_or_range_selection_Data = True  # all=True, range selection=False
    training_Data_from = 0  # Only "all_or_range_selection_Data = False"
    training_Data_to = 800  # Only "all_or_range_selection_Data = False"
    time = 1  # 1= Timesteps
    X_Signal_number = 0  # 0= Signal delimitation
    Input_from = 9  # Input Signal from 9 to 40 (9=frame1_real,10=frame1_imag,11=frame2_real,....,40=main8_imag)
    Input_to = 41
    # Signal (5=Yaw,6=Roll,7=Pitch,8=Residual) (41= For Classification, witch shelf)
    Output_from = 2  # Ouput Signal from 2 to 4 (2=X,3=Y,4=Z)
    Output_to = 5

    # Training Options
    trainings_epochs = 200
    batch_size = 128
    verbose = 2
    metrics = ['MSE']
    learning_rate = 0.0005
    optimizer_name = "Adam"
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)#,momentum=0.9,nesterov=True,decay=1e-6)

    ## Split Options
    split_test_size = 0.15
    split_KFold_set = 2 # bei 0.2 -> 5 Sets (0->0.0-0.2, 1->0.2-0.4, 2 -> 0.4-0.6 ...... 4 -> 0.8-1.0)
    split_random_state = 42
    split_shuffle = False

    ## Loss Options
    # self.loss_funktions= ['MSE','huber_loss'] #,'MSE','huber_loss','MAP' Probabilistic lossfunktion (Bayesian) lambda y, p_y: -p_y.log_prob(y)   #
    loss_funktions = ['MSE'] # ,'huber_loss',
    # 'logcosh', 'MAPE',
    # 'MSLE']  # ,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'

    ## Scaler Options
    scaler = preprocessing.RobustScaler()

    #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #scaler_end = preprocessing.OneHotEncoder()

    ##Plot option
    pickle = False  # save in pickel can open in pickle_plot.py
    png = True  # save the Picture in png

    # googLeNet para
    kernel_init = tf.keras.initializers.glorot_uniform()
    bias_init = tf.keras.initializers.Constant(value=0.2)

    # Threshold for Antenna threshold_value > Antenna Signal = 0
    threshold_antenna = False
    threshold_value = 0.0002

#loss_funktions=['MSE','KLD','MAE','MSLE','binary_crossentropy','categorical_crossentropy','categorical_hinge','cosine_similarity','hinge','kullback_leibler_divergence','logcosh','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error','mean_squared_logarithmic_error','poisson','squared_hinge','MAPE']#,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'
class Tensorflow:

    def __init__(self,creat_dataset_time_steps,time_steps_geo_folge,create_dataset_overlap,create_dataset_offset,sequence_training,sequence_training_cut_x,sequence_training_cut_y,sequence_training_cut_z,
                 sequence_test,sequence_test_cut_x,sequence_test_cut_y,sequence_test_cut_z,path,path_test_sequenz,filename_training,filename_test_sequenz,save_dir,
                 all_or_range_selection_Data,training_Data_from,training_Data_to,time,X_Signal_number,Input_from,
                 Input_to,Output_from,Output_to,trainings_epochs,batch_size,verbose,metrics,learning_rate,optimizer_name,
                 optimizer,split_test_size,split_KFold_set,split_random_state,split_shuffle,loss_funktions,scaler,threshold_antenna,threshold_value,pickle,png,kernel_init,bias_init):
        #self.a = outside.t
        # Time Steps (create_dataset)
        self.creat_dataset_time_steps = creat_dataset_time_steps
        self.time_steps_geo_folge= time_steps_geo_folge # geometrische-Folge(zweier Potenzen) for time steps, example: (time_steps=16 andtime_steps_geo_folge=2)->  16,32
        self.create_dataset_overlap = create_dataset_overlap
        self.create_dataset_offset = create_dataset_offset

        ####Training in sequence cut
        self.sequence_training = sequence_training
        self.sequence_training_cut_x = sequence_training_cut_x
        self.sequence_training_cut_y = sequence_training_cut_y
        self.sequence_training_cut_z = sequence_training_cut_z

        #### Testdata into Trainingssequence
        self.sequence_test = sequence_test
        self.sequence_test_cut_x = sequence_test_cut_x
        self.sequence_test_cut_y = sequence_test_cut_y
        self.sequence_test_cut_z = sequence_test_cut_z

        #Path
        self.path = path
        self.path_test_sequenz = path_test_sequenz
        self.filename_training = filename_training
        self.filename_test_sequenz = filename_test_sequenz
        self.save_dir = save_dir
        #self.save_dir='\\Neuronale_Netze_Time_Series_Generator_nur_mit_X'

        #Trainings Daten
        self.all_or_range_selection_Data= all_or_range_selection_Data# all=True, range selection=False
        self.training_Data_from = training_Data_from # Only "all_or_range_selection_Data = False"
        self.training_Data_to = training_Data_to # Only "all_or_range_selection_Data = False"
        self.time = time #1= Timesteps from Signal
        self.X_Signal_number =X_Signal_number #0= Signal delimitation
        self.Input_from=Input_from  #Input Signal from 9 to 41 (9=frame1_real,10=frame1_imag,11=frame2_real,....,40=main8_imag)
        self.Input_to=Input_to
        #Signal (5=Yaw,6=Roll,7=Pitch,8=Residual) (41= For Classification, witch shelf)
        self.Output_from=Output_from #Ouput Signal from 2 to 4 (2=X,3=Y,4=Z)
        self.Output_to=Output_to

        #Training Options
        self.trainings_epochs = trainings_epochs
        self.batch_size = batch_size
        self.verbose=verbose
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer = optimizer

        ## Split Options
        self.split_test_size = split_test_size
        self.split_KFold_set = split_KFold_set
        self.split_random_state=split_random_state
        self.split_shuffle = split_shuffle

        ## Loss Options
        #self.loss_funktions= ['MSE','huber_loss'] #,'MSE','huber_loss','MAP' Probabilistic lossfunktion (Bayesian) lambda y, p_y: -p_y.log_prob(y)   #
        self.loss_funktions = loss_funktions#,
                               #'logcosh', 'MAPE',
                               #'MSLE']  # ,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'

        ## Scaler Options
        self.scaler = scaler

        # Threshold Antenna
        self.threshold_antenna = threshold_antenna
        self.threshold_value = threshold_value

        ##Plot option
        self.pickle = pickle #save in pickel can open in pickle_plot.py
        self.png = png #save the Picture in png

        # googLeNet para
        self.kernel_init = kernel_init#tf.keras.initializers.glorot_uniform()
        self.bias_init = bias_init#tf.keras.initializers.Constant(value=0.2)

    @staticmethod
    def get_actuall_time_stamp():
        dateTimeObj = datetime.now()
        timestamp_from_NN = dateTimeObj.strftime("%d_%b_%Y_%H_%M")
        return timestamp_from_NN

    @staticmethod
    def create_dataset(X, y, time_t, time_steps=1, overlap=1, offset=0):
        if (offset < time_steps and overlap < time_steps):
            overlap = time_steps - (overlap)
            Xs, ys, times = [], [], []
            for i in range(int((len(X) - (time_steps)) / overlap)):
                v = X.iloc[(i * overlap):((i * overlap) + time_steps)].values
                Xs.append(v)
                ys.append(y.iloc[(i * overlap) + offset])
                times.append(time_t[(i * overlap) + offset])
            return np.array(Xs), np.array(ys), np.array(times)
        else:
            print("offset or overlap to high!!")

    @staticmethod
    def plot_comparison(model,x_train_scaled,y_train,x_test_scaled,y_test,y_scaler,start_idx,value_plot, length=100, train=True):
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

    def cut_Signal_into_sequencen_train(self, X_train, Y_train, X_Signal_number_train):
        if (self.sequence_training == True):
            X_train_cut_tm = [];
            X_train_cut_NaN = [];
            Y_train_cut_tm = [];
            Y_train_cut_NaN = [];
            X_Signal_number_train_cut_tm = [];
            X_Signal_number_train_cut_NaN = []
            flag = 0
            for c in range(0, len(Y_train)):
                if (self.sequence_training_cut_x[0] < Y_train[c, 0] < self.sequence_training_cut_x[1]
                        and self.sequence_training_cut_y[0] < Y_train[c, 1] < self.sequence_training_cut_y[1]
                        and self.sequence_training_cut_z[0] < Y_train[c, 2] < self.sequence_training_cut_z[1]):
                    X_train_cut_tm.append(X_train[c, :])
                    Y_train_cut_tm.append(Y_train[c, :])
                    X_Signal_number_train_cut_tm.append(X_Signal_number_train[c, :])
                    flag = 1
                elif (flag == 1):
                    X_Signal_number_train = X_Signal_number_train + 1
                    ## Not an NaN Training
                    #X_train_cut_NaN.append(X_train[c, :])
                    #Y_train[c, :] = np.nan
                    #Y_train_cut_NaN.append(Y_train[c, :])
                    #X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c,:])
                    flag = 0
                #else:
                    # X_Signal_number_train = X_Signal_number_train + 1
                    ## Not an NaN Training
                    #X_train_cut_NaN.append(X_train[c, :])
                    #Y_train[c, :] = np.nan
                    #Y_train_cut_NaN.append(Y_train[c, :])
                    #X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c,:])

            #last_X_Number_train_value = X_Signal_number_train[-1]
            #X_Signal_number_train_cut_NaN = np.array(X_Signal_number_train_cut_NaN)
            #X_Signal_number_train_cut_NaN[:,:] = (last_X_Number_train_value+1)
            #X_train_cut_tm = X_train_cut_tm + X_train_cut_NaN
            #Y_train_cut_tm = Y_train_cut_tm + Y_train_cut_NaN
            #X_Signal_number_train = np.vstack((np.array(X_Signal_number_train_cut_tm) , X_Signal_number_train_cut_NaN))

            X_train = np.array(X_train_cut_tm)
            Y_train = np.array(Y_train_cut_tm)
            X_Signal_number_train = np.array(X_Signal_number_train_cut_tm)
        return X_train, Y_train, X_Signal_number_train

    def cut_Signal_into_sequencen_test(self, X_test, Y_test, X_Signal_number_test):
        if (self.sequence_test):
            X_test_cut_tm = [];
            X_test_cut_NaN = [];
            Y_test_cut_tm = [];
            Y_test_cut_NaN = [];
            X_Signal_number_test_cut_tm = [];
            X_Signal_number_test_cut_NaN = []
            flag = 0
            for c in range(0, len(Y_test)):
                if (self.sequence_test_cut_x[0] < Y_test[c, 0] < self.sequence_test_cut_x[1]
                        and self.sequence_test_cut_y[0] < Y_test[c, 1] < self.sequence_test_cut_y[1]
                        and self.sequence_test_cut_z[0] < Y_test[c, 2] < self.sequence_test_cut_z[1]):
                    X_test_cut_tm.append(X_test[c, :])
                    Y_test_cut_tm.append(Y_test[c, :])
                    X_Signal_number_test_cut_tm.append(X_Signal_number_test[c, :])
                    flag = 1
                elif (flag == 1):
                    X_Signal_number_test = X_Signal_number_test + 1
                    ## Not an NaN Training
                    # X_train_cut_NaN.append(X_train[c, :])
                    # Y_train[c, :] = np.nan
                    # Y_train_cut_NaN.append(Y_train[c, :])
                    # X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c,:])
                    flag = 0
                # else:
                # X_Signal_number_train = X_Signal_number_train + 1
                ## Not an NaN Training
                # X_train_cut_NaN.append(X_train[c, :])
                # Y_train[c, :] = np.nan
                # Y_train_cut_NaN.append(Y_train[c, :])
                # X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c,:])

            # last_X_Number_train_value = X_Signal_number_train[-1]
            # X_Signal_number_train_cut_NaN = np.array(X_Signal_number_train_cut_NaN)
            # X_Signal_number_train_cut_NaN[:,:] = (last_X_Number_train_value+1)
            # X_train_cut_tm = X_train_cut_tm + X_train_cut_NaN
            # Y_train_cut_tm = Y_train_cut_tm + Y_train_cut_NaN
            # X_Signal_number_train = np.vstack((np.array(X_Signal_number_train_cut_tm) , X_Signal_number_train_cut_NaN))

            X_test = np.array(X_test_cut_tm)
            Y_test = np.array(Y_test_cut_tm)
            X_Signal_number_test = np.array(X_Signal_number_test_cut_tm)
        return X_test, Y_test, X_Signal_number_test

    def shape_the_Signal(self,X_Signal_number_train,X_train_time_series_normalize,Y_train_time_series_normalize,time_train,i):
        count = 0
        X_train_time_series = []
        Y_train_time_series = []
        for j in range(int(X_Signal_number_train[0]),int(X_Signal_number_train[-1] + 1)): #int(max(X_Signal_number_train))

            of = 0 + count
            count = count + np.count_nonzero(X_Signal_number_train[:] == j)
            to = count
            if(self.creat_dataset_time_steps*(2**i)<np.count_nonzero(X_Signal_number_train[:] == j)):
                X_train_time_series_tm, Y_train_time_series_tm,time_tm = Tensorflow.create_dataset(X_train_time_series_normalize[of:to][:],
                                                                                           Y_train_time_series_normalize[of:to][:],
                                                                                           time_train[of:to],
                                                                                           time_steps=self.creat_dataset_time_steps * (2 ** i),
                                                                                           overlap=self.create_dataset_overlap,offset=self.create_dataset_offset)
                if(j==int(X_Signal_number_train[0])):
                    X_train_time_series = X_train_time_series_tm
                    Y_train_time_series = Y_train_time_series_tm
                    time_train_series = time_tm
                elif(len(X_train_time_series_tm)!=0):
                    try:
                        X_train_time_series= np.vstack((X_train_time_series,X_train_time_series_tm))
                        Y_train_time_series = np.vstack((Y_train_time_series, Y_train_time_series_tm))
                        time_train_series = np.hstack((time_train_series, time_tm))
                    except:
                        print("Error line 339!!!" + str(X_train_time_series.shape) + " and  " + str(X_train_time_series_tm.shape))
        return X_train_time_series, Y_train_time_series, time_train_series

    def create_training_data_KFold(self,X_1,Y_1,X_Signal_number,i,time):

        count_Signal_beginn = 0
        count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == X_Signal_number[0, 0])

        min_max_interval_split_data = self.creat_dataset_time_steps * 4

        if (self.split_test_size * self.split_KFold_set <= 1):
            if (i == 1):
                if (min_max_interval_split_data * 2 < len(X_1[0:count_Signal_end, :])):
                    print("split interval from: " + str(
                        count_Signal_beginn) + "   split interval to:" + str(
                        count_Signal_beginn + int(len(X_1[count_Signal_beginn:count_Signal_end, :]))))
                    random_KFold_test_data_begin = 0 + int((self.split_test_size * self.split_KFold_set) * len(
                        X_1[count_Signal_beginn:count_Signal_end, :]))
                    random_KFold_test_data_end = 0 + int((self.split_test_size * (self.split_KFold_set + 1)) * len(
                        X_1[count_Signal_beginn:count_Signal_end, :]))
                    print(
                        "split from: " + str(random_KFold_test_data_begin) + "   split to: " + str(
                            random_KFold_test_data_end))
                    training_data_1_idx = np.arange(0, random_KFold_test_data_begin)
                    training_data_2_idx = np.arange(random_KFold_test_data_end, len(X_1[0:count_Signal_end, :]))
                    training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
                    test_data_idx = np.arange(random_KFold_test_data_begin, random_KFold_test_data_end)

                    X_train = X_1[training_data_1_2_idx]
                    Y_train = Y_1[training_data_1_2_idx]
                    time_train = time[training_data_1_2_idx]
                    X_Signal_number_dynamisch = X_Signal_number
                    X_Signal_number_neu = X_Signal_number_dynamisch + 1
                    X_Signal_number_train = np.vstack(
                        (X_Signal_number_dynamisch[training_data_1_idx], X_Signal_number_neu[training_data_2_idx]))
                    X_Signal_number_dynamisch = X_Signal_number_dynamisch + 1

                    X_test = X_1[test_data_idx]
                    Y_test = Y_1[test_data_idx]
                    time_test = time[test_data_idx]
                    X_Signal_number_test = X_Signal_number[test_data_idx]

                    count_Signal_beginn = count_Signal_end
                    last_Signal_number = (X_Signal_number[-1, :])
                    for s in range(1, last_Signal_number[0] + 1):
                        if (np.count_nonzero(X_Signal_number == s) != 0):
                            count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == s)
                            print("\nsplit interval from: " + str(
                                count_Signal_beginn) + "   split interval to:" + str(
                                count_Signal_beginn + int(len(X_1[count_Signal_beginn:count_Signal_end, :]))))
                            random_KFold_test_data_begin = count_Signal_beginn + int(
                                (self.split_test_size * self.split_KFold_set) * len(
                                    X_1[count_Signal_beginn:count_Signal_end, :]))
                            random_KFold_test_data_end = count_Signal_beginn + int(
                                (self.split_test_size * (self.split_KFold_set + 1)) * len(
                                    X_1[count_Signal_beginn:count_Signal_end, :]))
                            print("split from: " + str(random_KFold_test_data_begin) + "   split to: " + str(
                                random_KFold_test_data_end))
                            training_data_1_idx = np.arange(count_Signal_beginn, random_KFold_test_data_begin)
                            training_data_2_idx = np.arange(random_KFold_test_data_end,
                                                            count_Signal_beginn + len(
                                                                X_1[count_Signal_beginn:count_Signal_end, :]))
                            training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
                            test_data_idx = np.arange(random_KFold_test_data_begin, random_KFold_test_data_end)

                            X_train_tm = X_1[training_data_1_2_idx]
                            Y_train_tm = Y_1[training_data_1_2_idx]
                            time_train_tm = time[training_data_1_2_idx]
                            X_Signal_number_neu = X_Signal_number_dynamisch + 1
                            X_Signal_number_train_tm = np.vstack(
                                (X_Signal_number_dynamisch[training_data_1_idx],
                                 X_Signal_number_neu[training_data_2_idx]))
                            X_Signal_number_dynamisch = X_Signal_number_dynamisch + 1
                            X_test_tm = X_1[test_data_idx]
                            Y_test_tm = Y_1[test_data_idx]
                            time_test_tm = time[test_data_idx]
                            X_Signal_number_test_tm = X_Signal_number[test_data_idx]
                            X_train = np.vstack((X_train, X_train_tm))
                            X_test = np.vstack((X_test, X_test_tm))
                            Y_train = np.vstack((Y_train, Y_train_tm))
                            Y_test = np.vstack((Y_test, Y_test_tm))
                            X_Signal_number_train = np.vstack((X_Signal_number_train, X_Signal_number_train_tm));
                            X_Signal_number_test = np.vstack((X_Signal_number_test, X_Signal_number_test_tm))
                            count_Signal_beginn = count_Signal_end

                    X_train, Y_train, X_Signal_number_train = Tensorflow.cut_Signal_into_sequencen_train(self, X_train,Y_train,X_Signal_number_train)
                    X_test, Y_test, X_Signal_number_test = Tensorflow.cut_Signal_into_sequencen_test(self, X_test, Y_test,X_Signal_number_test)

                    normalizer_x = self.scaler
                    X_train_normalize = normalizer_x.fit_transform(X_train)
                    X_test_normalize = normalizer_x.transform(X_test)
                    X_train_time_series_normalize = pd.DataFrame(X_train_normalize)
                    X_test_time_series_normalize = pd.DataFrame(X_test_normalize)

                    normalizer_y = self.scaler
                    Y_train_normalize = normalizer_y.fit_transform(Y_train)
                    Y_test_normalize = normalizer_y.transform(Y_test)
                    Y_train_time_series_normalize = pd.DataFrame(Y_train_normalize)
                    Y_test_time_series_normalize = pd.DataFrame(Y_test_normalize)

            X_train_time_series, Y_train_time_series,time_train = Tensorflow.shape_the_Signal(self=self,
                                                                                   X_Signal_number_train=X_Signal_number_train,
                                                                                   X_train_time_series_normalize=X_train_time_series_normalize,
                                                                                   Y_train_time_series_normalize=Y_train_time_series_normalize,
                                                                                   time_train=time_train,
                                                                                   i=(i - 1))
            X_test_time_series, Y_test_time_series,time_test = Tensorflow.shape_the_Signal(self=self,
                                                                                 X_Signal_number_train=X_Signal_number_test,
                                                                                 X_train_time_series_normalize=X_test_time_series_normalize,
                                                                                 Y_train_time_series_normalize=Y_test_time_series_normalize,
                                                                                 time_train=time_test,
                                                                                 i=(i - 1))
        else:
            print("Split KFold to high!!")
        return X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series, normalizer_x, normalizer_y

    def create_training_data_random_KFold(self,X_1,Y_1,X_Signal_number,i,time):

        if(i==1):
            count_Signal_beginn = 0
            count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == X_Signal_number[0,0])

            min_max_interval_split_data = self.creat_dataset_time_steps * 4
            if(min_max_interval_split_data*2<len(X_1[0:count_Signal_end, :])):
                print("split interval from: " + str(
                    count_Signal_beginn + int(min_max_interval_split_data)) + "   split interval to:" + str(
                    count_Signal_beginn + int(len(X_1[count_Signal_beginn:count_Signal_end,
                                                  :]) - min_max_interval_split_data)))  # - int(len(X_1[count_Signal_beginn:count_Signal_end, :])*self.split_test_size)
                random_KFold_test_data_begin = int(random.uniform(int(min_max_interval_split_data), (int(
                    len(X_1[0:count_Signal_end, :]) - min_max_interval_split_data - int(
                        len(X_1[0:count_Signal_end, :]) * self.split_test_size)))))
                random_KFold_test_data_end = random_KFold_test_data_begin + int(
                    self.split_test_size * len(X_1[0:count_Signal_end, :]))
                print("split from: " + str(random_KFold_test_data_begin) + "   split to: " + str(
                    random_KFold_test_data_end))
                training_data_1_idx = np.arange(0, random_KFold_test_data_begin)
                training_data_2_idx = np.arange(random_KFold_test_data_end, len(X_1[0:count_Signal_end, :]))
                training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
                test_data_idx = np.arange(random_KFold_test_data_begin, random_KFold_test_data_end)

                X_train = X_1[training_data_1_2_idx]
                Y_train = Y_1[training_data_1_2_idx]
                time_train = time[training_data_1_2_idx]
                X_Signal_number_dynamisch = X_Signal_number
                X_Signal_number_neu = X_Signal_number_dynamisch + 1
                X_Signal_number_train = np.vstack(
                    (X_Signal_number_dynamisch[training_data_1_idx], X_Signal_number_neu[training_data_2_idx]))
                X_Signal_number_dynamisch = X_Signal_number_dynamisch + 1

                X_test = X_1[test_data_idx]
                Y_test = Y_1[test_data_idx]
                time_test = time[test_data_idx]
                X_Signal_number_test = X_Signal_number[test_data_idx]

                count_Signal_beginn = count_Signal_end
                last_Signal_number = (X_Signal_number[-1, :])
                for s in range(1 + X_Signal_number[0, 0], last_Signal_number[0] + 1):
                    if (np.count_nonzero(X_Signal_number == s) != 0):
                        count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == s)
                        print("\nsplit interval from: " + str(
                            count_Signal_beginn + int(min_max_interval_split_data)) + "   split interval to:" + str(
                            count_Signal_beginn + int(
                                len(X_1[count_Signal_beginn:count_Signal_end,
                                    :]) - min_max_interval_split_data)))  # -int(len(X_1[count_Signal_beginn:count_Signal_end, :])*self.split_test_size)
                        random_KFold_test_data_begin = int(
                            random.uniform((count_Signal_beginn + int(min_max_interval_split_data)),
                                           (count_Signal_beginn + int(
                                               len(X_1[count_Signal_beginn:count_Signal_end,
                                                   :]) - min_max_interval_split_data - int(len(
                                                   X_1[count_Signal_beginn:count_Signal_end,
                                                   :]) * self.split_test_size)))))
                        random_KFold_test_data_end = random_KFold_test_data_begin + int(
                            self.split_test_size * len(X_1[count_Signal_beginn:count_Signal_end, :]))
                        print("split from: " + str(random_KFold_test_data_begin) + "   split to: " + str(
                            random_KFold_test_data_end))
                        training_data_1_idx = np.arange(count_Signal_beginn, random_KFold_test_data_begin)
                        training_data_2_idx = np.arange(random_KFold_test_data_end, count_Signal_beginn + len(
                            X_1[count_Signal_beginn:count_Signal_end, :]))
                        training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
                        test_data_idx = np.arange(random_KFold_test_data_begin, random_KFold_test_data_end)

                        X_train_tm = X_1[training_data_1_2_idx]
                        Y_train_tm = Y_1[training_data_1_2_idx]
                        time_train_tm = time[training_data_1_2_idx]
                        X_Signal_number_neu = X_Signal_number_dynamisch + 1
                        X_Signal_number_train_tm = np.vstack(
                            (X_Signal_number_dynamisch[training_data_1_idx], X_Signal_number_neu[training_data_2_idx]))
                        X_Signal_number_dynamisch = X_Signal_number_dynamisch + 1
                        X_test_tm = X_1[test_data_idx]
                        Y_test_tm = Y_1[test_data_idx]
                        time_test_tm = time[test_data_idx]
                        X_Signal_number_test_tm = X_Signal_number[test_data_idx]
                        X_train = np.vstack((X_train, X_train_tm))
                        X_test = np.vstack((X_test, X_test_tm))
                        Y_train = np.vstack((Y_train, Y_train_tm))
                        Y_test = np.vstack((Y_test, Y_test_tm))
                        time_train = np.hstack((time_train, time_train_tm))
                        time_test = np.hstack((time_test, time_test_tm))
                        X_Signal_number_train = np.vstack((X_Signal_number_train, X_Signal_number_train_tm));
                        X_Signal_number_test = np.vstack((X_Signal_number_test, X_Signal_number_test_tm))
                        count_Signal_beginn = count_Signal_end

                X_train, Y_train, X_Signal_number_train = Tensorflow.cut_Signal_into_sequencen_train(self, X_train, Y_train,X_Signal_number_train)
                X_test, Y_test, X_Signal_number_test = Tensorflow.cut_Signal_into_sequencen_test(self, X_test, Y_test,X_Signal_number_test)

                normalizer_x = self.scaler
                X_train_normalize = normalizer_x.fit_transform(X_train)
                X_test_normalize = normalizer_x.transform(X_test)
                X_train_time_series_normalize = pd.DataFrame(X_train_normalize)
                X_test_time_series_normalize = pd.DataFrame(X_test_normalize)

                normalizer_y = self.scaler
                Y_train_normalize = normalizer_y.fit_transform(Y_train)
                Y_test_normalize = normalizer_y.transform(Y_test)
                Y_train_time_series_normalize = pd.DataFrame(Y_train_normalize)
                Y_test_time_series_normalize = pd.DataFrame(Y_test_normalize)

        X_train_time_series, Y_train_time_series, time_train = Tensorflow.shape_the_Signal(self=self,
                                                                                           X_Signal_number_train=X_Signal_number_train,
                                                                                           X_train_time_series_normalize=X_train_time_series_normalize,
                                                                                           Y_train_time_series_normalize=Y_train_time_series_normalize,
                                                                                           time_train=time_train,
                                                                                           i=(i - 1))
        X_test_time_series, Y_test_time_series, time_test = Tensorflow.shape_the_Signal(self=self,
                                                                                        X_Signal_number_train=X_Signal_number_test,
                                                                                        X_train_time_series_normalize=X_test_time_series_normalize,
                                                                                        Y_train_time_series_normalize=Y_test_time_series_normalize,
                                                                                        time_train=time_test,
                                                                                        i=(i - 1))

        return X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series, normalizer_x, normalizer_y, time_train, time_test

    def create_training_data_sequence_end_single_signal(self,X_1,Y_1,X_Signal_number,i):
        if(i==1):
            count_Signal_beginn = 0
            count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == X_Signal_number[0,0])
            X_train, X_test, Y_train, Y_test = train_test_split(X_1[0:count_Signal_end, :], Y_1[0:count_Signal_end],
                                                                test_size=self.split_test_size,
                                                                random_state=self.split_random_state,
                                                                shuffle=self.split_shuffle)
            X_Signal_number_train, X_Signal_number_test = train_test_split(
                X_Signal_number[count_Signal_beginn:count_Signal_end], test_size=self.split_test_size,
                random_state=self.split_random_state, shuffle=self.split_shuffle)
            count_Signal_beginn = count_Signal_end
            last_Signal_number = (X_Signal_number[-1, :])
            for s in range(1, last_Signal_number[0] + 1):
                if (np.count_nonzero(X_Signal_number == s) != 0):
                    count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == s)
                    X_train_tm, X_test_tm, Y_train_tm, Y_test_tm = train_test_split(
                        X_1[count_Signal_beginn:count_Signal_end, :], Y_1[count_Signal_beginn:count_Signal_end],
                        test_size=self.split_test_size, random_state=self.split_random_state, shuffle=self.split_shuffle)
                    X_train = np.vstack((X_train, X_train_tm));
                    X_test = np.vstack((X_test, X_test_tm));
                    Y_train = np.vstack((Y_train, Y_train_tm));
                    Y_test = np.vstack((Y_test, Y_test_tm))


                    X_Signal_number_train_tm, X_Signal_number_test_tm = train_test_split(
                        X_Signal_number[count_Signal_beginn:count_Signal_end], test_size=self.split_test_size,
                        random_state=self.split_random_state, shuffle=self.split_shuffle)
                    X_Signal_number_train = np.vstack((X_Signal_number_train, X_Signal_number_train_tm));
                    X_Signal_number_test = np.vstack((X_Signal_number_test, X_Signal_number_test_tm))
                    count_Signal_beginn = count_Signal_end

            if (self.sequence_training == True):
                X_train_cut_tm = []
                Y_train_cut_tm = []
                X_Signal_number_train_cut_tm = []

                flag = 0
                for c in range(0, len(Y_train)):
                    if (self.sequence_training_cut_x[0] < Y_train[c, 0] < self.sequence_training_cut_x[1]
                            and self.sequence_training_cut_y[0] < Y_train[c, 1] < self.sequence_training_cut_y[1]
                            and self.sequence_training_cut_z[0] < Y_train[c, 2] < self.sequence_training_cut_z[1]):
                        X_train_cut_tm.append(X_train[c, :])
                        Y_train_cut_tm.append(Y_train[c, :])
                        X_Signal_number_train_cut_tm.append(X_Signal_number_train[c])
                        flag = 1
                    elif (flag == 1):
                        X_Signal_number_train = X_Signal_number_train + 1
                        flag = 0
                X_train = np.array(X_train_cut_tm)
                Y_train = np.array(Y_train_cut_tm)
                X_Signal_number_train = np.array(X_Signal_number_train_cut_tm)

            if (self.sequence_test == True):
                X_test_cut_tm = []
                Y_test_cut_tm = []
                X_Signal_number_test_cut_tm = []
                flag = 0
                for c in range(0, len(Y_test)):
                    if (self.sequence_test_cut_x[0] < Y_test[c, 0] < self.sequence_test_cut_x[1]
                            and self.sequence_test_cut_y[0] < Y_test[c, 1] < self.sequence_test_cut_y[1]
                            and self.sequence_test_cut_z[0] < Y_test[c, 2] < self.sequence_test_cut_z[1]):
                        X_test_cut_tm.append(X_test[c, :])
                        Y_test_cut_tm.append(Y_test[c, :])
                        X_Signal_number_test_cut_tm.append(X_Signal_number_test[c])
                        flag = 1
                    elif (flag == 1):
                        X_Signal_number_test = X_Signal_number_test + 1
                        flag = 0
                X_test = np.array(X_test_cut_tm)
                Y_test = np.array(Y_test_cut_tm)
                X_Signal_number_test = np.array(X_Signal_number_test_cut_tm)

            X_train, Y_train, X_Signal_number_train = Tensorflow.cut_Signal_into_sequencen_train(self, X_train, Y_train,X_Signal_number_train)
            X_test, Y_test, X_Signal_number_test = Tensorflow.cut_Signal_into_sequencen_test(self, X_test, Y_test,X_Signal_number_test)

            normalizer_x = self.scaler
            X_train_normalize = normalizer_x.fit_transform(X_train)
            X_test_normalize = normalizer_x.transform(X_test)
            X_train_time_series_normalize = pd.DataFrame(X_train_normalize)
            X_test_time_series_normalize = pd.DataFrame(X_test_normalize)

            normalizer_y = self.scaler
            Y_train_normalize = normalizer_y.fit_transform(Y_train)
            Y_test_normalize = normalizer_y.transform(Y_test)
            Y_train_time_series_normalize = pd.DataFrame(Y_train_normalize)
            Y_test_time_series_normalize = pd.DataFrame(Y_test_normalize)

        X_train_time_series, Y_train_time_series = Tensorflow.shape_the_Signal(self=self,
                                                                            X_Signal_number_train=X_Signal_number_train,
                                                                            X_train_time_series_normalize=X_train_time_series_normalize,
                                                                            Y_train_time_series_normalize=Y_train_time_series_normalize,
                                                                            i=(i - 1))
        X_test_time_series, Y_test_time_series = Tensorflow.shape_the_Signal(self=self,
                                                                            X_Signal_number_train=X_Signal_number_test,
                                                                            X_train_time_series_normalize=X_test_time_series_normalize,
                                                                            Y_train_time_series_normalize=Y_test_time_series_normalize,
                                                                            i=(i - 1))
        return X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series,normalizer_x,normalizer_y

    def create_training_data_sequence_end_complete_signal(self,X_1,Y_1,X_Signal_number,i):
        if(i==1):
            X_train, X_test, Y_train, Y_test = train_test_split(X_1, Y_1, test_size=self.split_test_size,
                                                                random_state=self.split_random_state,
                                                                shuffle=self.split_shuffle)
            X_Signal_number_train, X_Signal_number_test = train_test_split(X_Signal_number, test_size=self.split_test_size,
                                                                           random_state=self.split_random_state,
                                                                           shuffle=self.split_shuffle)

            X_train, Y_train, X_Signal_number_train = Tensorflow.cut_Signal_into_sequencen_train(self, X_train, Y_train,X_Signal_number_train)
            X_test, Y_test, X_Signal_number_test = Tensorflow.cut_Signal_into_sequencen_test(self, X_test, Y_test,X_Signal_number_test)

            normalizer_x = self.scaler
            X_train_normalize = normalizer_x.fit_transform(X_train)
            X_test_normalize = normalizer_x.transform(X_test)
            X_train_time_series_normalize = pd.DataFrame(X_train_normalize)
            X_test_time_series_normalize = pd.DataFrame(X_test_normalize)

            normalizer_y = self.scaler
            Y_train_normalize = normalizer_y.fit_transform(Y_train)
            Y_test_normalize = normalizer_y.transform(Y_test)
            Y_train_time_series_normalize = pd.DataFrame(Y_train_normalize)
            Y_test_time_series_normalize = pd.DataFrame(Y_test_normalize)

        X_train_time_series, Y_train_time_series = Tensorflow.shape_the_Signal(self=self,
                                                                               X_Signal_number_train=X_Signal_number_train,
                                                                               X_train_time_series_normalize=X_train_time_series_normalize,
                                                                               Y_train_time_series_normalize=Y_train_time_series_normalize,
                                                                               i=(i - 1))
        X_test_time_series, Y_test_time_series = Tensorflow.shape_the_Signal(self=self,
                                                                             X_Signal_number_train=X_Signal_number_test,
                                                                             X_train_time_series_normalize=X_test_time_series_normalize,
                                                                             Y_train_time_series_normalize=Y_test_time_series_normalize,
                                                                             i=(i - 1))
        return X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series, normalizer_x, normalizer_y

    def training_NN(self):
        time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(self.path+self.filename_training,delimiter=';')
#
        if(self.all_or_range_selection_Data==True):
            time = time_xyz_antennen_Signal_Komplex_all_Files[:,self.time]
            X_Signal_number = (time_xyz_antennen_Signal_Komplex_all_Files[:, self.X_Signal_number:(self.X_Signal_number + 1)].astype(int))
            X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,self.Input_from:self.Input_to]#[:,9:]#
            Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,self.Output_from:self.Output_to]#[:,2:5]#

            index_for_Regression=0
            for r in range(0,len(X_Signal_number)):
                if(int(time_xyz_antennen_Signal_Komplex_all_Files[r,-1])>0):
                    index_for_Regression= index_for_Regression+1
            time = time[0:index_for_Regression]
            X_Signal_number = X_Signal_number[0:index_for_Regression,:]
            X_1 = X_1[0:index_for_Regression,:]
            Y_1 = Y_1[0:index_for_Regression,:]
        else:
            time = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.time]
            X_Signal_number = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.X_Signal_number:(self.X_Signal_number + 1)]
            X_1 = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.Input_from:self.Input_to]  # [:,9:]#
            Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.Output_from:self.Output_to]  # [:,2:5]#

        if (self.threshold_antenna == True):
            X_1[X_1 < self.threshold_value] = 0

        last_Signal_number = (X_Signal_number[-1, :])
        count_begin = 0
        time_ver = 0

        count_end = count_begin + np.count_nonzero(X_Signal_number == 0)
        time_tm = (time[count_begin:count_end])
        time_ver = time_ver + time_tm[count_end - 1]
        time_2 = time + time_ver
        count_begin = count_end

        for t in range(1, last_Signal_number[0] + 1):
            count_end = count_begin + np.count_nonzero(X_Signal_number == t)
            time_tm = np.hstack((time_tm, time_2[count_begin:count_end]))
            time_ver = time_tm[count_end - 1]
            time_2 = time + time_ver
            count_begin = count_end

        time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz = np.loadtxt(self.path_test_sequenz+self.filename_test_sequenz,delimiter=';')
        time_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,0]

        X_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,7:40]
        Y_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,1:4]

        print("X_beginn_shape =",X_1.shape)
        print("Y_beginn_shape =",Y_1.shape)

        #X_1 = pd.DataFrame(X_1,columns=["frame1_real","frame1_imag","frame2_real","frame2_imag","frame3_real","frame3_imag","frame4_real","frame4_imag","frame5_real","frame5_imag","frame6_real","frame6_imag","frame7_real","frame7_imag","frame8_real","frame8_imag","main1_real","main1_imag","main2_real","main2_imag","main3_real","main3_imag","main4_real","main4_imag","main5_real","main5_imag","main6_real","main6_imag","main7_real","main7_imag","main8_real","main8_imag",])
        #Y_1 = pd.DataFrame(Y_1,columns=['X_postion'])#,'Y_position','Z_position'])#

        for i in range(1,(self.time_steps_geo_folge+1)):
            print("Time Steps: "+str(self.creat_dataset_time_steps*(2**(i-1))))

            # Random KFold (0.34-0.54, ...???)
            X_train_time_series,Y_train_time_series,X_test_time_series,Y_test_time_series,normalizer_x ,normalizer_y,time_train,time_test = Tensorflow.create_training_data_random_KFold(self,X_1,Y_1,X_Signal_number,i,time_tm)

            # KFold per steps (0.0-0.2, 0.2-0.4,....)
            #X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series,normalizer_x,normalizer_y = Tensorflow.create_training_data_KFold(self, X_1, Y_1, X_Signal_number, i)

            # old version (0.8-1) the last
            #X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series,normalizer_x,normalizer_y = Tensorflow.create_training_data_sequence_end_single_signal(self, X_1, Y_1, X_Signal_number, i)

            print("X_Time_Series_generator_shape =",X_train_time_series.shape)
            print("Y_Time_Series_generator_shape =",Y_train_time_series.shape)

            model = tf.keras.Sequential([#
                tf.keras.layers.Conv1D(32,3,strides=1, activation='relu',padding="same", input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(32,3, activation='relu',padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.SpatialDropout1D(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                #tf.keras.layers.BatchNormalization(), # important that ist commented out
                #tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=Y_train_time_series.shape[1])
            ])
            tf.keras.utils.plot_model(model)
            model.summary()
            time_stamp_from_NN =Tensorflow.get_actuall_time_stamp()
            doku = []
            for loss in range(0, len(self.loss_funktions)):
                print('\n\n' + self.loss_funktions[loss])
                model.compile(loss=self.loss_funktions[loss],
                              optimizer=self.optimizer, metrics=self.metrics)
                # ,decay=1e-6,nesterov=True,momentum=0.9
                start_time_count = time_1.perf_counter()
                history = model.fit(X_train_time_series, Y_train_time_series, epochs=self.trainings_epochs,
                                    shuffle=True, verbose=self.verbose, batch_size=self.batch_size,
                                    validation_data=(X_test_time_series, Y_test_time_series))  # x=generator
                elapsed_time = time_1.perf_counter() - start_time_count
                path_NN = os.path.join(self.path + self.save_dir)
                if (False == os.path.isdir(path_NN)):
                    os.mkdir(path_NN)
                path_NN_tm = os.path.join(path_NN + '\\NN_from_' + time_stamp_from_NN)
                if (False == os.path.isdir(path_NN_tm)):
                    os.mkdir(path_NN_tm)
                path_NN_tm_loss = os.path.join(path_NN_tm + '\\' + self.loss_funktions[loss] + '_Time_Step_' + str(
                    self.creat_dataset_time_steps * (2 ** (i - 1))))
                if (False == os.path.isdir(path_NN_tm_loss)):
                    os.mkdir(path_NN_tm_loss)
                path_NN_tm_loss = os.path.join(path_NN_tm_loss + '\\' + self.loss_funktions[loss])
                try:
                    model.save(os.path.join(path_NN_tm_loss + ".h5"))
                except:
                    print("The model not saved")
                # print(model.evaluate(X_test_time_series,Y_test_time_series))
                Y_pred_time_series = model.predict(X_test_time_series)  # [:4000]

                Y_pred_time_series = normalizer_y.inverse_transform(Y_pred_time_series)
                Y_test_time_series_inverse = normalizer_y.inverse_transform(Y_test_time_series)  # [:4000]

                error_test= Y_pred_time_series[:, :] - Y_test_time_series_inverse[:, :]
                mse_test = np.mean((error_test) ** 2)
                print("MSE Test: " + str(mse_test) + "mm")

                fig = plt.figure(1)
                plt.subplot(2, 1, 1)
                plt.title('X Position')
                plt.plot(time_test, Y_pred_time_series[:, 0])
                plt.plot(time_test, Y_test_time_series_inverse[:, 0])
                plt.legend(["X_pred", "X_true"])
                plt.subplot(2, 1, 2)
                plt.plot(time_test, np.sqrt((Y_pred_time_series[:, 0] - Y_test_time_series_inverse[:, 0]) ** 2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Training_X.png")
                if (self.pickle == True):
                    with open(path_NN_tm_loss + "_Training_X.pkl", "wb") as fp:  # Save Plots
                        pickle.dump(fig, fp, protocol=4)
                if (loss < len(self.loss_funktions) - 1 or i < (self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(2)
                plt.subplot(2, 1, 1)
                plt.title('Y Position')
                plt.plot(time_test, Y_pred_time_series[:, 1])
                plt.plot(time_test, Y_test_time_series_inverse[:, 1])
                plt.legend(["Y_pred", "Y_true"])
                plt.subplot(2, 1, 2)
                plt.plot(time_test, np.sqrt((Y_pred_time_series[:, 1] - Y_test_time_series_inverse[:, 1]) ** 2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Training_Y.png")
                if (self.pickle == True):
                    with open(path_NN_tm_loss + "_Training_Y.pkl", "wb") as fp:  # Save Plots
                        pickle.dump(fig, fp, protocol=4)
                if (loss < len(self.loss_funktions) - 1 or i < (self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(3)
                plt.subplot(2, 1, 1)
                plt.title('Z Position')
                plt.plot(time_test, Y_pred_time_series[:, 2])
                plt.plot(time_test, Y_test_time_series_inverse[:, 2])
                plt.legend(["Z_pred", "Z_true"])
                plt.subplot(2, 1, 2)
                plt.plot(time_test, np.sqrt((Y_pred_time_series[:, 2] - Y_test_time_series_inverse[:, 2]) ** 2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Training_Z.png")
                if (self.pickle == True):
                    with open(path_NN_tm_loss + "_Training_Z.pkl", "wb") as fp:  # Save Plots
                        pickle.dump(fig, fp, protocol=4)

                Y_pred_train_time_series = model.predict(X_train_time_series)  # [:4000]
                Y_pred_train_time_series = normalizer_y.inverse_transform(Y_pred_train_time_series)
                Y_train_time_series_inverse = normalizer_y.inverse_transform(Y_train_time_series)  # [:4000]

                error_train = Y_pred_train_time_series[:, :] - Y_train_time_series_inverse[:, :]
                mse_train = np.mean((error_train) ** 2)
                print("MSE Train: " + str(mse_train) + "mm")

                if (loss < len(self.loss_funktions) - 1 or i < (self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(4)
                plt.subplot(2, 1, 1)
                plt.title('X Position train')
                plt.plot(time_train, Y_pred_train_time_series[:, 0])
                plt.plot(time_train, Y_train_time_series_inverse[:, 0])
                plt.legend(["X_pred", "X_true"])
                plt.subplot(2, 1, 2)
                plt.plot(time_train, np.sqrt((Y_pred_train_time_series[:, 0] - Y_train_time_series_inverse[:, 0]) ** 2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Test_X.png")
                if (self.pickle == True):
                    with open(path_NN_tm_loss + "_Test_X.pkl", "wb") as fp:  # Save Plots
                        pickle.dump(fig, fp, protocol=4)
                if (loss < len(self.loss_funktions) - 1 or i < (self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(5)
                plt.subplot(2, 1, 1)
                plt.title('Y Position train')
                plt.plot(time_train, Y_pred_train_time_series[:, 1])
                plt.plot(time_train, Y_train_time_series_inverse[:, 1])
                plt.legend(["Y_pred", "Y_true"])
                plt.subplot(2, 1, 2)
                plt.plot(time_train, np.sqrt((Y_pred_train_time_series[:, 1] - Y_train_time_series_inverse[:, 1]) ** 2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Test_Y.png")
                if (self.pickle == True):
                    with open(path_NN_tm_loss + "_Test_Y.pkl", "wb") as fp:  # Save Plots
                        pickle.dump(fig, fp, protocol=4)
                if (loss < len(self.loss_funktions) - 1 or i < (self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(6)
                plt.subplot(2, 1, 1)
                plt.title('Z Position train')
                plt.plot(time_train, Y_pred_train_time_series[:, 2])
                plt.plot(time_train, Y_train_time_series_inverse[:, 2])
                plt.legend(["Z_pred", "Z_true"])
                plt.subplot(2, 1, 2)
                plt.plot(time_train, np.sqrt((Y_pred_train_time_series[:, 2] - Y_train_time_series_inverse[:, 2]) ** 2))

                plt.figure(9)
                plt.plot(Y_train_time_series[:, 0])
                plt.figure(10)
                plt.plot(Y_train_time_series[:, 1])
                plt.figure(11)
                plt.plot(Y_train_time_series[:, 2])
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Test_Z.png")
                if (self.pickle == True):
                    with open(path_NN_tm_loss + "_Test_Z.pkl", "wb") as fp:  # Save Plots
                        pickle.dump(fig, fp, protocol=4)
                if (loss < len(self.loss_funktions) - 1 or i < (self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                # result_train =
                print(
                    '\n------------------------------------------------------------------------------------------------\n')
                print('\n\n' + self.loss_funktions[loss])
                print('Loss and MSE Training:', history.history['loss'][-1], history.history['MSE'][-1])
                print('Loss and MSE val_Training:', history.history['val_loss'][-1], history.history['val_MSE'][-1])
                print('Training Time: %.3f seconds.' % elapsed_time)
                result = model.evaluate(X_test_time_series, Y_test_time_series, batch_size=self.batch_size)
                # print("loss (test-set):", str(result))
                # print('loss test_sequenz:', str(model.evaluate(x=np.expand_dims(X_test_time_series, axis=0),y=np.expand_dims(Y_test_time_series, axis=0))))
                print(
                    '\n------------------------------------------------------------------------------------------------\n')
                if (loss == 0):
                    doku.append("Model:")
                    doku.append('\n' + str(model.inputs))
                    for j in range(0, len(model._layers)):
                        if (str(model._layers[j].output.op.inputs) != "()"):
                            doku.append('\n' + str(model._layers[j].output.op.inputs))
                    doku.append('\n')
                doku.append(
                    '\n------------------------------------------------------------------------------------------------')
                doku.append("\nX_Time_Series_generator_shape =" + str(
                    X_train_time_series.shape) + "\nY_Time_Series_generator_shape =" + str(Y_train_time_series.shape))
                doku.append(
                    "\nTime Steps: " + str(self.creat_dataset_time_steps * (2 ** (i - 1))) + ", Overlap: " + str(
                        self.create_dataset_overlap) + ", Offset: " + str(self.create_dataset_offset))
                doku.append("\nScaler: " + str(self.scaler))
                doku.append(
                    '\n' + self.loss_funktions[loss] + " Loss, " + str(self.trainings_epochs) + " Epochs, " + str(
                        self.learning_rate) + " Learning Rate, " + str(self.optimizer_name) + " Optimizer, " + str(
                        self.batch_size) + " Batch Size, " + str(self.metrics) + " Metrics")
                doku.append('\n' + self.loss_funktions[loss] + ', Loss Training: ' + str(
                    history.history['loss'][-1]) + '  Train Metrics' + str(self.metrics) + ':' + str(
                    history.history['MSE'][-1]))
                doku.append('\n' + self.loss_funktions[loss] + ', Loss Validation: ' + str(
                    history.history['val_loss'][-1]) + '  Validation Metrics' + str(self.metrics) + ':' + str(
                    history.history['val_MSE'][-1]))
                doku.append('\nTraining Time: %.3f seconds.' % elapsed_time)
                # doku.append("\nloss (test-set): " +str(result))
                # doku.append('loss test_sequenz:'+str(model.evaluate(x=np.expand_dims(X_test_time_series, axis=0),y=np.expand_dims(Y_test_time_series, axis=0))))
                doku.append(
                    '\n------------------------------------------------------------------------------------------------\n')

            open_path = os.path.join(
                self.path + self.save_dir + '\\NN_from_' + time_stamp_from_NN + '\\Doku_Time_Step_' + str(
                    self.creat_dataset_time_steps * (2 ** (i - 1))))
            file = open(open_path + '.txt', 'w')
            for d in range(0, len(doku)):
                file.write(str(doku[d]))
            file.close()
        plt.show()

if __name__ == '__main__':
    conf = config_Test_Tensorflow
    tensor = Tensorflow(conf.creat_dataset_time_steps,conf.time_steps_geo_folge,conf.create_dataset_overlap,conf.create_dataset_offset,conf.sequence_training,
                        conf.sequence_training_cut_x,conf.sequence_training_cut_y,conf.sequence_training_cut_z,conf.sequence_test,conf.sequence_test_cut_x,
                        conf.sequence_test_cut_y,conf.sequence_test_cut_z,conf.path, conf.path_test_sequenz,conf.filename_training,
                        conf.filename_test_sequenz, conf.save_dir, conf.all_or_range_selection_Data,
                        conf.training_Data_from, conf.training_Data_to,
                        conf.time, conf.X_Signal_number, conf.Input_from, conf.Input_to, conf.Output_from,
                        conf.Output_to, conf.trainings_epochs, conf.batch_size,
                        conf.verbose, conf.metrics, conf.learning_rate, conf.optimizer_name, conf.optimizer,
                        conf.split_test_size, conf.split_KFold_set, conf.split_random_state,
                        conf.split_shuffle, conf.loss_funktions, conf.scaler,conf.threshold_antenna,conf.threshold_value, conf.pickle, conf.png,conf.kernel_init,conf.bias_init)
    tensor.training_NN()