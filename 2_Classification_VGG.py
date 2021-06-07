from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from datetime import datetime
import time as time_1
import pandas as pd
import os
import random
import pickle
#import tensorflow_probability as tf

class config_Test_Tensorflow():
    # self.a = outside.
    # Time Steps (create_dataset)
    creat_dataset_time_steps = 16
    time_steps_geo_folge = 1  # geometrische-Folge(zweier Potenzen) for time steps, example: (time_steps=16 andtime_steps_geo_folge=2)->  1
    create_dataset_overlap = 11  # example time_steps=16(0-15); Overlap from Feature to Feature example: 0= no overlap, 15= 0-15,1-16
    create_dataset_offset = 7 # example time_steps=16(0-15)  the middle for y is = offset 7-8 (from 0-15) #0 first value, 15 last value

    #### Trainingsdata into Trainingssequence (Only No Cut Files!!) (with shape)
    sequence_training = True
    sequence_training_cut_x = [0,1500]
    sequence_training_cut_y = [0,1200]
    sequence_training_cut_z = [-100,400]

    #### Testdata into Trainingssequence
    sequence_test = True
    sequence_test_cut_x = [0, 1500]
    sequence_test_cut_y = [0, 1200]
    sequence_test_cut_z = [-100, 400]

    # Path
    path = r"C:\Users\dauserml\Documents\2020_11_16"
    path_test_sequenz = r"C:\Users\dauserml\Documents\2020_11_16"
    filename_training = "\\2020_11_16_files_No_cut_with_interp_40.csv"#"\\all_Files_C_R.csv"
    filename_test_sequenz = "\\2020_11_16_files_Z_800.csv"
    save_dir = '\\VGG_Time_Series_Classification'
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
    Output_from = 2  # Ouput Signal from 41 -> Class
    Output_to = 5

    # Training Options
    trainings_epochs = 200
    batch_size = 128
    verbose = 2
    metrics = ['sparse_categorical_accuracy']
    learning_rate = 0.0007
    optimizer_name = "Adam"
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)#,momentum=0.9,nesterov=True,decay=1e-6)

    ## Split Options
    split_test_size = 0.2
    split_KFold_set = 2 # bei 0.2 -> 5 Sets (0->0.0-0.2, 1->0.2-0.4, 2 -> 0.4-0.6 ...... 4 -> 0.8-1.0)
    split_random_state = 42
    split_shuffle = False

    ## Loss Options
    # self.loss_funktions= ['MSE','huber_loss'] #,'MSE','huber_loss','MAP' Probabilistic lossfunktion (Bayesian) lambda y, p_y: -p_y.log_prob(y)   #
    loss_funktions = ['sparse_categorical_crossentropy'] # ,'huber_loss',
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
    def create_dataset(X, y,Class, time_steps=1, overlap=1, offset=0):
        if (offset < time_steps and overlap < time_steps):
            overlap = time_steps - (overlap)
            Xs, ys,Cl = [], [],[]
            for i in range(int((len(X) - (time_steps)) / overlap)+1):
                v = X.iloc[(i * overlap):((i * overlap) + time_steps)].values
                Xs.append(v)
                ys.append(y.iloc[(i * overlap) + offset])
                Cl.append(Class[(i * overlap) + offset])
            return np.array(Xs), np.array(ys), np.array(Cl)
        else:
            print("offset or overlap to high!!")

    def cut_Signal_into_sequencen_train(self,X_train,Y_train,X_Signal_number_train,Class_train):
        if (self.sequence_training == True):
            X_train_cut_tm = [];X_train_cut_NaN = [];
            Class_train_tm =[]; Class_train_cut_NaN =[];
            Y_train_cut_tm = [];Y_train_cut_NaN = [];
            X_Signal_number_train_cut_tm = []; X_Signal_number_train_cut_NaN=[]

            flag = 0
            for c in range(0, len(Y_train)):
                if (self.sequence_training_cut_x[0] < Y_train[c, 0] < self.sequence_training_cut_x[1]
                        and self.sequence_training_cut_y[0] < Y_train[c, 1] < self.sequence_training_cut_y[1]
                        and self.sequence_training_cut_z[0] < Y_train[c, 2] < self.sequence_training_cut_z[1]):
                    X_train_cut_tm.append(X_train[c, :])
                    Class_train_tm.append(Class_train[c])
                    Y_train_cut_tm.append(Y_train[c, :])
                    X_Signal_number_train_cut_tm.append(X_Signal_number_train[c,:])
                    flag = 1
                elif (flag == 1):
                    X_Signal_number_train = X_Signal_number_train + 1
                    ## No intervention Training
                    X_train_cut_NaN.append(X_train[c, :])
                    Class_train[c,:] = 0
                    Class_train_cut_NaN.append(Class_train[c,:] )
                    #Y_train[c, :] = [8000,8000,8000]
                    Y_train_cut_NaN.append(Y_train[c, :])
                    X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c,:])
                    flag = 0
                else:
                    #X_Signal_number_train = X_Signal_number_train + 1
                    ## No intervention Training
                    X_train_cut_NaN.append(X_train[c, :])
                    Class_train[c, :] = 0
                    Class_train_cut_NaN.append(Class_train[c, :])
                    #Y_train[c, :] = [8000, 8000, 8000]
                    Y_train_cut_NaN.append(Y_train[c, :])
                    X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c, :])

            last_X_Number_train_value = X_Signal_number_train[-1]
            X_Signal_number_train_cut_NaN = np.array(X_Signal_number_train_cut_NaN)
            X_Signal_number_train_cut_NaN[:,:] = (last_X_Number_train_value+1)
            X_train_cut_tm = X_train_cut_tm + X_train_cut_NaN
            Class_train =  np.vstack((np.array(Class_train_tm) , Class_train_cut_NaN))
            Y_train_cut_tm = Y_train_cut_tm + Y_train_cut_NaN
            X_Signal_number_train = np.vstack((np.array(X_Signal_number_train_cut_tm) , X_Signal_number_train_cut_NaN))

            X_train = np.array(X_train_cut_tm)
            #Class_train = np.array(Class_train_tm)
            Y_train = np.array(Y_train_cut_tm)
            #X_Signal_number_train = np.array(X_Signal_number_train_cut_tm)
        return X_train,Y_train,X_Signal_number_train,Class_train

    def cut_Signal_into_sequencen_test(self,X_test,Y_test,X_Signal_number_test,Class_test):
        if (self.sequence_test == True):
            X_test_cut_tm = [];X_test_cut_NaN = [];
            Class_test_tm = [];Class_test_cut_NaN = [];
            Y_test_cut_tm = [];Y_test_cut_NaN = [];
            X_Signal_number_test_cut_tm = []; X_Signal_number_test_cut_NaN=[]

            flag = 0
            for c in range(0, len(Y_test)):
                if (self.sequence_test_cut_x[0] < Y_test[c, 0] < self.sequence_test_cut_x[1]
                        and self.sequence_test_cut_y[0] < Y_test[c, 1] < self.sequence_test_cut_y[1]
                        and self.sequence_test_cut_z[0] < Y_test[c, 2] < self.sequence_test_cut_z[1]):
                    X_test_cut_tm.append(X_test[c, :])
                    Class_test_tm.append(Class_test[c])
                    Y_test_cut_tm.append(Y_test[c, :])
                    X_Signal_number_test_cut_tm.append(X_Signal_number_test[c,:])
                    flag = 1
                elif (flag == 1):
                    X_Signal_number_test = X_Signal_number_test + 1
                    ## No intervention Training
                    X_test_cut_NaN.append(X_test[c, :])
                    Class_test[c, :] = 0
                    Class_test_cut_NaN.append(Class_test[c, :])
                    #Y_test[c, :] = [8000, 8000, 8000]
                    Y_test_cut_NaN.append(Y_test[c, :])
                    X_Signal_number_test_cut_NaN.append(X_Signal_number_test[c, :])
                    flag = 0
                else:
                    # X_Signal_number_train = X_Signal_number_train + 1
                    ## No intervention Training
                    X_test_cut_NaN.append(X_test[c, :])
                    Class_test[c, :] = 0
                    Class_test_cut_NaN.append(Class_test[c, :])
                    #Y_test[c, :] = [8000, 8000, 8000]
                    Y_test_cut_NaN.append(Y_test[c, :])
                    X_Signal_number_test_cut_NaN.append(X_Signal_number_test[c, :])

            last_X_Number_test_value = X_Signal_number_test[-1]
            X_Signal_number_test_cut_NaN = np.array(X_Signal_number_test_cut_NaN)
            X_Signal_number_test_cut_NaN[:, :] = (last_X_Number_test_value + 1)
            X_test_cut_tm = X_test_cut_tm + X_test_cut_NaN
            Class_test = np.vstack((np.array(Class_test_tm), Class_test_cut_NaN))
            Y_test_cut_tm = Y_test_cut_tm + Y_test_cut_NaN
            X_Signal_number_test = np.vstack((np.array(X_Signal_number_test_cut_tm), X_Signal_number_test_cut_NaN))

            X_test = np.array(X_test_cut_tm)
            # Class_test = np.array(Class_test_tm)
            Y_test = np.array(Y_test_cut_tm)
            # X_Signal_number_test= np.array(X_Signal_number_test_cut_tm)
        return X_test,Y_test,X_Signal_number_test,Class_test

    def shape_the_Signal(self,X_Signal_number_train,X_train_time_series_normalize,Y_train_time_series_normalize,Class_train,i):
        count = 0
        X_train_time_series = []
        Class=[]
        Y_train_time_series = []
        first_data = 0
        for j in range(int(X_Signal_number_train[0]),int(X_Signal_number_train[-1] + 1)): #int(max(X_Signal_number_train))
            of = 0 + count
            count = count + np.count_nonzero(X_Signal_number_train[:] == j)
            to = count

            if(self.creat_dataset_time_steps*(2**i)<np.count_nonzero(X_Signal_number_train[:] == j)):
                X_train_time_series_tm, Y_train_time_series_tm,Class_train_tm = Tensorflow.create_dataset(X_train_time_series_normalize[of:to][:],
                                                                                           Y_train_time_series_normalize[of:to][:],
                                                                                           Class_train[of:to],
                                                                                           time_steps=self.creat_dataset_time_steps * (2 ** i),
                                                                                           overlap=self.create_dataset_overlap,offset=self.create_dataset_offset)
                if(len(X_train_time_series_tm)!=0 and len(X_train_time_series)==0 ):
                    X_train_time_series = X_train_time_series_tm
                    Class = Class_train_tm
                    Y_train_time_series = Y_train_time_series_tm
                elif(len(X_train_time_series)!=0 and len(X_train_time_series_tm)!=0):
                    X_train_time_series= np.vstack((X_train_time_series,X_train_time_series_tm))
                    Class = np.vstack((Class,Class_train_tm))
                    Y_train_time_series = np.vstack((Y_train_time_series, Y_train_time_series_tm))
        return X_train_time_series,Y_train_time_series,Class

    def create_training_data_KFold(self, X_1, Y_1, X_Signal_number, i):
        count_Signal_beginn = 0
        count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == X_Signal_number[0,0])

        min_max_interval_split_data = self.creat_dataset_time_steps * 4

        if(self.split_test_size*self.split_KFold_set<=1):
            if(i==1):
                if (min_max_interval_split_data * 2 < len(X_1[0:count_Signal_end, :])):
                    print("split interval from: " + str(
                        count_Signal_beginn) + "   split interval to:" + str(
                        count_Signal_beginn + int(len(X_1[count_Signal_beginn:count_Signal_end, :]))))
                    random_KFold_test_data_begin = 0 + int((self.split_test_size*self.split_KFold_set)*len(X_1[count_Signal_beginn:count_Signal_end, :]))
                    random_KFold_test_data_end = 0 + int((self.split_test_size*(self.split_KFold_set+1))*len(X_1[count_Signal_beginn:count_Signal_end, :]))
                    print(
                        "split from: " + str(random_KFold_test_data_begin) + "   split to: " + str(random_KFold_test_data_end))
                    training_data_1_idx = np.arange(0, random_KFold_test_data_begin)
                    training_data_2_idx = np.arange(random_KFold_test_data_end, len(X_1[0:count_Signal_end, :]))
                    training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
                    test_data_idx = np.arange(random_KFold_test_data_begin, random_KFold_test_data_end)

                    X_train = X_1[training_data_1_2_idx]
                    Y_train = Y_1[training_data_1_2_idx]
                    X_Signal_number_dynamisch = X_Signal_number
                    X_Signal_number_neu = X_Signal_number_dynamisch + 1
                    X_Signal_number_train = np.vstack(
                        (X_Signal_number_dynamisch[training_data_1_idx], X_Signal_number_neu[training_data_2_idx]))
                    X_Signal_number_dynamisch = X_Signal_number_dynamisch + 1

                    X_test = X_1[test_data_idx]
                    Y_test = Y_1[test_data_idx]
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
                                (self.split_test_size * self.split_KFold_set) * len(X_1[count_Signal_beginn:count_Signal_end, :]))
                            random_KFold_test_data_end = count_Signal_beginn + int((self.split_test_size * (self.split_KFold_set + 1)) * len(
                                X_1[count_Signal_beginn:count_Signal_end, :]))
                            print("split from: " + str(random_KFold_test_data_begin) + "   split to: " + str(
                                random_KFold_test_data_end))
                            training_data_1_idx = np.arange(count_Signal_beginn, random_KFold_test_data_begin)
                            training_data_2_idx = np.arange(random_KFold_test_data_end,
                                                            count_Signal_beginn + len(X_1[count_Signal_beginn:count_Signal_end, :]))
                            training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
                            test_data_idx = np.arange(random_KFold_test_data_begin, random_KFold_test_data_end)

                            X_train_tm = X_1[training_data_1_2_idx]
                            Y_train_tm = Y_1[training_data_1_2_idx]
                            X_Signal_number_neu = X_Signal_number_dynamisch + 1
                            X_Signal_number_train_tm = np.vstack(
                                (X_Signal_number_dynamisch[training_data_1_idx], X_Signal_number_neu[training_data_2_idx]))
                            X_Signal_number_dynamisch = X_Signal_number_dynamisch + 1
                            X_test_tm = X_1[test_data_idx]
                            Y_test_tm = Y_1[test_data_idx]
                            X_Signal_number_test_tm = X_Signal_number[test_data_idx]
                            X_train = np.vstack((X_train, X_train_tm))
                            X_test = np.vstack((X_test, X_test_tm))
                            Y_train = np.vstack((Y_train, Y_train_tm))
                            Y_test = np.vstack((Y_test, Y_test_tm))
                            X_Signal_number_train = np.vstack((X_Signal_number_train, X_Signal_number_train_tm));
                            X_Signal_number_test = np.vstack((X_Signal_number_test, X_Signal_number_test_tm))
                            count_Signal_beginn = count_Signal_end

                    X_train, Y_train, X_Signal_number_train = Tensorflow.cut_Signal_into_sequencen_train(self, X_train,Y_train,X_Signal_number_train)
                    X_test, Y_test, X_Signal_number_test = Tensorflow.cut_Signal_into_sequencen_test(self, X_test,Y_test,X_Signal_number_test)

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
        else:
            print("Split KFold to high!!")
        return X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series,normalizer_x,normalizer_y

    def create_training_test_data_random_KFold(self,X_1,Y_1,X_Signal_number,Class,count_Signal_beginn,count_Signal_end,min_max_interval_split_data):
        print("split interval from: " + str(
            count_Signal_beginn + int(min_max_interval_split_data)) + "   split interval to:" + str(
            count_Signal_beginn + int(len(X_1[count_Signal_beginn:count_Signal_end,
                                          :]) - min_max_interval_split_data)))  # - int(len(X_1[count_Signal_beginn:count_Signal_end, :])*self.split_test_size)
        random_KFold_test_data_begin = int(random.uniform(int(min_max_interval_split_data), (int(
            len(X_1[0:count_Signal_end, :]) - min_max_interval_split_data - int(
                len(X_1[0:count_Signal_end, :]) * self.split_test_size)))))
        random_KFold_test_data_end = random_KFold_test_data_begin + int(
            self.split_test_size * len(X_1[0:count_Signal_end, :]))
        print("split from: " + str(random_KFold_test_data_begin) + "   split to: " + str(random_KFold_test_data_end))
        training_data_1_idx = np.arange(0, random_KFold_test_data_begin)
        training_data_2_idx = np.arange(random_KFold_test_data_end, len(X_1[0:count_Signal_end, :]))
        training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
        test_data_idx = np.arange(random_KFold_test_data_begin, random_KFold_test_data_end)

        X_train = X_1[training_data_1_2_idx]
        Class_train = Class[training_data_1_2_idx]
        Y_train = Y_1[training_data_1_2_idx]
        X_Signal_number_dynamisch = X_Signal_number
        X_Signal_number_neu = X_Signal_number_dynamisch + 1
        X_Signal_number_train = np.vstack(
            (X_Signal_number_dynamisch[training_data_1_idx], X_Signal_number_neu[training_data_2_idx]))
        X_Signal_number_dynamisch = X_Signal_number_dynamisch + 1

        X_test = X_1[test_data_idx]
        Class_test = Class[test_data_idx]
        Y_test = Y_1[test_data_idx]
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
                    random.uniform((count_Signal_beginn + int(min_max_interval_split_data)), (count_Signal_beginn + int(
                        len(X_1[count_Signal_beginn:count_Signal_end, :]) - min_max_interval_split_data - int(
                            len(X_1[count_Signal_beginn:count_Signal_end, :]) * self.split_test_size)))))
                random_KFold_test_data_end = random_KFold_test_data_begin + int(
                    self.split_test_size * len(X_1[count_Signal_beginn:count_Signal_end, :]))
                print("split from: " + str(random_KFold_test_data_begin) + "   split to: " + str(
                    random_KFold_test_data_end))
                training_data_1_idx = np.arange(count_Signal_beginn, random_KFold_test_data_begin)
                training_data_2_idx = np.arange(random_KFold_test_data_end,
                                                count_Signal_beginn + len(X_1[count_Signal_beginn:count_Signal_end, :]))
                training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
                test_data_idx = np.arange(random_KFold_test_data_begin, random_KFold_test_data_end)

                X_train_tm = X_1[training_data_1_2_idx]
                Class_train_tm = Class[training_data_1_2_idx]
                Y_train_tm = Y_1[training_data_1_2_idx]
                X_Signal_number_neu = X_Signal_number_dynamisch + 1
                X_Signal_number_train_tm = np.vstack(
                    (X_Signal_number_dynamisch[training_data_1_idx], X_Signal_number_neu[training_data_2_idx]))
                X_Signal_number_dynamisch = X_Signal_number_dynamisch + 1
                X_test_tm = X_1[test_data_idx]
                Class_test_tm = Class[test_data_idx]
                Y_test_tm = Y_1[test_data_idx]
                X_Signal_number_test_tm = X_Signal_number[test_data_idx]
                X_train = np.vstack((X_train, X_train_tm))
                Class_train = np.vstack((Class_train, Class_train_tm))
                X_test = np.vstack((X_test, X_test_tm))
                Class_test = np.vstack((Class_test, Class_test_tm))
                Y_train = np.vstack((Y_train, Y_train_tm))
                Y_test = np.vstack((Y_test, Y_test_tm))
                X_Signal_number_train = np.vstack((X_Signal_number_train, X_Signal_number_train_tm))
                X_Signal_number_test = np.vstack((X_Signal_number_test, X_Signal_number_test_tm))
                count_Signal_beginn = count_Signal_end
        return X_train,X_test,Y_train,Y_test,X_Signal_number_train,X_Signal_number_test,Class_train,Class_test

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

    def create_training_test_data(self,X_1,Y_1,X_Signal_number,Class,i):

        if(i==1):
            count_Signal_beginn = 0
            count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == X_Signal_number[0,0])

            min_max_interval_split_data = self.creat_dataset_time_steps * 4
            if(min_max_interval_split_data*2<len(X_1[0:count_Signal_end, :])):

                # KFold random
                X_train,X_test,Y_train,Y_test,X_Signal_number_train,X_Signal_number_test,Class_train,Class_test = Tensorflow.create_training_test_data_random_KFold(self,X_1,Y_1,X_Signal_number,Class,count_Signal_beginn,count_Signal_end,min_max_interval_split_data)

                # KFold per steps (0.0-0.2, 0.2-0.4,....)
                #X_train, X_test, Y_train, Y_test, X_Signal_number_train, X_Signal_number_test, Class_train, Class_test = Tensorflow.create_training_test_data_KFold(self, X_1, Y_1, X_Signal_number, Class, count_Signal_beginn, count_Signal_end,min_max_interval_split_data)

                # old version (0.8-1) the last

                X_train, Y_train, X_Signal_number_train, Class_train = Tensorflow.cut_Signal_into_sequencen_train(self, X_train,Y_train,X_Signal_number_train,Class_train)
                X_test, Y_test, X_Signal_number_test, Class_test = Tensorflow.cut_Signal_into_sequencen_test(self, X_test, Y_test,X_Signal_number_test,Class_test)

                normalizer_x = self.scaler
                X_train_normalize = normalizer_x.fit_transform(X_train)
                X_test_normalize = normalizer_x.transform(X_test)
                X_train_time_series_normalize = pd.DataFrame(X_train_normalize)
                X_test_time_series_normalize = pd.DataFrame(X_test_normalize)

                normalizer_y = self.scaler
                #Y_train_normalize = normalizer_y.fit_transform(Y_train)
                #Y_test_normalize = normalizer_y.transform(Y_test)
                Y_train_time_series_normalize = pd.DataFrame(Y_train)
                Y_test_time_series_normalize = pd.DataFrame(Y_test)

        X_train_time_series, Y_train_time_series, Class_train = Tensorflow.shape_the_Signal(self=self,
                                                                               X_Signal_number_train=X_Signal_number_train,
                                                                               X_train_time_series_normalize=X_train_time_series_normalize,
                                                                               Y_train_time_series_normalize=Y_train_time_series_normalize,
                                                                               Class_train=Class_train,
                                                                               i=(i - 1))
        X_test_time_series, Y_test_time_series, Class_test = Tensorflow.shape_the_Signal(self=self,
                                                                             X_Signal_number_train=X_Signal_number_test,
                                                                             X_train_time_series_normalize=X_test_time_series_normalize,
                                                                             Y_train_time_series_normalize=Y_test_time_series_normalize,
                                                                             Class_train=Class_test,
                                                                             i=(i - 1))

        return X_train_time_series,Y_train_time_series,X_test_time_series,Y_test_time_series,normalizer_x,normalizer_y, Class_train,Class_test

    def plot_3D_Classification(self,model,X_train_time_series,Y_train_time_series,Class_train,i,name,path_NN_tm_loss):
        diff = i*4
        doku =[]
        #Y_pred = model.predict_classes(X_train_time_series)
        Y_pred = np.argmax(model.predict(X_train_time_series),axis=-1)
        tm_xyz_correct = []; tm_xyz_correct_no_intervention = []
        tm_xyz_false = []; tm_xyz_false_no_intervention = []
        count_no_intervention_true = 0; count_intervention_true =0
        count_no_intervention_false = 0; count_intervention_false=0
        for p in range(len(Y_pred)):
            #if(Y_train_time_series[p,0]==8000 and Y_train_time_series[p,1]==8000 and Y_train_time_series[p,2]==8000):
            if(Class_train[p]==0):
                if (Y_pred[p] == Class_train[p]):
                    tm_xyz_correct_no_intervention.append(Y_train_time_series[p])
                    count_no_intervention_true = count_no_intervention_true +1
                else:
                    tm_xyz_false_no_intervention.append(Y_train_time_series[p])
                    count_no_intervention_false = count_no_intervention_false +1
            else:
                if(Y_pred[p]==Class_train[p]):
                    tm_xyz_correct.append(Y_train_time_series[p])
                    count_intervention_true = count_intervention_true +1
                else:
                    tm_xyz_false.append(Y_train_time_series[p])
                    count_intervention_false = count_intervention_false +1
        print(name+"\n True Interventions = " + str(count_intervention_true))
        doku.append(name+" True Interventions = " + str(count_intervention_true))
        print(name+"\n True No Interventions = " + str(count_no_intervention_true))
        doku.append(name+" True No Interventions = " + str(count_no_intervention_true))
        print(name+"\n False Interventions = " + str(count_intervention_false))
        doku.append(name+" False Interventions = " + str(count_intervention_false))
        print(name+"\n False No Interventions = " + str(count_no_intervention_false))
        doku.append(name+" False No Interventions = " + str(count_no_intervention_false))
        fig = plt.figure(1+diff)
        ax = plt.axes(projection="3d")
        tm_xyz_correct=np.array(tm_xyz_correct)
        ax.scatter(tm_xyz_correct[:,0],tm_xyz_correct[:,2],tm_xyz_correct[:,1],c="blue")
        tm_xyz_false = np.array(tm_xyz_false)
        if(len(tm_xyz_false)!=0):
            ax.scatter(tm_xyz_false[:, 0], tm_xyz_false[:, 2], tm_xyz_false[:, 1], c="red")
        ax.set_title("True and False Prediction " + name)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if (self.png == True):
            plt.savefig(path_NN_tm_loss + "_"+name+"_True_False_Prediction.png")
        if (self.pickle == True):
            with open(path_NN_tm_loss + "_"+name+"_True_False_Prediction.pkl", "wb") as fp:  # Save Plots
                pickle.dump(fig, fp, protocol=4)
        fig = plt.figure(2+diff)
        ax = plt.axes(projection="3d")
        #tm_xyz_false = np.array(tm_xyz_false)
        if (len(tm_xyz_false) != 0):
            ax.scatter(tm_xyz_false[:,0],tm_xyz_false[:,2],tm_xyz_false[:,1], c="red")
        ax.set_title("False Prediction " + name)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if (self.png == True):
            plt.savefig(path_NN_tm_loss + "_"+name+"_False_Prediction.png")
        if (self.pickle == True):
            with open(path_NN_tm_loss + "_"+name+"_False_Prediction.pkl", "wb") as fp:  # Save Plots
                pickle.dump(fig, fp, protocol=4)
        fig = plt.figure(3+diff)
        ax = plt.axes(projection="3d")
        tm_xyz_correct_no_intervention = np.array(tm_xyz_correct_no_intervention)
        tm_xyz_false_no_intervention = np.array(tm_xyz_false_no_intervention)
        ax.scatter(tm_xyz_correct_no_intervention[:, 0], tm_xyz_correct_no_intervention[:, 2], tm_xyz_correct_no_intervention[:, 1], c="blue")
        if (len(tm_xyz_false_no_intervention) != 0):
            ax.scatter(tm_xyz_false_no_intervention[:, 0], tm_xyz_false_no_intervention[:, 2], tm_xyz_false_no_intervention[:, 1], c="red")
        ax.set_title("True and False No intervention " + name)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if (self.png == True):
            plt.savefig(path_NN_tm_loss + "_"+name+"_True_False_no_intervention.png")
        if (self.pickle == True):
            with open(path_NN_tm_loss + "_"+name+"_True_False_no_intervention.pkl", "wb") as fp:  # Save Plots
                pickle.dump(fig, fp, protocol=4)
        fig = plt.figure(4+diff)
        ax = plt.axes(projection="3d")
        if (len(tm_xyz_false_no_intervention) != 0):
            ax.scatter(tm_xyz_false_no_intervention[:, 0], tm_xyz_false_no_intervention[:, 2],tm_xyz_false_no_intervention[:, 1], c="red")
        ax.set_title("False No intervention " + name)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if (self.png == True):
            plt.savefig(path_NN_tm_loss + "_"+name+"_False_no_intervention.png")
        if (self.pickle == True):
            with open(path_NN_tm_loss + "_"+name+"_False_no_intervention.pkl", "wb") as fp:  # Save Plots
                pickle.dump(fig, fp, protocol=4)
        return doku

    def training_NN(self):
        time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(self.path+self.filename_training,delimiter=';')

        if(self.all_or_range_selection_Data==True):
            time = time_xyz_antennen_Signal_Komplex_all_Files[:,self.time]
            X_Signal_number=(time_xyz_antennen_Signal_Komplex_all_Files[:,self.X_Signal_number:(self.X_Signal_number+1)].astype(int))
            Class = (time_xyz_antennen_Signal_Komplex_all_Files[:,41:42].astype(int))
            X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,self.Input_from:self.Input_to]#[:,9:]#
            Y_1 = (time_xyz_antennen_Signal_Komplex_all_Files[:,self.Output_from:self.Output_to].astype(int))#[:,2:5]#
        else:
            time = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.time]
            X_Signal_number = (time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to,self.X_Signal_number:(self.X_Signal_number+1)].astype(int))
            X_1 = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.Input_from:self.Input_to]  # [:,9:]#
            Y_1 = (time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.Output_from:self.Output_to].astype(int) ) # [:,2:5]#

        time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz = np.loadtxt(
            self.path_test_sequenz + self.filename_test_sequenz, delimiter=';')
        time_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:, 0]

        X_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:, 7:40]
        Y_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:, 1:4]

        print("X_begin_shape =", X_1.shape)
        print("Y_begin_shape =", Y_1.shape)
        print("Class_begin_shape", Class.shape)

        for i in range(1, (self.time_steps_geo_folge + 1)):
            print("Time Steps: " + str(self.creat_dataset_time_steps * (2 ** (i - 1))))

            X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series, normalizer_x, normalizer_y, Class_train, Class_test = Tensorflow.create_training_test_data(
                self, X_1, Y_1, X_Signal_number,Class, i)

            print("X_Time_Series_train_shape =", X_train_time_series.shape)
            print("Y_Time_Series_train_shape =", Y_train_time_series.shape)
            print("Class_train_shape = ", Class_train.shape)

            model = tf.keras.Sequential([#
                tf.keras.layers.Conv1D(32,3, activation='relu',padding="same", input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),
                tf.keras.layers.Conv1D(32,3, activation='relu',padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(64, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.MaxPooling1D(2),
                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.SpatialDropout1D(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(7,activation="softmax")
            ])
            model.summary()
            time_stamp_from_NN =Tensorflow.get_actuall_time_stamp()
            doku = []
            for loss in range(0,len(self.loss_funktions)):
                print('\n\n' + self.loss_funktions[loss])
                model.compile(loss=self.loss_funktions[loss],
                              optimizer=self.optimizer,metrics=self.metrics)
                #,decay=1e-6,nesterov=True,momentum=0.9
                start_time_count = time_1.perf_counter()
                history = model.fit(X_train_time_series, Class_train, epochs=self.trainings_epochs,shuffle=True, verbose=self.verbose,batch_size=self.batch_size,validation_data=(X_test_time_series, Class_test)) #x=generator
                elapsed_time = time_1.perf_counter() -start_time_count
                path_NN = os.path.join(self.path + self.save_dir)
                if (False == os.path.isdir(path_NN)):
                    os.mkdir(path_NN)
                path_NN_tm = os.path.join(path_NN+'\\NN_from_'+time_stamp_from_NN)
                if(False==os.path.isdir(path_NN_tm)):
                    os.mkdir(path_NN_tm)
                path_NN_tm_loss = os.path.join(path_NN_tm+'\\'+self.loss_funktions[loss]+'_Time_Step_'+str(self.creat_dataset_time_steps*(2**(i-1))))
                if (False == os.path.isdir(path_NN_tm_loss)):
                    os.mkdir(path_NN_tm_loss)
                path_NN_tm_loss= os.path.join(path_NN_tm_loss+'\\' + self.loss_funktions[loss])
                try:
                    model.save(os.path.join(path_NN_tm_loss+".h5"))
                except:
                    print("The model not saved")
                result_test =model.evaluate(X_test_time_series,Class_test)
                print("Evaluate: " + str(np.array(result_test)))
                Y_pred_time_series =model.predict(X_test_time_series)#[:4000]
                #Y_pred_time_series = normalizer_y.inverse_transform(Y_pred_time_series)
                #Y_test_time_series = normalizer_y.inverse_transform(Y_test_time_series)#[:4000]

                doku_training =Tensorflow.plot_3D_Classification(self, model, X_train_time_series, Y_train_time_series, Class_train, 0,
                                                  "Training", path_NN_tm_loss)
                doku_test = Tensorflow.plot_3D_Classification(self, model, X_test_time_series, Y_test_time_series, Class_test, 1,
                                                  "Test", path_NN_tm_loss)


                #result_train =
                print('\n------------------------------------------------------------------------------------------------\n')
                print('\n\n' + self.loss_funktions[loss])
                #print('Loss and accuracy Training:',history.history['loss'][-1],history.history["sparse_categorical_crossentropy"][-1])
                #print('Loss and accuracy val_Training:',history.history['val_loss'][-1],history.history['val_sparse_categorical_crossentropy'][-1])
                print('Training Time: %.3f seconds.' % elapsed_time)
                result_train = model.evaluate(X_train_time_series,Class_train,batch_size=self.batch_size)
                print("Evaluate Training: " + str(np.array(result_train)))
                #print("loss (test-set):", str(result))
                #print('loss test_sequenz:', str(model.evaluate(x=np.expand_dims(X_test_time_series, axis=0),y=np.expand_dims(Y_test_time_series, axis=0))))
                print('\n------------------------------------------------------------------------------------------------\n')
                if (loss == 0):
                    doku.append("Model:")
                    doku.append('\n' + str(model.inputs))
                    for j in range(0, len(model._layers)):
                        if (str(model._layers[j].output.op.inputs) != "()"):
                            doku.append('\n' + str(model._layers[j].output.op.inputs))
                    doku.append('\n')
                doku.append('\n------------------------------------------------------------------------------------------------')
                doku.append("\nX_Time_Series_generator_shape ="+str(X_train_time_series.shape)+"\nY_Time_Series_generator_shape ="+str(Y_train_time_series.shape))
                doku.append("\nTime Steps: "+str(self.creat_dataset_time_steps*(2**(i-1))))
                doku.append("\nScaler: "+str(self.scaler))
                doku.append('\n' + self.loss_funktions[loss]+ " Loss, " + str(self.trainings_epochs) + " Epochs, " + str(
                    self.learning_rate) + " Learning Rate, " + str(self.optimizer_name) + " Optimizer, " + str(
                    self.batch_size) + " Batch Size, " + str(self.metrics) + " Metrics")
                #doku.append('\n'+self.loss_funktions[loss]+', Loss Training: '+str(history.history['loss'][-1])+'  Train Metrics'+str(self.metrics)+':'+str(history.history['sparse_categorical_crossentropy'][-1]))
                #doku.append('\n'+self.loss_funktions[loss]+', Loss Validation: '+str(history.history['val_loss'][-1])+'  Validation Metrics'+str(self.metrics)+':'+str(history.history['val_sparse_categorical_crossentropy'][-1]))
                doku.append('\nTraining Time: %.3f seconds.' % elapsed_time)
                doku.append("\n")
                doku.append(doku_training)
                doku.append("\n")
                doku.append(doku_test)
                doku.append("\nloss (Training-set): " +str(result_train))
                doku.append("\nloss (Test-set): " + str(result_test))

                #doku.append('loss test_sequenz:'+str(model.evaluate(x=np.expand_dims(X_test_time_series, axis=0),y=np.expand_dims(Y_test_time_series, axis=0))))
                doku.append('\n------------------------------------------------------------------------------------------------\n')



            open_path=os.path.join(self.path + self.save_dir+ '\\NN_from_'+time_stamp_from_NN+ '\\Doku_Time_Step_'+str(self.creat_dataset_time_steps*(2**(i-1))))
            file = open(open_path+'.txt', 'w')
            for i in range(0,len(doku)):
                file.write(str(doku[i]))
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