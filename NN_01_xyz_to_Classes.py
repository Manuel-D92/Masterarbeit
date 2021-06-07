from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
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
from tensorboard.plugins.hparams import api as hp
import pylab as pl
from keras import regularizers
from tcn import TCN
import seaborn as sns
#import tensorflow_probability as tf


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class config_Test_Tensorflow():
    # self.a = outside.
    # Time Steps (create_dataset)
    creat_dataset_time_steps = 16
    time_steps_geo_folge = 1  # geometrische-Folge(zweier Potenzen) for time steps, example: (time_steps=16 andtime_steps_geo_folge=2)->  1
    create_dataset_overlap = 15 #15  # example time_steps=16(0-15); Overlap from Feature to Feature example: 0= no overlap, 15= 0-15,1-16
    create_dataset_offset = 7 # example time_steps=16(0-15)  the middle for y is = offset 7-8 (from 0-15) #0 first value, 15 last value

    #### Trainingsdata into Trainingssequence (Only No Cut Files!!) (with shape)
    sequence_training = True
    sequence_training_cut_x = [-5000, 16000]
    sequence_training_cut_y = [-5000, 13000]
    sequence_training_cut_z = [-400,100]

    #### Testdata into Trainingssequence
    sequence_test = True
    sequence_test_cut_x = [-5000, 16000]
    sequence_test_cut_y = [-5000, 13000]
    sequence_test_cut_z = [-400, 100]

    # Path
    path = r"C:\Users\dauserml\Documents\2020_11_16_und_2020_12_03"
    filename_training = "\\01_Manuel_03_12_2020.csv"#"\\all_Files_C_R.csv"



    path_test_sequenz = r"C:\Users\dauserml\Documents\2020_11_16"
    filename_test_sequenz = "\\2020_11_16_files_Z_800.csv"

    save_dir = "\\xyz_to_Classes_Manuel_Training"#'\\VGG_Time_Series_Classification_Sequence_teil_1'
    save=False #True -> save Data
    # self.save_dir='\\Neuronale_Netze_Time_Series_Generator_nur_mit_X'

    # Trainings Daten
    all_or_range_selection_Data = True  # all=True, range selection=False
    training_Data_from = 0  # Only "all_or_range_selection_Data = False"
    training_Data_to = 20000  # Only "all_or_range_selection_Data = False"
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
    learning_rate = 0.0001 # 2e-6=0.000 002
    optimizer_name = "Adam"
    opti_name = ["Adam","Nadam","SGD with Nesterov","RMSprop","Adadelta"]
    #optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)#,momentum=0.9,nesterov=True,decay=1e-6)
    optimizer = [tf.keras.optimizers.Adam(learning_rate=learning_rate)]#,tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                 #tf.keras.optimizers.SGD(learning_rate=learning_rate,nesterov=True), tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                 #tf.keras.optimizers.Adadelta(learning_rate=learning_rate)]
    ## Split Options
    split_test_size = 0.2
    split_KFold_set = 2 # bei 0.2 -> 5 Sets (0->0.0-0.2, 1->0.2-0.4, 2 -> 0.4-0.6 ...... 4 -> 0.8-1.0)
    split_random_state = 42
    split_shuffle = False
#
    ## Loss Options
    # self.loss_funktions= ['MSE','huber_loss'] #,'MSE','huber_loss','MAP' Probabilistic lossfunktion (Bayesian) lambda y, p_y: -p_y.log_prob(y)   #
    loss_funktions = ['sparse_categorical_crossentropy'] # ,'huber_loss',
    # 'logcosh', 'MAPE',
    # 'MSLE']  # ,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'

    ## Scaler Options
    scaler = preprocessing.Normalizer()

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
                 optimizer,split_test_size,split_KFold_set,split_random_state,split_shuffle,loss_funktions,scaler,threshold_antenna,threshold_value,pickle,png,kernel_init,bias_init,save):
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
        self.save=save
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
            return Xs, np.array(ys), np.array(Cl)
        else:
            print("offset or overlap to high!!")

    def cut_Signal_into_sequencen_train_and_val(self,X_train,Y_train,X_Signal_number_train,Class_train):
        if (self.sequence_training == True):
            print("Training")
            Class_train = Class_train +2
            X_train_cut_tm = [];X_train_cut_NaN = [];
            Class_train_tm =[]; Class_train_cut_NaN =[];
            Y_train_cut_tm = [];Y_train_cut_NaN = [];
            X_Signal_number_train_cut_tm = []; X_Signal_number_train_cut_NaN=[]
            count_sequence=0
            flag = 0
            ze=0
            count_out=0
            flag=0
            for c in range(0, len(Y_train)):
                if (self.sequence_training_cut_x[0] < Y_train[c, 0] < self.sequence_training_cut_x[1]
                        and self.sequence_training_cut_y[0] < Y_train[c, 1] < self.sequence_training_cut_y[1]):
                    zb = 0
                    false_more_peaks=False
                    if ((c < ze) == False or (ze == 0) == True):
                        if (self.sequence_training_cut_z[1] > Y_train[c, 2]):
                            while ((self.sequence_training_cut_z[1] < Y_train[c + zb, 2]) == False):
                                zb = zb + 1
                                if (len(Y_train) == c + zb):
                                    break
                            ze = c + zb
                            z_interval= Y_train[c:ze,2]
                            z_minimum_position = int(np.argmin(z_interval))
                            z_interval_left = z_interval[0:z_minimum_position]
                            z_interval_right = z_interval[int(z_minimum_position) + 1:(len(z_interval) - 1)]
                            if(len(z_interval_left)>=1 and len(z_interval_right)>=1):
                                if(len(z_interval_left)>self.creat_dataset_time_steps and len(z_interval_right)>self.creat_dataset_time_steps):
                                    for zdown in range(0,len(z_interval_left)-2):
                                        if((Y_train[c + zdown, 2] < Y_train[c + zdown+1, 2])):
                                            false_more_peaks=True
                                            print("Training sequence false, left side")
                                    for zup in range(0,len(z_interval_right)-1):
                                        if (len(Y_train) >= (ze + len(z_interval_right) - 1)):
                                            if ((Y_train[ze, 2] < Y_train[ze - 1 - zup, 2])):
                                                false_more_peaks = True
                                                print("Test sequence false, right side")
                                        else:
                                            false_more_peaks = True
                                            print("Test sequence false, right side (end Signal)")
                            else:
                                print("Training sequence false!")
                                false_more_peaks = True

                        if(zb>=self.creat_dataset_time_steps*2+10  and false_more_peaks==False): # whether Sequence is big enough
                            begin_offset = (self.creat_dataset_time_steps - self.create_dataset_offset - 1)
                            end_offset = (self.create_dataset_offset)
                            if(c>=self.creat_dataset_time_steps):
                                in_z_begin = c - self.creat_dataset_time_steps - begin_offset + 1
                                out_z_begin = ze - self.creat_dataset_time_steps - begin_offset
                                class_info_z_begin = c + self.creat_dataset_time_steps + 1 - begin_offset
                                class_info_z_end = ze - self.creat_dataset_time_steps + end_offset
                                sequence_end = True
                                for beg in range (0,self.creat_dataset_time_steps):
                                    if(Y_train[c-beg,2]>int(self.sequence_training_cut_z[1]*4)):
                                        sequence_end = False
                                for end in range (0,self.creat_dataset_time_steps):
                                    if(len(Y_train)>ze+end):
                                        if(Y_train[ze+end,2]>int(self.sequence_training_cut_z[1]*4)):
                                            sequence_end = False
                                    else:
                                        sequence_end = False

                                if(sequence_end == True):
                                    print("Good Training Sequence ;)")
                                    if (count_sequence == 0):
                                        for z_out_count in range(0, count_out - self.creat_dataset_time_steps * 2 + (
                                                self.creat_dataset_time_steps - self.create_dataset_offset)):
                                            X_train_cut_tm.append(X_train[c - count_out + z_out_count, :])
                                            class_train_tm = np.array([0])
                                            Class_train_tm.append(class_train_tm)
                                            Y_train_cut_tm.append(Y_train[c - count_out + z_out_count, :])
                                            X_Signal_number_train_cut_tm.append(
                                                X_Signal_number_train[c - count_out + z_out_count])
                                        count_out = 0
                                    else:
                                        for z_out_count in range(0, count_out - self.creat_dataset_time_steps * 2 + 1):
                                            X_train_cut_tm.append(X_train[c + self.creat_dataset_time_steps - (
                                                        self.creat_dataset_time_steps - self.create_dataset_offset) + 1 - count_out + z_out_count,
                                                                 :])
                                            class_train_tm = np.array([0])
                                            Class_train_tm.append(class_train_tm)
                                            Y_train_cut_tm.append(Y_train[c + self.creat_dataset_time_steps - (
                                                        self.creat_dataset_time_steps - self.create_dataset_offset) + 1 - count_out + z_out_count,
                                                                 :])
                                            X_Signal_number_train_cut_tm.append(X_Signal_number_train[
                                                                                   c + self.creat_dataset_time_steps - (
                                                                                               self.creat_dataset_time_steps + 1 - self.create_dataset_offset) - count_out + z_out_count])
                                        count_out = 0
                                    count_sequence = count_sequence+1
                                    for bz in range(0,int(self.creat_dataset_time_steps*2)+self.creat_dataset_time_steps-1):  #Class 1 in shelf #
                                        X_train_cut_tm.append(X_train[in_z_begin+bz,:])
                                        class_tm = np.array([1])
                                        Class_train_tm.append(class_tm)
                                        Y_train_cut_tm.append(Y_train[in_z_begin+bz,:])
                                        X_Signal_number_train_cut_tm.append(X_Signal_number_train[in_z_begin+bz,:])
                                    X_Signal_number_train = X_Signal_number_train +1
                                    for cz in range(0,(class_info_z_end-class_info_z_begin)): #Class Intervention 3-8
                                        X_train_cut_tm.append(X_train[class_info_z_begin+cz, :])
                                        Class_train_tm.append(Class_train[class_info_z_begin+cz])#+self.creat_dataset_time_steps
                                        Y_train_cut_tm.append(Y_train[class_info_z_begin+cz, :])
                                        X_Signal_number_train_cut_tm.append(X_Signal_number_train[class_info_z_begin+cz, :])
                                    X_Signal_number_train = X_Signal_number_train + 1
                                    for ez in range(0,int(self.creat_dataset_time_steps*2)+self.creat_dataset_time_steps-1): #Class 2 out shelf #
                                        X_train_cut_tm.append(X_train[out_z_begin+ez, :])
                                        class_tm=np.array([2])
                                        Class_train_tm.append(class_tm)
                                        Y_train_cut_tm.append(Y_train[out_z_begin+ez, :])
                                        X_Signal_number_train_cut_tm.append(X_Signal_number_train[out_z_begin+ez, :])
                                    X_Signal_number_train = X_Signal_number_train + 1
                                else:
                                    print("Training Sequence not good!")
                        elif(zb>=self.creat_dataset_time_steps*2+10):
                            count_out = 0
                    #flag = 1
                elif (flag == 1):
                    X_Signal_number_train = X_Signal_number_train + 1
                    ## No intervention Training
                    X_train_cut_NaN.append(X_train[c, :])
                    class_tm = np.array([0])
                    Class_train_cut_NaN.append(class_tm)
                    Y_train_cut_NaN.append(Y_train[c, :])
                    X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c,:])
                    flag = 0
                else:
                    #X_Signal_number_train = X_Signal_number_train + 1
                    ## No intervention Training
                    X_train_cut_NaN.append(X_train[c, :])
                    class_tm = np.array([0])
                    Class_train_cut_NaN.append(class_tm)
                    Y_train_cut_NaN.append(Y_train[c, :])
                    X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c, :])

            print("Training Sequence Value: " + str(count_sequence))
            #Y_test_cut_tm_sequence = np.array(Y_train_cut_tm)
            #plt.plot(Y_test_cut_tm_sequence[:, 2])
            #plt.show()


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

    def cut_Signal_into_sequencen_test_or_train(self,X_test,Y_test,X_Signal_number_test,Class_test,test_or_train):
        if (test_or_train == True):
            print("Test")
            test_or_training = "Test "
        else:
            print("Training")
            test_or_training = "Training "
        Class_test = Class_test + 2
        X_test_cut_tm,Class_test_tm,Y_test_cut_tm,X_Signal_number_test_cut_tm = [],[],[],[]
        count_sequence = 0
        ze = 0
        count_out=0
        flag=0
        for c in range(0, len(Y_test)):
            if (self.sequence_test_cut_x[0] < Y_test[c, 0] < self.sequence_test_cut_x[1]
                    and self.sequence_test_cut_y[0] < Y_test[c, 1] < self.sequence_test_cut_y[1]):
                zb = 0
                false_more_peaks = False
                if ((c < ze) == False or (ze == 0) == True):
                    if (self.sequence_test_cut_z[1] > Y_test[c, 2]):
                        while ((self.sequence_training_cut_z[1] > Y_test[c + zb, 2])):
                            zb = zb + 1
                            if (len(Y_test) == c + zb):
                                break
                        ze = c + zb
                        z_interval = Y_test[c:ze, 2]
                        z_minimum_position = int(np.argmin(z_interval))
                        z_interval_left = z_interval[0:z_minimum_position]
                        z_interval_right = z_interval[int(z_minimum_position) + 1:(len(z_interval) - 1)]
                        if (len(z_interval_left) >= 1 and len(z_interval_right) >= 1):
                            if (len(z_interval_left) > self.creat_dataset_time_steps and len(
                                    z_interval_right) > self.creat_dataset_time_steps):
                                for zdown in range(0, len(z_interval_left) - 2):
                                    if ((Y_test[c + zdown, 2] < Y_test[c + zdown + 1, 2])):
                                        false_more_peaks = True
                                        print(test_or_training+"sequence false, left side")
                                for zup in range(0, len(z_interval_right) - 1):
                                    if(len(Y_test)>= (ze+len(z_interval_right) - 1)):
                                        if ((Y_test[ze, 2] < Y_test[ze - 1 - zup, 2])):
                                            false_more_peaks = True
                                            print(test_or_training+"sequence false, right side")
                                    else:
                                        false_more_peaks = True
                                        print(test_or_training+"sequence false, right side (end Signal)")
                        else:
                            print(test_or_training+"sequence false")
                            false_more_peaks = True
                    if (zb >= self.creat_dataset_time_steps * 2 + 10 and false_more_peaks==False):  # whether Sequence is big enough
                        begin_offset = (self.creat_dataset_time_steps - self.create_dataset_offset - 1)
                        end_offset = (self.create_dataset_offset)
                        if (c >= self.creat_dataset_time_steps):
                            in_z_begin = c - self.creat_dataset_time_steps - begin_offset + 1
                            out_z_begin = ze - self.creat_dataset_time_steps - begin_offset
                            class_info_z_begin = c + self.creat_dataset_time_steps + 1 - begin_offset
                            class_info_z_end = ze - self.creat_dataset_time_steps + end_offset
                            sequence_end = True
                            for beg in range(0, self.creat_dataset_time_steps):
                                if (Y_test[c - beg, 2] > int(self.sequence_test_cut_z[1] * 4)):
                                    sequence_end = False
                            for end in range(0, self.creat_dataset_time_steps):
                                if (len(Y_test) > ze + end):
                                    if (Y_test[ze + end, 2] > int(self.sequence_test_cut_z[1] * 4)):
                                        sequence_end = False
                                else:
                                    sequence_end = False
                            if (sequence_end == True):
                                if (count_out > (self.creat_dataset_time_steps+(self.creat_dataset_time_steps-self.create_dataset_offset)) ):
                                    if(count_sequence==0):
                                        for z_out_count in range(0,count_out - self.creat_dataset_time_steps*2+(self.creat_dataset_time_steps-self.create_dataset_offset) ):
                                            X_test_cut_tm.append(X_test[c - count_out  + z_out_count,:])
                                            class_test_tm = np.array([0])
                                            Class_test_tm.append(class_test_tm)
                                            Y_test_cut_tm.append(Y_test[c - count_out + z_out_count,:])
                                            X_Signal_number_test_cut_tm.append(X_Signal_number_test[c - count_out + z_out_count])
                                        count_out = 0
                                    else:
                                        for z_out_count in range(0,count_out - self.creat_dataset_time_steps*2 + 1):
                                            X_test_cut_tm.append(X_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps - self.create_dataset_offset) + 1 - count_out + z_out_count,:])
                                            class_test_tm = np.array([0])
                                            Class_test_tm.append(class_test_tm)
                                            Y_test_cut_tm.append(Y_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps - self.create_dataset_offset) + 1 - count_out + z_out_count,:])
                                            X_Signal_number_test_cut_tm.append(X_Signal_number_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps + 1 - self.create_dataset_offset) - count_out + z_out_count])
                                        count_out = 0
                                print("Good "+test_or_training+"Sequence ;)")
                                count_sequence = count_sequence + 1
                                for bz in range(0, int(self.creat_dataset_time_steps * 2 )+self.creat_dataset_time_steps-1):  # Class 1 in shelf #
                                    X_test_cut_tm.append(X_test[in_z_begin + bz, :])
                                    class_tm = np.array([1])
                                    Class_test_tm.append(class_tm)  # Class_train[in_z_begin+bz]
                                    Y_test_cut_tm.append(Y_test[in_z_begin + bz, :])
                                    X_Signal_number_test_cut_tm.append(X_Signal_number_test[in_z_begin + bz, :])
                                X_Signal_number_test = X_Signal_number_test + 1
                                for cz in range(0, (class_info_z_end - class_info_z_begin)):  # Class Intervention 3-8
                                    X_test_cut_tm.append(X_test[class_info_z_begin + cz, :])
                                    Class_test_tm.append(Class_test[class_info_z_begin + cz])
                                    Y_test_cut_tm.append(Y_test[class_info_z_begin + cz, :])
                                    X_Signal_number_test_cut_tm.append(X_Signal_number_test[class_info_z_begin + cz, :])
                                X_Signal_number_test = X_Signal_number_test + 1
                                for ez in range(0, int(self.creat_dataset_time_steps * 2)+self.creat_dataset_time_steps-1):  # Class 2 out shelf
                                    X_test_cut_tm.append(X_test[out_z_begin + ez, :])
                                    class_tm = np.array([2])
                                    Class_test_tm.append(class_tm)  # Class_train[in_z_begin+bz]
                                    Y_test_cut_tm.append(Y_test[out_z_begin + ez, :])
                                    X_Signal_number_test_cut_tm.append(X_Signal_number_test[out_z_begin + ez, :])
                                X_Signal_number_test = X_Signal_number_test + 1
                            else:
                                count_out = count_out + 1
                                print(test_or_training+"Sequence not good!")
                        else:
                            count_out = count_out + 1
                    else:
                        count_out = count_out+1
                elif((ze == 0) == True):
                    count_out = count_out + 1
            else:
                if(flag==0):
                    for z_out_count in range(0, count_out - self.create_dataset_offset-1):
                        X_test_cut_tm.append(X_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps - self.create_dataset_offset) + 1 - count_out + z_out_count,:])
                        class_test_tm = np.array([0])
                        Class_test_tm.append(class_test_tm)
                        Y_test_cut_tm.append(Y_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps - self.create_dataset_offset) + 1 - count_out + z_out_count,:])
                        X_Signal_number_test_cut_tm.append(X_Signal_number_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps + 1 - self.create_dataset_offset) - count_out + z_out_count])
                        flag=1
                else:
                    for z_out_count in range(0, count_out):
                        X_test_cut_tm.append(X_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps - self.create_dataset_offset) + 1 - count_out + z_out_count,:])
                        class_test_tm = np.array([0])
                        Class_test_tm.append(class_test_tm)
                        Y_test_cut_tm.append(Y_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps - self.create_dataset_offset) + 1 - count_out + z_out_count,:])
                        X_Signal_number_test_cut_tm.append(X_Signal_number_test[c + self.creat_dataset_time_steps - (self.creat_dataset_time_steps + 1 - self.create_dataset_offset) - count_out + z_out_count])
                X_test_cut_tm.append(X_test[c, :])
                class_test_tm = np.array([0])
                Class_test_tm.append(class_test_tm)
                Y_test_cut_tm.append(Y_test[c, :])
                X_Signal_number_test_cut_tm.append(X_Signal_number_test[c, :])
                count_out = 0

        for z_out_count in range(0, count_out):
                X_test_cut_tm.append(X_test[c - count_out + z_out_count, :])
                class_test_tm = np.array([0])
                Class_test_tm.append(class_test_tm)
                Y_test_cut_tm.append(Y_test[c  - count_out + z_out_count, :])
                X_Signal_number_test_cut_tm.append(X_Signal_number_test[c - count_out + z_out_count])

        print(test_or_training+"Sequence Value: " + str(count_sequence))
        #Y_test_cut_tm_sequence= np.array(Y_test_cut_tm)
        #plt.plot(Y_test_cut_tm_sequence[:,2])
        #plt.show()

        Class_test = np.array(Class_test_tm)
        X_Signal_number_test = np.array(X_Signal_number_test_cut_tm)
        X_test = np.array(X_test_cut_tm)
        Y_test = np.array(Y_test_cut_tm)
        return X_test,Y_test,X_Signal_number_test,Class_test

    def shape_the_Signal(self,X_Signal_number_train,X_train_time_series_normalize,Y_train_time_series_normalize,Class_train,i):
        count = 0
        X_train_time_series = []
        Class=[]
        Y_train_time_series = []
        first_data = 0
        old_y_values=[] #for overlap
        for j in range(int(X_Signal_number_train[0]),int(X_Signal_number_train[-1] + 1)): #int(max(X_Signal_number_train))
            of = 0 + count
            count = count + np.count_nonzero(X_Signal_number_train[:] == j)
            to = count
            #X_train_time_series_normalize_tet = np.ndarray(X_train_time_series_normalize)
            if(self.creat_dataset_time_steps*(2**i)<np.count_nonzero(X_Signal_number_train[:] == j)):
                X_train_time_series_tm, Y_train_time_series_tm,Class_train_tm = Tensorflow.create_dataset(X_train_time_series_normalize[of:to][:],
                                                                                           Y_train_time_series_normalize[of:to][:],
                                                                                           Class_train[of:to],
                                                                                           time_steps=self.creat_dataset_time_steps * (2 ** i),
                                                                                           overlap=self.create_dataset_overlap,
                                                                                           offset=self.create_dataset_offset)
                #if(len(old_y_values)==0):
                #    old_y_values=Y_train_time_series_tm[:,-1]
                #else:
                #    if(old_y_values[0]==Y_train_time_series_tm[0,-1] and old_y_values[1]==Y_train_time_series_tm[1,-1] and old_y_values[2]==Y_train_time_series_tm[2,-1]):
                #        np.delete(X_train_time_series_tm[:,-1])
                #        np.delete(Y_train_time_series_tm[:,-1])
                #        np.delete(Class_train_tm[-1])
                #    old_y_values = Y_train_time_series_tm[:, -1]

                if (len(X_train_time_series_tm) != 0 and len(X_train_time_series) == 0):
                    X_train_time_series = X_train_time_series_tm
                    Class = Class_train_tm
                    Y_train_time_series = Y_train_time_series_tm
                elif (len(X_train_time_series) != 0 and len(X_train_time_series_tm) != 0):
                    X_train_time_series = np.vstack((X_train_time_series, X_train_time_series_tm))
                    Class = np.vstack((Class, Class_train_tm))
                    Y_train_time_series = np.vstack((Y_train_time_series, Y_train_time_series_tm))
        return X_train_time_series, Y_train_time_series, Class

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
        training_data_1_idx = np.arange(0, random_KFold_test_data_begin + self.create_dataset_offset)
        training_data_2_idx = np.arange(random_KFold_test_data_end-(self.creat_dataset_time_steps-self.create_dataset_offset), len(X_1[0:count_Signal_end, :])+ self.create_dataset_offset)
        training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
        test_data_idx = np.arange(random_KFold_test_data_begin -(self.creat_dataset_time_steps-self.create_dataset_offset) , random_KFold_test_data_end + self.create_dataset_offset)

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
                training_data_1_idx = np.arange(count_Signal_beginn, random_KFold_test_data_begin+self.create_dataset_offset)
                if(last_Signal_number[0] + 1 >= count_Signal_beginn + len(X_1[count_Signal_beginn:count_Signal_end,:])+self.create_dataset_offset):
                    training_data_2_idx = np.arange(random_KFold_test_data_end-(self.creat_dataset_time_steps-self.create_dataset_offset),
                                                    count_Signal_beginn + len(X_1[count_Signal_beginn:count_Signal_end,:])+self.create_dataset_offset)
                else:
                    training_data_2_idx = np.arange(random_KFold_test_data_end-(self.creat_dataset_time_steps-self.create_dataset_offset),
                                                    count_Signal_beginn + len(X_1[count_Signal_beginn:count_Signal_end, :]))
                training_data_1_2_idx = np.hstack((training_data_1_idx, training_data_2_idx))
                test_data_idx = np.arange(random_KFold_test_data_begin-(self.creat_dataset_time_steps-self.create_dataset_offset) , random_KFold_test_data_end+ self.create_dataset_offset)

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

    def create_training_test_data_leave_one_out(self,X_1,Y_1,X_Signal_number,Class,count_Signal_beginn,count_Signal_end,min_max_interval_split_data):
        loo = LeaveOneGroupOut()
        #print(str(loo.get_n_splits(X_1)))
        for train_index, test_index in loo.split(X_1,Y_1,Y_1):
            X_train, X_test = X_1[train_index,:], X_1[test_index,:]
            Y_train, Y_test = Y_1[train_index,:], Y_1[test_index,:]
            X_Signal_number_train, X_Signal_number_test = X_Signal_number[train_index], X_Signal_number[test_index]
            Class_train,Class_test = Class[train_index],Class[test_index]
        return X_train,X_test,Y_train,Y_test,X_Signal_number_train,X_Signal_number_test,Class_train,Class_test

    def create_training_test_data(self,X_1,Y_1,X_Signal_number,Class,i):

        if(i==1):
            count_Signal_beginn = 0
            count_Signal_end = count_Signal_beginn + np.count_nonzero(X_Signal_number == X_Signal_number[0,0])

            min_max_interval_split_data = self.creat_dataset_time_steps * 4
            if(min_max_interval_split_data*2<len(X_1[0:count_Signal_end, :])):

                # KFold random
                X_train,X_test,Y_train,Y_test,X_Signal_number_train,X_Signal_number_test,Class_train,Class_test = Tensorflow.create_training_test_data_random_KFold(self,X_1,Y_1,X_Signal_number,Class,count_Signal_beginn,count_Signal_end,min_max_interval_split_data)

                #leave_one_out
                #X_train, X_test, Y_train, Y_test, X_Signal_number_train, X_Signal_number_test, Class_train, Class_test = Tensorflow.create_training_test_data_leave_one_out(
                #    self, X_1, Y_1, X_Signal_number, Class, count_Signal_beginn, count_Signal_end,
                #    min_max_interval_split_data)

                # KFold per steps (0.0-0.2, 0.2-0.4,....)
                #X_train, X_test, Y_train, Y_test, X_Signal_number_train, X_Signal_number_test, Class_train, Class_test = Tensorflow.create_training_test_data_KFold(self, X_1, Y_1, X_Signal_number, Class, count_Signal_beginn, count_Signal_end,min_max_interval_split_data)

                # old version (0.8-1) the last

                #X_train, Y_train, X_Signal_number_train, Class_train = Tensorflow.cut_Signal_into_sequencen_train(self, X_train,Y_train,X_Signal_number_train,Class_train)
                X_train, Y_train, X_Signal_number_train, Class_train = Tensorflow.cut_Signal_into_sequencen_test_or_train(self,X_train,Y_train,X_Signal_number_train,Class_train,False)
                X_test, Y_test, X_Signal_number_test, Class_test = Tensorflow.cut_Signal_into_sequencen_test_or_train(self, X_test, Y_test,X_Signal_number_test,Class_test,True)

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
        tm_in_correct=[];tm_out_correct=[]
        tm_in_false=[];tm_out_false=[]
        tm_xyz_false = []; tm_xyz_false_no_intervention = []
        count_no_intervention_true = 0; count_intervention_true =0
        count_no_intervention_false = 0; count_intervention_false=0
        count_in_true=0;count_out_true=0
        count_in_false=0;count_out_false=0
        for p in range(len(Y_pred)):
            #if(Y_train_time_series[p,0]==8000 and Y_train_time_series[p,1]==8000 and Y_train_time_series[p,2]==8000):
            if(Class_train[p]==0):
                if (Y_pred[p] == Class_train[p]):
                    tm_xyz_correct_no_intervention.append(Y_train_time_series[p])
                    count_no_intervention_true = count_no_intervention_true +1
                else:
                    tm_xyz_false_no_intervention.append(Y_train_time_series[p])
                    count_no_intervention_false = count_no_intervention_false +1
            elif(Class_train[p]==1):
                if (Y_pred[p] == Class_train[p]):
                    tm_in_correct.append(Y_train_time_series[p])
                    count_in_true = count_in_true +1
                else:
                    tm_in_false.append(Y_train_time_series[p])
                    count_in_false = count_in_false +1
            elif (Class_train[p] == 2):
                if (Y_pred[p] == Class_train[p]):
                    tm_out_correct.append(Y_train_time_series[p])
                    count_out_true = count_out_true +1
                else:
                    tm_out_false.append(Y_train_time_series[p])
                    count_out_false = count_out_false +1
            else:
                if(Y_pred[p]==Class_train[p]):
                    tm_xyz_correct.append(Y_train_time_series[p])
                    count_intervention_true = count_intervention_true +1
                else:
                    tm_xyz_false.append(Y_train_time_series[p])
                    count_intervention_false = count_intervention_false +1
        print(name+"\n True Interventions = " + str(count_intervention_true))
        doku.append(name+" True Interventions = " + str(count_intervention_true))
        print(name+"\n True In = " + str(count_in_true))
        doku.append(name+"\n True In = " + str(count_in_true))
        print(name+"\n True Out = " + str(count_out_true))
        doku.append(name+"\n True Out = " + str(count_out_true))
        print(name+"\n True No Interventions = " + str(count_no_intervention_true))
        doku.append(name+" True No Interventions = " + str(count_no_intervention_true))
        print(name+"\n False Interventions = " + str(count_intervention_false))
        doku.append(name+" False Interventions = " + str(count_intervention_false))
        print(name+"\n False In = " + str(count_in_false))
        doku.append(name+"\n False In = " + str(count_in_false))
        print(name+"\n False Out = " + str(count_out_false))
        doku.append(name+"\n False Out = " + str(count_out_false))
        print(name+"\n False No Interventions = " + str(count_no_intervention_false))
        doku.append(name+" False No Interventions = " + str(count_no_intervention_false))
        fig = plt.figure(1+diff)
        ax = plt.axes(projection="3d")
        # Arrays
        tm_xyz_correct=np.array(tm_xyz_correct)
        tm_in_correct=np.array(tm_in_correct)
        tm_out_correct=np.array(tm_out_correct)
        tm_xyz_correct_no_intervention = np.array(tm_xyz_correct_no_intervention)
        tm_xyz_false = np.array(tm_xyz_false)
        tm_in_false=np.array(tm_in_false)
        tm_out_false=np.array(tm_out_false)
        tm_xyz_false_no_intervention = np.array(tm_xyz_false_no_intervention)
        if (len(tm_xyz_correct) != 0):
            ax.scatter(tm_xyz_correct[:,0],tm_xyz_correct[:,2],tm_xyz_correct[:,1],c="blue")
        if (len(tm_in_correct) != 0):
            ax.scatter(tm_in_correct[:,0],tm_in_correct[:,2],tm_in_correct[:,1],c="green")
        if (len(tm_out_correct) != 0):
            ax.scatter(tm_out_correct[:,0],tm_out_correct[:,2],tm_out_correct[:,1],c="yellow")
        if(len(tm_xyz_false)!=0):
            ax.scatter(tm_xyz_false[:, 0], tm_xyz_false[:, 2], tm_xyz_false[:, 1], c="red")
        if (len(tm_in_false) != 0):
            ax.scatter(tm_in_false[:, 0], tm_in_false[:, 2], tm_in_false[:, 1], c="red")
        if (len(tm_out_false) != 0):
            ax.scatter(tm_out_false[:, 0], tm_out_false[:, 2], tm_out_false[:, 1], c="red")
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
        if (len(tm_xyz_false)!=0):
            ax.scatter(tm_xyz_false[:,0],tm_xyz_false[:,2],tm_xyz_false[:,1], c="red")
        if (len(tm_in_false) != 0):
            ax.scatter(tm_in_false[:, 0], tm_in_false[:, 2], tm_in_false[:, 1], c="red")
        if (len(tm_out_false) != 0):
            ax.scatter(tm_out_false[:, 0], tm_out_false[:, 2], tm_out_false[:, 1], c="red")
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

        if (len(tm_xyz_correct_no_intervention) != 0):
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

    def plot_2D_Classification(self,model,X_train_time_series,Y_train_time_series,Class_train,i,name,path_NN_tm_loss):
        print("2D plot")
        Y_pred = np.argmax(model.predict(X_train_time_series), axis=-1)
        tm_xyz_correct = [];
        tm_xyz_correct_no_intervention = []
        tm_in_correct = [];
        tm_out_correct = []
        tm_in_false = [];
        tm_out_false = []
        tm_xyz_false = [];
        tm_xyz_false_no_intervention = []
        count_no_intervention_true = 0;
        count_intervention_true = 0
        count_no_intervention_false = 0;
        count_intervention_false = 0
        count_in_true = 0;
        count_out_true = 0
        count_in_false = 0;
        count_out_false = 0
        timestamp_no_intervention =[]
        timestamp_in = []
        timestamp_out = []
        timestamp_intervention=[]
        tm_Y_train_time_series_x=[]
        tm_Y_train_time_series_y=[]
        tm_Y_train_time_series_z=[]
        tm_Class_train=[]
        tm_timestamp=[]
        tm_class_train_num=[]


        timestamp = 0
        print("Y_pred len: "+str(len(Y_pred)))
        for p in range(len(Y_pred)):
            #if(Y_train_time_series[p,0]==8000 and Y_train_time_series[p,1]==8000 and Y_train_time_series[p,2]==8000):
            if(Class_train[p]==0):
                if (Y_pred[p] == Class_train[p]):
                    tm_xyz_correct_no_intervention.append(Y_train_time_series[p])
                    count_no_intervention_true = count_no_intervention_true +1
                    timestamp_no_intervention.append(timestamp)
                    tm_Y_train_time_series_x.append(Y_train_time_series[p, 0])
                    tm_Y_train_time_series_y.append(Y_train_time_series[p, 1])
                    tm_Y_train_time_series_z.append(Y_train_time_series[p, 2])
                    tm_Class_train.append("No Intervention")
                    tm_class_train_num.append(0)
                    tm_timestamp.append(timestamp)
                else:
                    tm_xyz_false_no_intervention.append(Y_train_time_series[p])
                    count_no_intervention_false = count_no_intervention_false +1
                    tm_Y_train_time_series_x.append(Y_train_time_series[p, 0])
                    tm_Y_train_time_series_y.append(Y_train_time_series[p, 1])
                    tm_Y_train_time_series_z.append(Y_train_time_series[p, 2])
                    tm_Class_train.append("False No Intervention")
                    tm_class_train_num.append(0)
                    tm_timestamp.append(timestamp)
            elif(Class_train[p]==1):
                if (Y_pred[p] == Class_train[p]):
                    tm_in_correct.append(Y_train_time_series[p])
                    count_in_true = count_in_true +1
                    tm_Y_train_time_series_x.append(Y_train_time_series[p, 0])
                    tm_Y_train_time_series_y.append(Y_train_time_series[p, 1])
                    tm_Y_train_time_series_z.append(Y_train_time_series[p, 2])
                    tm_Class_train.append("In")
                    tm_class_train_num.append(1)
                    tm_timestamp.append(timestamp)
                else:
                    tm_in_false.append(Y_train_time_series[p])
                    count_in_false = count_in_false +1
                    tm_Y_train_time_series_x.append(Y_train_time_series[p, 0])
                    tm_Y_train_time_series_y.append(Y_train_time_series[p, 1])
                    tm_Y_train_time_series_z.append(Y_train_time_series[p, 2])
                    tm_Class_train.append("False")
                    tm_class_train_num.append(1)
                    tm_timestamp.append(timestamp)
            elif (Class_train[p] == 2):
                if (Y_pred[p] == Class_train[p]):
                    tm_out_correct.append(Y_train_time_series[p])
                    count_out_true = count_out_true +1
                    tm_Y_train_time_series_x.append(Y_train_time_series[p, 0])
                    tm_Y_train_time_series_y.append(Y_train_time_series[p, 1])
                    tm_Y_train_time_series_z.append(Y_train_time_series[p, 2])
                    tm_Class_train.append("Out")
                    tm_class_train_num.append(2)
                    tm_timestamp.append(timestamp)
                else:
                    tm_out_false.append(Y_train_time_series[p])
                    count_out_false = count_out_false +1
                    tm_Y_train_time_series_x.append(Y_train_time_series[p, 0])
                    tm_Y_train_time_series_y.append(Y_train_time_series[p, 1])
                    tm_Y_train_time_series_z.append(Y_train_time_series[p, 2])
                    tm_Class_train.append("False")
                    tm_class_train_num.append(2)
                    tm_timestamp.append(timestamp)
            else:
                if(Y_pred[p]==Class_train[p]):
                    tm_xyz_correct.append(Y_train_time_series[p])
                    count_intervention_true = count_intervention_true +1
                    tm_Y_train_time_series_x.append(Y_train_time_series[p, 0])
                    tm_Y_train_time_series_y.append(Y_train_time_series[p, 1])
                    tm_Y_train_time_series_z.append(Y_train_time_series[p, 2])
                    tm_Class_train.append("Class "+str(int(Class_train[p])-2))
                    tm_class_train_num.append(int(Class_train[p])-2)
                    tm_timestamp.append(timestamp)
                else:
                    tm_xyz_false.append(Y_train_time_series[p])
                    count_intervention_false = count_intervention_false +1
                    tm_Y_train_time_series_x.append(Y_train_time_series[p, 0])
                    tm_Y_train_time_series_y.append(Y_train_time_series[p, 1])
                    tm_Y_train_time_series_z.append(Y_train_time_series[p, 2])
                    tm_Class_train.append("False")
                    tm_class_train_num.append(int(Class_train[p]) - 2)
                    tm_timestamp.append(timestamp)
            timestamp = timestamp + 0.00714

        #tm_xyz_correct = np.array(tm_xyz_correct)
        #tm_in_correct = np.array(tm_in_correct)
        #tm_out_correct = np.array(tm_out_correct)
        #tm_xyz_correct_no_intervention = np.array(tm_xyz_correct_no_intervention)
        #timestamp_no_intervention = np.array(timestamp_no_intervention)
        #timestamp_in = np.array(timestamp_in)
        #timestamp_out = np.array(timestamp_out)
        #timestamp_intervention = np.array(timestamp_intervention)

        colors={"No Intervention":"black","False No Intervention":"orange","False":"red","In":"green","Out":"yellow","Class 1":"blue","Class 2":"olive","Class 3":"brown","Class 4":"pink","Class 5":"purple","Class 6":"gray"}
        legend={"0":"False","1":"In","2":"Out","3":"Class 1","4":"Class 2","5":"Class 3","6":"Class 4","7":"Class 5","8":"Class 6"}

        plt.figure(20+i)
        #tm_Class_train = np.array(tm_Class_train)
        #tm_Y_train_time_series = np.array(tm_Y_train_time_series)
        df_x=pd.DataFrame(dict(Y_train_time_series_x=tm_Y_train_time_series_x,Class_train=tm_Class_train,tm_timestamp=tm_timestamp,class_train_num=tm_class_train_num))
        df_y=pd.DataFrame(dict(Y_train_time_series_y=tm_Y_train_time_series_y,Class_train=tm_Class_train,tm_timestamp=tm_timestamp,class_train_num=tm_class_train_num))
        df_z = pd.DataFrame(dict(Y_train_time_series_z=tm_Y_train_time_series_z,Class_train=tm_Class_train,tm_timestamp=tm_timestamp,class_train_num=tm_class_train_num))

        plt.subplot(3,1,1)
        plt.suptitle(name)
        plt.scatter(df_x["tm_timestamp"],df_x["Y_train_time_series_x"],c=df_x["Class_train"].map(colors)) #,hue=df_x["Class_train"].map(legend)
        plt.plot(np.array(tm_timestamp),np.array(tm_Y_train_time_series_x),c="black")
        plt.title("X Position")
        plt.subplot(3,1,2)
        plt.scatter(df_y["tm_timestamp"], df_y["Y_train_time_series_y"], c=df_y["Class_train"].map(colors))
        plt.plot(np.array(tm_timestamp), np.array(tm_Y_train_time_series_y), c="black")
        plt.title("Y Position")
        plt.subplot(3, 1, 3)
        plt.scatter(df_z["tm_timestamp"], df_z["Y_train_time_series_z"], c=df_z["Class_train"].map(colors))
        plt.plot(np.array(tm_timestamp), np.array(tm_Y_train_time_series_z), c="black")
        plt.title("Z Position")
        #plt.legend(["False","In","Out","Class"])

        #plt.figure(23+i)
        #pl.scatter(df_z["tm_timestamp"], df_z["Y_train_time_series_z"], c=df_z["Class_train"].map(colors))
        #for df_t,df_z,class_train in zip(df_z["tm_timestamp"],df_z["Y_train_time_series_z"],df_z["class_train_num"])  :
        #    pl.text(df_t,df_z,str(class_train),fontsize=9,color="red") #,,bbox=dict(color=None, alpha=0.5)

        #plt.figure(21)
        ##plt.scatter(timestamp_no_intervention, tm_xyz_correct_no_intervention[:, 0], color="black",marker=".")
        #plt.scatter(timestamp_in, tm_in_correct[:, 0], color="green",marker=".")
        #plt.scatter(timestamp_out,tm_out_correct[:, 0], color="yellow",marker=".")
        #plt.scatter(timestamp_intervention,tm_xyz_correct[:, 0], color="blue",marker=".")
        #plt.figure(23)
        ##plt.scatter(timestamp_no_intervention, tm_xyz_correct_no_intervention[:, 2], color="black",marker="." )
        #plt.scatter(timestamp_in, tm_in_correct[:, 2], color="green",marker=".")
        #plt.scatter(timestamp_out, tm_out_correct[:, 2], color="yellow",marker=".")
        #plt.scatter(timestamp_intervention, tm_xyz_correct[:, 2], color="blue",marker=".")

    def training_NN(self):
        time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(self.path+self.filename_training,delimiter=';')

        if(self.all_or_range_selection_Data==True):
            time = time_xyz_antennen_Signal_Komplex_all_Files[:,self.time]
            X_Signal_number=(time_xyz_antennen_Signal_Komplex_all_Files[:,self.X_Signal_number:(self.X_Signal_number+1)].astype(int))
            Class = (time_xyz_antennen_Signal_Komplex_all_Files[:,41:42].astype(int))
            X_1 = (time_xyz_antennen_Signal_Komplex_all_Files[:,self.Output_from:self.Output_to].astype(int))#[:,2:5]#
            Y_1 = (time_xyz_antennen_Signal_Komplex_all_Files[:,self.Output_from:self.Output_to].astype(int))#[:,2:5]#
        else:
            time = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.time]
            X_Signal_number = (time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to,self.X_Signal_number:(self.X_Signal_number+1)].astype(int))
            Class = (time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, 41:42].astype(int))
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

            if(self.save==False):
                X_train_time_series, Y_train_time_series, X_test_time_series, Y_test_time_series, normalizer_x, normalizer_y, Class_train, Class_test = Tensorflow.create_training_test_data(
                    self, X_1, Y_1, X_Signal_number,Class, i)
                Trainings_set_x = X_train_time_series
                Trainings_set_y = np.hstack((Y_train_time_series,Class_train))
                Test_set_x = X_test_time_series
                Test_set_y = np.hstack(( Y_test_time_series,Class_test))
                np.save(self.path+"\\Trainingsset_leave_one_out_x_xyz_to_Classes_02",Trainings_set_x)
                np.save(self.path + "\\Trainingsset_leave_one_out_y_xyz_to_Classes_02",Trainings_set_y)
                np.save(self.path + "\\Testset_leave_one_out_x_xyz_to_Classes_02", Test_set_x)
                np.save(self.path + "\\Testset_leave_one_out_y_xyz_to_Classes_02", Test_set_y)

            X_train_time_series = np.load(self.path+"\\Trainingsset_leave_one_out_x_xyz_to_Classes_02.npy")
            Training_set_y = np.load(self.path + "\\Trainingsset_leave_one_out_y_xyz_to_Classes_02.npy")
            X_test_time_series = np.load(self.path + "\\Testset_leave_one_out_x_xyz_to_Classes_02.npy")
            Test_set_y = np.load(self.path + "\\Testset_leave_one_out_y_xyz_to_Classes_02.npy")

            Y_train_time_series = Training_set_y[:,0:3]
            Class_train = Training_set_y[:,3:]
            Y_test_time_series = Test_set_y[:, 0:3]
            Class_test = Test_set_y[:, 3:]


            print("X_Time_Series_train_shape =", X_train_time_series.shape)
            print("Y_Time_Series_train_shape =", Y_train_time_series.shape)
            print("Class_train_shape = ", Class_train.shape)

            model = tf.keras.Sequential([#
                #TCN(64,3, activation='relu',padding="same", input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),
                tf.keras.layers.Conv1D(64,3, activation='relu',padding="same", input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2]),kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.Conv1D(64,3, activation='relu',padding="same",kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.SpatialDropout1D(0.5),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same",kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same",kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.SpatialDropout1D(0.5),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same",kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same",kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same",kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.SpatialDropout1D(0.5),
                tf.keras.layers.MaxPooling1D(2),
                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.MaxPooling1D(2),

                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.Conv1D(512, 3, activation='relu', padding="same"),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.SpatialDropout1D(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                #tf.keras.layers.Dense(64, activation='relu'),
                #tf.keras.layers.Dropout(0.5),
                #tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(9,activation="softmax")
            ])
            model.summary()
            time_stamp_from_NN =Tensorflow.get_actuall_time_stamp()
            doku = []
            loss = 0
            batch = self.batch_size
            print_vali=[]
            print_train=[]
            for opti in range(0,1):#len(self.optimizer)
            #for loss in range(0,len(self.loss_funktions)):
                print('\n\n' + self.loss_funktions[loss])
                model.compile(loss=self.loss_funktions[loss],optimizer=self.optimizer[0],metrics=self.metrics)
                #,decay=1e-6,nesterov=True,momentum=0.9
                start_time_count = time_1.perf_counter()
                callback = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=200,restore_best_weights=True)
                log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                #tensorboard_callback = tf.keras.callbacks.TensorBoard(
                #    log_dir=log_dir, histogram_freq=1)
                hparams_callback = hp.KerasCallback(log_dir, {
                    'num_relu_units': 512,
                    'dropout': 0.2
                }) #,callbacks=[callback]
                history = model.fit(X_train_time_series, Class_train, epochs=self.trainings_epochs,shuffle=True, verbose=self.verbose,batch_size=batch,validation_data=(X_test_time_series, Class_test)) #x=generator
                elapsed_time = time_1.perf_counter() -start_time_count
                path_NN = os.path.join(self.path + self.save_dir)
                if (False == os.path.isdir(path_NN)):
                    os.mkdir(path_NN)
                path_NN_tm = os.path.join(path_NN+'\\NN_from_'+time_stamp_from_NN)
                if(False==os.path.isdir(path_NN_tm)):
                    os.mkdir(path_NN_tm)
                path_NN_tm_loss = os.path.join(path_NN_tm+'\\'+self.loss_funktions[loss]+'_Batch_'+str(batch))#+str(self.creat_dataset_time_steps*(2**(i-1)))
                if (False == os.path.isdir(path_NN_tm_loss)):
                    os.mkdir(path_NN_tm_loss)
                path_NN_tm_loss= os.path.join(path_NN_tm_loss+'\\' + self.loss_funktions[loss])
                try:
                    model.save(os.path.join(path_NN_tm_loss+".h5"))
                    del model
                    model = tf.keras.models.load_model(os.path.join(path_NN_tm_loss+".h5"))
                except:
                    print("The model not saved")
                result_test =model.evaluate(X_test_time_series,Class_test)
                print("Evaluate: " + str(np.array(result_test)))
                Y_pred_test_time_series =np.argmax(model.predict(X_test_time_series),axis=-1)#[:4000]
                Y_pred_train_time_series = np.argmax(model.predict(X_train_time_series),axis=-1)
                #Y_pred_time_series = normalizer_y.inverse_transform(Y_pred_time_series)
                #Y_test_time_series = normalizer_y.inverse_transform(Y_test_time_series)#[:4000]

                Tensorflow.plot_2D_Classification(self, model, X_train_time_series, Y_train_time_series, Class_train, 0,
                                                  "Training", path_NN_tm_loss)

                Tensorflow.plot_2D_Classification(self, model, X_test_time_series, Y_test_time_series, Class_test, 1,
                                                  "Test", path_NN_tm_loss)


                #doku_training =Tensorflow.plot_3D_Classification(self, model, X_train_time_series, Y_train_time_series, Class_train, 0,
                #                                  "Training", path_NN_tm_loss)
                #doku_test = Tensorflow.plot_3D_Classification(self, model, X_test_time_series, Y_test_time_series, Class_test, 1,
                #                                  "Test", path_NN_tm_loss)

                #result_train =
                print('\n------------------------------------------------------------------------------------------------\n')
                print('\n\n' + self.loss_funktions[loss])
                print(str(self.optimizer[0]))
                opti_name = ["Adam", "Nadam", "SGD with Nesterov", "RMSprop", "Adadelta"]
                loss_history = history.history["loss"][:]
                loss_val_history = history.history["val_loss"][:]
                plt.figure(71)
                plt.title(str(opti_name[0]))
                plt.plot(loss_history)
                plt.plot(loss_val_history)
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend(["Training Loss","Validation Loss"])
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_" + str(opti_name[0]) + ".png")
                plt.close()
                plt.figure(72)
                acc_history = history.history["sparse_categorical_accuracy"][:]
                acc_val_history = history.history["val_sparse_categorical_accuracy"][:]
                pl.title("Adam")
                plt.plot(acc_history)
                plt.plot(acc_val_history)
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend(["Training Accuracy", "Validation Accuracy"])
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Accuracy"  + ".png")
                plt.close()

                plt.figure(75)
                plt.title(str(opti_name[0]))
                plt.plot(loss_val_history)
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                print_vali.append("Validation Loss ("+str(batch)+")")
                plt.legend(print_vali)
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_" + str(opti_name[0]) + "_" +str(batch)+"_val.png")

                plt.figure(76)
                plt.title(str(opti_name[0]))
                plt.plot(loss_history)##
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                print_train.append("Training Loss ("+str(batch)+")")
                plt.legend(print_train)
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_" + str(opti_name[0]) + "_" +str(batch)+"_train.png")


                plt.figure(73)
                plt.subplot(2,1,1)
                plt.title("Loss")
                plt.plot(loss_history)
                plt.legend(opti_name)
                plt.xlabel("Epochs")
                plt.ylabel("Training Loss")
                plt.subplot(2, 1, 2)
                plt.title("Validation Loss")
                plt.plot(loss_val_history)
                plt.legend(opti_name)
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                if (self.png == True and opti==len(opti_name)-1):
                    plt.savefig(path_NN_tm_loss + "_" + str(opti_name[opti]) + ".png")
                    plt.close(plt.figure(73))
                #print('Loss and accuracy Training:',history.history['loss'][-1],history.history["sparse_categorical_crossentropy"][-1])
                #print('Loss and accuracy val_Training:',history.history['val_loss'][-1],history.history['val_sparse_categorical_crossentropy'][-1])
                print('Training Time: %.3f seconds.' % elapsed_time)
                result_train = model.evaluate(X_train_time_series,Class_train,batch_size=self.batch_size)
                print("Evaluate Training: " + str(np.array(result_train)))

                bal_erg_test = balanced_accuracy_score(Y_pred_test_time_series, Class_test)
                bal_erg_train = balanced_accuracy_score(Y_pred_train_time_series,Class_train)
                print("Test balanced accuracy score: " + str(bal_erg_test))
                print("Training balanced accuracy score: " + str(bal_erg_train))

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
                doku.append("\nX_Time_Series_generator_shape ="+str(X_train_time_series.shape)+"\nY_Time_Series_generator_shape ="+str(X_train_time_series.shape))
                doku.append("\nTime Steps: "+str(self.creat_dataset_time_steps*(2**(i-1))))
                doku.append("\nScaler: "+str(self.scaler))
                doku.append('\n' + self.loss_funktions[loss]+ " Loss, " + str(self.trainings_epochs) + " Epochs, " + str(
                    self.learning_rate) + " Learning Rate, " + str(self.optimizer_name) + " Optimizer, " + str(
                    batch) + " Batch Size, " + str(self.metrics) + " Metrics")
                #doku.append('\n'+self.loss_funktions[loss]+', Loss Training: '+str(history.history['loss'][-1])+'  Train Metrics'+str(self.metrics)+':'+str(history.history['sparse_categorical_crossentropy'][-1]))
                #doku.append('\n'+self.loss_funktions[loss]+', Loss Validation: '+str(history.history['val_loss'][-1])+'  Validation Metrics'+str(self.metrics)+':'+str(history.history['val_sparse_categorical_crossentropy'][-1]))
                doku.append('\nTraining Time: %.3f seconds.' % elapsed_time)
                doku.append("\n")
                #doku.append(doku_training)
                #doku.append("\n")
                #doku.append(doku_test)
                doku.append("\nloss (Training-set): " +str(result_train))
                doku.append("\nloss (Test-set): " + str(result_test))
                doku.append("\nTraining Balance Accuracy: "+str(bal_erg_train))
                doku.append("\nTest Balance Accuracy: "+ str(bal_erg_test))

                #doku.append('loss test_sequenz:'+str(model.evaluate(x=np.expand_dims(X_test_time_series, axis=0),y=np.expand_dims(Y_test_time_series, axis=0))))
                doku.append('\n------------------------------------------------------------------------------------------------\n')
                if(len(self.loss_funktions)<loss):
                    plt.close()

                batch= int(batch/2)


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
                        conf.split_shuffle, conf.loss_funktions, conf.scaler,conf.threshold_antenna,conf.threshold_value, conf.pickle, conf.png,conf.kernel_init,conf.bias_init,conf.save)
    tensor.training_NN()