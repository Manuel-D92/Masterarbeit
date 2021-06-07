from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import intel_tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import time as time_1
import pandas as pd
import os
import pickle
#import tensorflow_probability as tf

class config_Test_Tensorflow():
    # self.a = outside.t
    # Time Steps (create_dataset)
    creat_dataset_time_steps = 16
    time_steps_geo_folge = 2  # geometrische-Folge(zweier Potenzen) for time steps, example: (time_steps=16 andtime_steps_geo_folge=2)->  1

    # Path
    path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1_Test"
    path_test_sequenz = r"C:\Users\dauserml\Documents\2020_09_25\Testsequenz_1"
    filename_training = "\\all_Files_Z_400_mm_Mit_seperater_Signal_trennung.csv"
    filename_test_sequenz = "\\all_Files.csv"
    save_dir = '\\Batch_Size'
    # self.save_dir='\\Neuronale_Netze_Time_Series_Generator_nur_mit_X'

    # Trainings Daten
    all_or_range_selection_Data = False  # all=True, range selection=False
    training_Data_from = 0  # Only "all_or_range_selection_Data = False"
    training_Data_to = 200  # Only "all_or_range_selection_Data = False"
    time = 1  # 1= Timesteps
    X_Signal_number = 0  # 0= Signal delimitation
    Input_from = 9  # Input Signal from 9 to 41 (9=frame1_real,10=frame1_imag,11=frame2_real,....,40=main8_imag)
    Input_to = 41
    # Signal (5=Yaw,6=Roll,7=Pitch,8=Residual) (41= For Classification, witch shelf)
    Output_from = 41  # Ouput Signal from 2 to 4 (2=X,3=Y,4=Z)
    Output_to = 42

    # Training Options
    trainings_epochs = 200
    batch_size = 128
    verbose = 2
    metrics = ['accuracy']
    learning_rate = 0.001
    optimizer_name = "Adam"
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ## Split Options
    split_test_size = 0.2
    split_random_state = 42
    split_shuffle = False

    ## Loss Options
    # self.loss_funktions= ['MSE','huber_loss'] #,'MSE','huber_loss','MAP' Probabilistic lossfunktion (Bayesian) lambda y, p_y: -p_y.log_prob(y)   #
    loss_funktions = ['categorical_crossentropy','sparse_categorical_crossentropy']  # ,
    # 'logcosh', 'MAPE',
    # 'MSLE']  # ,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'

    ## Scaler Options
    scaler = preprocessing.Normalizer()

    ##Plot option
    pickle = False  # save in pickel can open in pickle_plot.py
    png = True  # save the Picture in png

#loss_funktions=['MSE','KLD','MAE','MSLE','binary_crossentropy','categorical_crossentropy','categorical_hinge','cosine_similarity','hinge','kullback_leibler_divergence','logcosh','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error','mean_squared_logarithmic_error','poisson','squared_hinge','MAPE']#,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'
class Tensorflow:

    def __init__(self,creat_dataset_time_steps,time_steps_geo_folge,path,path_test_sequenz,filename_training,filename_test_sequenz,save_dir,
                 all_or_range_selection_Data,training_Data_from,training_Data_to,time,X_Signal_number,Input_from,
                 Input_to,Output_from,Output_to,trainings_epochs,batch_size,verbose,metrics,learning_rate,optimizer_name,
                 optimizer,split_test_size,split_random_state,split_shuffle,loss_funktions,scaler,pickle,png):
        #self.a = outside.t
        # Time Steps (create_dataset)
        self.creat_dataset_time_steps = creat_dataset_time_steps
        self.time_steps_geo_folge= time_steps_geo_folge # geometrische-Folge(zweier Potenzen) for time steps, example: (time_steps=16 andtime_steps_geo_folge=2)->  16,32

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
        self.split_random_state=split_random_state
        self.split_shuffle = split_shuffle

        ## Loss Options
        #self.loss_funktions= ['MSE','huber_loss'] #,'MSE','huber_loss','MAP' Probabilistic lossfunktion (Bayesian) lambda y, p_y: -p_y.log_prob(y)   #
        self.loss_funktions = loss_funktions#,
                               #'logcosh', 'MAPE',
                               #'MSLE']  # ,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'

        ## Scaler Options
        self.scaler = scaler

        ##Plot option
        self.pickle = pickle #save in pickel can open in pickle_plot.py
        self.png = png #save the Picture in png

    @staticmethod
    def get_actuall_time_stamp():
        dateTimeObj = datetime.now()
        timestamp_from_NN = dateTimeObj.strftime("%d_%b_%Y_%H_%M")
        return timestamp_from_NN

    def first_NN(self,X_1,Y_1):
        X_train, X_test, Y_train, Y_test= train_test_split(X_1, Y_1, test_size=0.2,random_state=42,shuffle=True)
        normalizer = self.scaler
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
        history = model.fit(X_train,Y_train, epochs=2,verbose=2,validation_split=0.2,batch_size=128)#,validation_data=(X_test, Y_test))
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

    @staticmethod
    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    def shape_the_Signal(self,X_Signal_number_train,X_train_time_series_normalize,Y_train_time_series_normalize,i):
        count = 0
        X_train_time_series = []
        Y_train_time_series = []
        for j in range(int(X_Signal_number_train[0]),int(X_Signal_number_train[-1] + 1)):
            of = 0 + count
            count = count + np.count_nonzero(X_Signal_number_train[:] == j)
            to = count
            if(self.creat_dataset_time_steps*(2**i)<np.count_nonzero(X_Signal_number_train[:] == j)):
                X_train_time_series_tm, Y_train_time_series_tm = Tensorflow.create_dataset(X_train_time_series_normalize[of:to][:],
                                                                                           Y_train_time_series_normalize[of:to][:],
                                                                                           time_steps=self.creat_dataset_time_steps * (2 ** i))
                if(j==int(X_Signal_number_train[0])):
                    X_train_time_series = X_train_time_series_tm
                    Y_train_time_series = Y_train_time_series_tm
                else:
                    X_train_time_series= np.vstack((X_train_time_series,X_train_time_series_tm))
                    Y_train_time_series = np.vstack((Y_train_time_series, Y_train_time_series_tm))
        return X_train_time_series,Y_train_time_series

    def training_NN(self):
        time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(self.path+self.filename_training,delimiter=';')

        if(self.all_or_range_selection_Data==True):
            time = time_xyz_antennen_Signal_Komplex_all_Files[:,self.time]
            X_Signal_number=time_xyz_antennen_Signal_Komplex_all_Files[:,self.X_Signal_number]
            X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,self.Input_from:self.Input_to]#[:,9:]#
            Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,self.Output_from:self.Output_to]#[:,2:5]#
        else:
            time = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.time]
            X_Signal_number = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.X_Signal_number]
            X_1 = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.Input_from:self.Input_to]  # [:,9:]#
            Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.Output_from:self.Output_to]  # [:,2:5]#
        ##Look up table Daten
        #ver = pd.read_csv(r'C:\Users\dauserml\Desktop\goalref_simulation-master\gui_multipleTabs\tables\holzRegal_119KHz_1.csv', skiprows=17,sep=';')
        #ver = np.array(ver)
        #X_1 = ver[:,6:25]
        #Y_1 = ver[:,:3]

        time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz = np.loadtxt(self.path_test_sequenz+self.filename_test_sequenz,delimiter=';')
        time_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,0]

        X_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,7:40]
        Y_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:,1:4]

        print("X_beginn_shape =",X_1.shape)
        print("Y_beginn_shape =",Y_1.shape)

        #first_NN(X_1,Y_1)
        normalizer_test_sequenz_x = self.scaler
        X_1_test_sequenz_normlize=normalizer_test_sequenz_x.fit_transform(X_1_test_sequenz)
        normalizer_test_sequenz_y = self.scaler
        Y_1_test_sequenz_normlize=normalizer_test_sequenz_y.fit_transform(Y_1_test_sequenz)

        #X_1 = pd.DataFrame(X_1,columns=["frame1_real","frame1_imag","frame2_real","frame2_imag","frame3_real","frame3_imag","frame4_real","frame4_imag","frame5_real","frame5_imag","frame6_real","frame6_imag","frame7_real","frame7_imag","frame8_real","frame8_imag","main1_real","main1_imag","main2_real","main2_imag","main3_real","main3_imag","main4_real","main4_imag","main5_real","main5_imag","main6_real","main6_imag","main7_real","main7_imag","main8_real","main8_imag",])
        #Y_1 = pd.DataFrame(Y_1,columns=['X_postion'])#,'Y_position','Z_position'])

        X_train, X_test, Y_train, Y_test = train_test_split(X_1, Y_1, test_size=self.split_test_size,random_state=self.split_random_state,shuffle=self.split_shuffle)
        X_Signal_number_train,X_Signal_number_test = train_test_split(X_Signal_number, test_size=self.split_test_size,random_state=self.split_random_state,shuffle=self.split_shuffle)
        normalizer_x = self.scaler
        X_train_normalize= normalizer_x.fit_transform(X_train)
        X_test_normalize=normalizer_x.transform(X_test)
        X_train_time_series_normalize = pd.DataFrame(X_train_normalize)
        X_test_time_series_normalize = pd.DataFrame(X_test_normalize)

        normalizer_y = self.scaler
        Y_train_normalize= normalizer_y.fit_transform(Y_train)
        Y_test_normalize=normalizer_y.transform(Y_test)
        Y_train_time_series_normalize=pd.DataFrame(Y_train_normalize)
        Y_test_time_series_normalize=pd.DataFrame(Y_test_normalize)

        for i in range(1,(self.time_steps_geo_folge+1)):
            print("Time Steps: "+str(self.creat_dataset_time_steps*(2**(i-1))))

            X_train_time_series,Y_train_time_series= Tensorflow.shape_the_Signal(self=self,X_Signal_number_train=X_Signal_number_train,
                                                                                 X_train_time_series_normalize=X_train_time_series_normalize,
                                                                                 Y_train_time_series_normalize=Y_train_time_series_normalize,i=(i-1))
            X_test_time_series, Y_test_time_series = Tensorflow.shape_the_Signal(self=self,X_Signal_number_train=X_Signal_number_test,
                                                                                 X_train_time_series_normalize=X_test_time_series_normalize,
                                                                                 Y_train_time_series_normalize=Y_test_time_series_normalize,i=(i-1))

            #X_train_time_series, Y_train_time_series = create_dataset(X_train_time_series_normalize,Y_train_time_series_normalize,time_steps=creat_dataset_time_steps*(2**(i-1)))
            #X_test_time_series, Y_test_time_series = create_dataset(X_test_time_series_normalize,Y_test_time_series_normalize,time_steps=creat_dataset_time_steps*(2**(i-1)))

            print("X_Time_Series_generator_shape =",X_train_time_series.shape)
            print("Y_Time_Series_generator_shape =",Y_train_time_series.shape)


            model = tf.keras.Sequential([#
                tf.keras.layers.Conv1D(32,11,strides=2, activation='relu',padding="same", input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),
                tf.keras.layers.Conv1D(32,11, activation='relu',padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.AveragePooling1D(2),
                tf.keras.layers.Conv1D(64, 5, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(64, 5, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.AveragePooling1D(2),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(128, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.AveragePooling1D(2),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.AveragePooling1D(2),
                #tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                #tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                #tf.keras.layers.Conv1D(256, 3, activation='relu', padding="same"),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.SpatialDropout1D(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(units=Y_train_time_series.shape[1])
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
                history = model.fit(X_train_time_series, Y_train_time_series, epochs=self.trainings_epochs,shuffle=True, verbose=self.verbose,batch_size=self.batch_size,validation_data=(X_test_time_series, Y_test_time_series)) #x=generator
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
                #print(model.evaluate(X_test_time_series,Y_test_time_series))
                Y_pred_time_series =model.predict(X_test_time_series)#[:4000]
                #Y_pred_time_series = normalizer_y.inverse_transform(Y_pred_time_series)
                #Y_test_time_series = normalizer_y.inverse_transform(Y_test_time_series)#[:4000]

                fig = plt.figure(1)
                plt.subplot(2,1,1)
                plt.title('X Position')
                plt.plot(Y_pred_time_series[:, 0])
                plt.plot(Y_test_time_series[:, 0])
                plt.legend(["X_pred","X_true"])
                plt.subplot(2,1,2)
                plt.plot(np.sqrt((Y_pred_time_series[:, 0]-Y_test_time_series[:, 0])**2))
                if(self.png==True):
                    plt.savefig(path_NN_tm_loss+"_Training_X.png")
                if(self.pickle==True):
                    with open(path_NN_tm_loss+"_Training_X.pkl","wb") as fp:# Save Plots
                        pickle.dump(fig,fp,protocol=4)
                if(loss<len(self.loss_funktions)-1 or i<(self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(2)
                plt.subplot(2, 1, 1)
                plt.title('Y Position')
                plt.plot(Y_pred_time_series[:, 1])
                plt.plot(Y_test_time_series[:, 1])
                plt.legend(["Y_pred", "Y_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_time_series[:, 1] - Y_test_time_series[:, 1])**2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Training_Y.png")
                if(self.pickle==True):
                    with open(path_NN_tm_loss+"_Training_Y.pkl","wb") as fp:# Save Plots
                        pickle.dump(fig,fp,protocol=4)
                if(loss<len(self.loss_funktions)-1 or i<(self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(3)
                plt.subplot(2, 1, 1)
                plt.title('Z Position')
                plt.plot(Y_pred_time_series[:, 2])
                plt.plot(Y_test_time_series[:, 2])
                plt.legend(["Z_pred", "Z_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_time_series[:, 2] - Y_test_time_series[:, 2])**2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Training_Z.png")
                if(self.pickle==True):
                    with open(path_NN_tm_loss+"_Training_Z.pkl","wb") as fp:# Save Plots
                        pickle.dump(fig,fp,protocol=4)

                Y_pred_train_time_series = model.predict(X_train_time_series)  # [:4000]
                #Y_pred_train_time_series = normalizer_y.inverse_transform(Y_pred_train_time_series)
                #Y_train_time_series = normalizer_y.inverse_transform(Y_train_time_series)  # [:4000]
                if(loss<len(self.loss_funktions)-1 or i<(self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(4)
                plt.subplot(2, 1, 1)
                plt.title('X Position train')
                plt.plot(Y_pred_train_time_series[:, 0])
                plt.plot(Y_train_time_series[:, 0])
                plt.legend(["X_pred", "X_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_train_time_series[:, 0] - Y_train_time_series[:, 0])**2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Test_X.png")
                if(self.pickle==True):
                    with open(path_NN_tm_loss+"_Test_X.pkl","wb") as fp:# Save Plots
                        pickle.dump(fig,fp,protocol=4)
                if(loss<len(self.loss_funktions)-1 or i<(self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(5)
                plt.subplot(2, 1, 1)
                plt.title('Y Position train')
                plt.plot(Y_pred_train_time_series[:, 1])
                plt.plot(Y_train_time_series[:, 1])
                plt.legend(["Y_pred", "Y_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_train_time_series[:, 1] - Y_train_time_series[:, 1])**2))
                if(self.png==True):
                    plt.savefig(path_NN_tm_loss + "_Test_Y.png")
                if(self.pickle==True):
                    with open(path_NN_tm_loss + "_Test_Y.pkl", "wb") as fp:  # Save Plots
                        pickle.dump(fig, fp, protocol=4)
                if(loss<len(self.loss_funktions)-1 or i<(self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(6)
                plt.subplot(2, 1, 1)
                plt.title('Z Position train')
                plt.plot(Y_pred_train_time_series[:, 2])
                plt.plot(Y_train_time_series[:, 2])
                plt.legend(["Z_pred", "Z_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_train_time_series[:, 2] - Y_train_time_series[:, 2])**2))
                if(self.png==True):
                    plt.savefig(path_NN_tm_loss + "_Test_Z.png")
                if(self.pickle==True):
                    with open(path_NN_tm_loss+"_Test_Z.pkl","wb") as fp:# Save Plots
                        pickle.dump(fig,fp,protocol=4)
                if(loss<len(self.loss_funktions)-1 or i<(self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                #result_train =
                print('\n------------------------------------------------------------------------------------------------\n')
                print('\n\n' + self.loss_funktions[loss])
                print('Loss and MSE Training:',history.history['loss'][-1],history.history['MSE'][-1])
                print('Loss and MSE val_Training:',history.history['val_loss'][-1],history.history['val_MSE'][-1])
                print('Training Time: %.3f seconds.' % elapsed_time)
                result = model.evaluate(X_test_time_series,Y_test_time_series,batch_size=self.batch_size)
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
                doku.append('\n'+self.loss_funktions[loss]+', Loss Training: '+str(history.history['loss'][-1])+'  Train Metrics'+str(self.metrics)+':'+str(history.history['MSE'][-1]))
                doku.append('\n'+self.loss_funktions[loss]+', Loss Validation: '+str(history.history['val_loss'][-1])+'  Validation Metrics'+str(self.metrics)+':'+str(history.history['val_MSE'][-1]))
                doku.append('\nTraining Time: %.3f seconds.' % elapsed_time)
                #doku.append("\nloss (test-set): " +str(result))
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
    tensor = Tensorflow(conf.creat_dataset_time_steps,conf.time_steps_geo_folge,conf.path,conf.path_test_sequenz,conf.filename_training,
                        conf.filename_test_sequenz,conf.save_dir,conf.all_or_range_selection_Data,conf.training_Data_from,conf.training_Data_to,
                        conf.time,conf.X_Signal_number,conf.Input_from,conf.Input_to,conf.Output_from,conf.Output_to,conf.trainings_epochs,conf.batch_size,
                        conf.verbose,conf.metrics,conf.learning_rate,conf.optimizer_name,conf.optimizer,conf.split_test_size,conf.split_random_state,
                        conf.split_shuffle,conf.loss_funktions,conf.scaler,conf.pickle,conf.png)
    tensor.training_NN()