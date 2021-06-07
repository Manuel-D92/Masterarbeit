from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import time as time_1
import pandas as pd
import os
import pickle
import tensorflow_probability as tfp

#loss_funktions=['MSE','KLD','MAE','MSLE','binary_crossentropy','categorical_crossentropy','categorical_hinge','cosine_similarity','hinge','kullback_leibler_divergence','logcosh','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error','mean_squared_logarithmic_error','poisson','squared_hinge','MAPE']#,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'

class RNN_LSTM_Tensorflow:
    def __init__(self):
        # Time Steps (create_dataset)
        self.creat_dataset_time_steps = 2
        self.time_steps_geo_folge = 7  # geometrische-Folge(zweier Potenzen) for time steps, example: (time_steps=16 andtime_steps_geo_folge=2)->  16,32

        # Path
        self.path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1_Test"
        self.path_test_sequenz = r"C:\Users\dauserml\Documents\2020_09_25\Testsequenz_1"
        self.filename_training = "\\all_Files_Z_400_mm_Mit_seperater_Signal_trennung.csv"
        self.filename_test_sequenz = "\\all_Files.csv"
        self.save_dir = '\\LSTM_Neuronale_Netze_Time_Series'
        # self.save_dir='\\Neuronale_Netze_Time_Series_Generator_nur_mit_X'

        # Trainings Daten
        self.all_or_range_selection_Data = True  # all=True, range selection=False
        self.training_Data_from = 0  # Only "all_or_range_selection_Data = False"
        self.training_Data_to = 400  # Only "all_or_range_selection_Data = False"
        self.time = 1  # 1= Timesteps
        self.X_Signal_number = 0  # 0= Signal delimitation
        self.Input_from = 9  # Input Signal from 9 to 41 (9=frame1_real,10=frame1_imag,11=frame2_real,....,40=main8_imag)
        self.Input_to = 41
        # Signal (5=Yaw,6=Roll,7=Pitch,8=Residual) (41= For Classification, witch shelf)
        self.Output_from = 2  # Ouput Signal from 2 to 4 (2=X,3=Y,4=Z)
        self.Output_to = 5

        # Training Options
        self.trainings_epochs = 20
        self.batch_size = 128
        self.verbose = 2
        self.metrics = ['MSE']
        self.learning_rate = 0.001
        self.optimizer_name = "Adam"
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        ## Split Options
        self.split_test_size = 0.2
        self.split_random_state = 42
        self.split_shuffle = False

        ## Loss Options
        # self.loss_funktions= ['MSE','huber_loss'] #,'MSE','huber_loss','MAP' Probabilistic lossfunktion (Bayesian) lambda y, p_y: -p_y.log_prob(y)   #
        self.loss_funktions = ['MSE']  # , 'MAE', 'huber_loss',
        # 'logcosh', 'MAPE',
        # 'MSLE']  # ,'sparse_categorical_crossentropy','serialize''log_cosh','kl_divergence','get','deserialize','huber_loss'

        ## Scaler Options
        self.scaler = preprocessing.Normalizer()

        ##Plot option
        self.pickle = True  # save in pickel can open in pickle_plot.py
        self.png = True  # save the Picture in png

    @staticmethod
    def get_actuall_time_stamp():
        dateTimeObj = datetime.now()
        timestamp_from_NN = dateTimeObj.strftime("%d_%b_%Y_%H_%M")
        return timestamp_from_NN

    @staticmethod
    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

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

    def shape_the_Signal(self,X_Signal_number_train,X_train_time_series_normalize,Y_train_time_series_normalize,i):
        count = 0
        X_train_time_series = []
        Y_train_time_series = []
        for j in range(int(X_Signal_number_train[0]),int(X_Signal_number_train[-1] + 1)):
            of = 0 + count
            count = count + np.count_nonzero(X_Signal_number_train[:] == j)
            to = count
            if(self.creat_dataset_time_steps*(2**i)<np.count_nonzero(X_Signal_number_train[:] == j)):
                X_train_time_series_tm, Y_train_time_series_tm = self.create_dataset(X_train_time_series_normalize[of:to][:],
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
        time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(self.path + self.filename_training, delimiter=';')

        if (self.all_or_range_selection_Data == True):
            time = time_xyz_antennen_Signal_Komplex_all_Files[:, self.time]
            X_Signal_number = time_xyz_antennen_Signal_Komplex_all_Files[:, self.X_Signal_number]
            X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:, self.Input_from:self.Input_to]  # [:,9:]#
            Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:, self.Output_from:self.Output_to]  # [:,2:5]#
        else:
            time = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to, self.time]
            X_Signal_number = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to,
                              self.X_Signal_number]
            X_1 = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to,
                  self.Input_from:self.Input_to]  # [:,9:]#
            Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[self.training_Data_from:self.training_Data_to,
                  self.Output_from:self.Output_to]  # [:,2:5]#
        ##Look up table Daten
        # ver = pd.read_csv(r'C:\Users\dauserml\Desktop\goalref_simulation-master\gui_multipleTabs\tables\holzRegal_119KHz_1.csv', skiprows=17,sep=';')
        # ver = np.array(ver)
        # X_1 = ver[:,6:25]
        # Y_1 = ver[:,:3]

        time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz = np.loadtxt(
            self.path_test_sequenz + self.filename_test_sequenz, delimiter=';')
        time_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:, 0]

        X_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:, 7:40]
        Y_1_test_sequenz = time_xyz_antennen_Signal_Komplex_all_Files_test_sequenz[:, 1:4]

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

            X_train_time_series,Y_train_time_series= RNN_LSTM_Tensorflow.shape_the_Signal(self=self,X_Signal_number_train=X_Signal_number_train,
                                                                      X_train_time_series_normalize=X_train_time_series_normalize,
                                                                      Y_train_time_series_normalize=Y_train_time_series_normalize,i=(i-1))
            X_test_time_series, Y_test_time_series = RNN_LSTM_Tensorflow.shape_the_Signal(self=self,X_Signal_number_train=X_Signal_number_test,
                                                                      X_train_time_series_normalize=X_test_time_series_normalize,
                                                                      Y_train_time_series_normalize=Y_test_time_series_normalize,i=(i-1))

            #X_train_time_series, Y_train_time_series = create_dataset(X_train_time_series_normalize,Y_train_time_series_normalize,time_steps=creat_dataset_time_steps*(2**(i-1)))
            #X_test_time_series, Y_test_time_series = create_dataset(X_test_time_series_normalize,Y_test_time_series_normalize,time_steps=creat_dataset_time_steps*(2**(i-1)))

            print("X_Time_Series_generator_shape =",X_train_time_series.shape)
            print("Y_Time_Series_generator_shape =",Y_train_time_series.shape)


            model = tf.keras.Sequential([#
                #Endcoder
                tf.keras.layers.LSTM(8,activation='relu',input_shape=(X_train_time_series.shape[1],X_train_time_series.shape[2])),#,return_sequences=True),
                #Decoder
                tf.keras.layers.RepeatVector(X_train_time_series.shape[1]),
                tf.keras.layers.LSTM(8,activation='relu',return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3)),
                #tf.keras.layers.Flatten(),
                #tf.keras.layers.RepeatVector(X_train_time_series.shape[1]),
                #tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True),
                #tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32)),
                ###
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=Y_train_time_series.shape[1])#,activation='linear')
            ])
            model.summary()
            time_stamp_from_NN = RNN_LSTM_Tensorflow.get_actuall_time_stamp()
            doku = []
            for loss in range(0, len(self.loss_funktions)):

                print('\n\n' + self.loss_funktions[loss])
                model.compile(loss=self.loss_funktions[loss],
                              optimizer=self.optimizer, metrics=self.metrics)
                # ,decay=1e-6,nesterov=True,momentum=0.9
                start_time_count = time_1.perf_counter()
                history = model.fit(X_train_time_series, Y_train_time_series, epochs=self.trainings_epochs, shuffle=True,
                                    verbose=self.verbose, batch_size=self.batch_size,
                                    validation_data=(X_test_time_series, Y_test_time_series))  # x=generator
                elapsed_time = time_1.perf_counter() - start_time_count
                path_NN = (self.path + self.save_dir)
                if (False == os.path.isdir(path_NN)):
                    os.mkdir(path_NN)
                path_NN_tm = (path_NN + '\\NN_from_' + time_stamp_from_NN)
                if (False == os.path.isdir(path_NN_tm)):
                    os.mkdir(path_NN_tm)
                path_NN_tm_loss = path_NN_tm + '\\' + self.loss_funktions[loss]+'_Time_Step_'+str(self.creat_dataset_time_steps*(2**(i-1)))
                if (False == os.path.isdir(path_NN_tm_loss)):
                    os.mkdir(path_NN_tm_loss)
                path_NN_tm_loss = path_NN_tm_loss + '\\' + self.loss_funktions[loss]
                try:
                    model.save(path_NN_tm_loss + ".h5")
                except:
                    print("The model not saved")
                # print(model.evaluate(X_test_time_series,Y_test_time_series))
                Y_pred_time_series = model.predict(X_test_time_series)  # [:4000]
                # Y_pred_time_series = normalizer_y.inverse_transform(Y_pred_time_series)
                # Y_test_time_series = normalizer_y.inverse_transform(Y_test_time_series)#[:4000]

                fig = plt.figure(1)
                plt.subplot(2, 1, 1)
                plt.title('X Position')
                plt.plot(Y_pred_time_series[:, 0])
                plt.plot(Y_test_time_series[:, 0])
                plt.legend(["X_pred", "X_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_time_series[:, 0] - Y_test_time_series[:, 0]) ** 2))
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
                plt.plot(Y_pred_time_series[:, 1])
                plt.plot(Y_test_time_series[:, 1])
                plt.legend(["Y_pred", "Y_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_time_series[:, 1] - Y_test_time_series[:, 1]) ** 2))
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
                plt.plot(Y_pred_time_series[:, 2])
                plt.plot(Y_test_time_series[:, 2])
                plt.legend(["Z_pred", "Z_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_time_series[:, 2] - Y_test_time_series[:, 2]) ** 2))
                if (self.png == True):
                    plt.savefig(path_NN_tm_loss + "_Training_Z.png")
                if (self.pickle == True):
                    with open(path_NN_tm_loss + "_Training_Z.pkl", "wb") as fp:  # Save Plots
                        pickle.dump(fig, fp, protocol=4)

                Y_pred_train_time_series = model.predict(X_train_time_series)  # [:4000]
                # Y_pred_train_time_series = normalizer_y.inverse_transform(Y_pred_train_time_series)
                # Y_train_time_series = normalizer_y.inverse_transform(Y_train_time_series)  # [:4000]
                if (loss < len(self.loss_funktions) - 1 or i < (self.time_steps_geo_folge)):
                    plt.close(fig=fig)
                fig = plt.figure(4)
                plt.subplot(2, 1, 1)
                plt.title('X Position train')
                plt.plot(Y_pred_train_time_series[:, 0])
                plt.plot(Y_train_time_series[:, 0])
                plt.legend(["X_pred", "X_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_train_time_series[:, 0] - Y_train_time_series[:, 0]) ** 2))
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
                plt.plot(Y_pred_train_time_series[:, 1])
                plt.plot(Y_train_time_series[:, 1])
                plt.legend(["Y_pred", "Y_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_train_time_series[:, 1] - Y_train_time_series[:, 1]) ** 2))
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
                plt.plot(Y_pred_train_time_series[:, 2])
                plt.plot(Y_train_time_series[:, 2])
                plt.legend(["Z_pred", "Z_true"])
                plt.subplot(2, 1, 2)
                plt.plot(np.sqrt((Y_pred_train_time_series[:, 2] - Y_train_time_series[:, 2]) ** 2))
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
                result = model.evaluate(X_test_time_series, Y_test_time_series, batch_size=128)
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
                doku.append("\nTime Steps: " + str(self.creat_dataset_time_steps * (2 ** (i - 1))))
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

            file = open(self.path + self.save_dir + '\\NN_from_' + time_stamp_from_NN + '\\Doku_Time_Step_' + str(
                self.creat_dataset_time_steps * (2 ** (i - 1))) + '.txt', 'w')
            for i in range(0, len(doku)):
                file.write(str(doku[i]))
            file.close()

        plt.show()

if __name__ == '__main__':
    RNN_LSTME_tensor = RNN_LSTM_Tensorflow()
    RNN_LSTME_tensor.training_NN()