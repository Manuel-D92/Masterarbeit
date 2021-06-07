import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
import load_than_save_qualisys_and_wireshark_data as ltsqawd

#from keras.models import Sequential
#from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler,MinMaxScaler, PolynomialFeatures, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR,SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,Ridge,BayesianRidge,LinearRegression,Lasso
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from mpl_toolkits.mplot3d import Axes3D
import math
import re

class config_from_lqd_Script():  #load_qualisys_daten from Script
    path = r"C:\Users\dauserml\Documents\2020_11_16"
    path_test = r"C:\Users\dauserml\Documents\2020_11_16"
    name_file  = "2020_11_16_files"#"all_Files"

    # komplexer orentation faktor
    komplex_orentation_faktor = (np.exp(-1j * np.deg2rad(-27)) / 4.65)

    ##Cut and cut range
    cut = False # False-> No Cutting
    x_cut = [0, 1500]  # X cut from, and to
    y_cut = [0, 1200]  # Y cut from, and to
    z_cut = [-300, 300]  # Z cut from, and to [-2000, 100] Classification

    # cut_x_y_z -> out of Cut-Range -> Antenna Signal = 0
    cut_x_y_z_position_higher_then_antenna_null = False

    # Interpolation between NaN Sequence
    interpolation = True
    interpolation_value = 60 # 40*0.00714 = 0.2856s

    # Threshold for Antenna threshold_value > Antenna Signal = 0
    threshold_antenna = False
    threshold_value = 0.0001

class load_and_save_merge_data_qualisys_and_wireshark():
    def __init__(self,path,path_test,name_file,komplex_orentation_faktor,cut,x_cut,y_cut,z_cut,cut_x_y_z_position_higher_then_antenna_null,interpolation,interpolation_value,threshold_antenna,threshold_value):

        self.path = path # r"C:\Users\User\Desktop\dauserml_Messungen_2020_07_22\Referenz_alfred"
        self.path_test = path_test #r"C:\Users\User\Desktop\dauserml_Messungen_2020_07_22\Referenz_alfred"
        self.name_file = name_file #"all_Files"

        #komplexer orentation faktor
        self.komplex_orentation_faktor = komplex_orentation_faktor #(np.exp(-1j * np.deg2rad(-27)) / 4.65)

        #NaN options
        self.interpolation = interpolation
        self.interpolation_value = interpolation_value

        #Threshold Antenna
        self.threshold_antenna = threshold_antenna
        self.threshold_value = threshold_value

        ##Cut and cut range
        self.cut = cut # True
        self.x_cut = x_cut # [0,1500] # X cut from, and to
        self.y_cut = y_cut # [0,1200] # Y cut from, and to
        self.z_cut = z_cut # [-200,200] # Z cut from, and to

        self.cut_x_y_z_position_higher_then_antenna_null = cut_x_y_z_position_higher_then_antenna_null

    def covert_and_save_merge_data_qualisys_and_wireshark(self):
        #ltsqawd.load_than_save_qualisys_and_wireshark_data.load_than_save_qualisys_and_wireshark_data_compare(self=0)
        folder_list = ltsqawd.load_than_save_qualisys_and_wireshark_data.load_File_List_from_folder(path=self.path,name_file=self.name_file)
        klass_list = ltsqawd.load_than_save_qualisys_and_wireshark_data.get_which_subject_intervened(folder_list)
        time_xyz_antennen_Signal_Komplex_all_Files=[]
        no_intervention_all_Files=[]
        doku_interpolation=[]
        if(len(folder_list)!=0):
            k=0
            while(len(time_xyz_antennen_Signal_Komplex_all_Files)==0):
                if(klass_list[k]>0):
                    time_xyz_antennen_Signal_Komplex_all_Files,no_intervention,doku_interpolation_tm =ltsqawd.load_than_save_qualisys_and_wireshark_data.load_than_save_qualisys_and_wireshark_data_to_learning(self=self,path=self.path,File=folder_list[k],which_klass=klass_list[k],number_Signal=k)
                    doku_interpolation.append(str(folder_list[k]))
                    for dl in range (0,len(doku_interpolation_tm)):
                        doku_interpolation.append(str(doku_interpolation_tm[dl]))
                    if (len(no_intervention_all_Files) == 0):
                        no_intervention_all_Files= no_intervention
                    else:
                        no_intervention_all_Files = np.concatenate((no_intervention_all_Files, no_intervention))
                else:
                    if(len(no_intervention_all_Files)==0):
                        nope, no_intervention_all_Files,doku_interpolation_tm = ltsqawd.load_than_save_qualisys_and_wireshark_data.load_than_save_qualisys_and_wireshark_data_to_learning(
                            self=self, path=self.path, File=folder_list[k], which_klass=klass_list[k], number_Signal=k)
                        doku_interpolation.append(str(folder_list[k]))
                        for dl in range(0, len(doku_interpolation_tm)):
                            doku_interpolation.append(str(doku_interpolation_tm[dl]))
                    else:
                        nope, no_intervention,doku_interpolation_tm = ltsqawd.load_than_save_qualisys_and_wireshark_data.load_than_save_qualisys_and_wireshark_data_to_learning(
                            self=self, path=self.path, File=folder_list[k], which_klass=klass_list[k], number_Signal=k)
                        doku_interpolation.append(str(folder_list[k]))
                        for dl in range(0, len(doku_interpolation_tm)):
                            doku_interpolation.append(str(doku_interpolation_tm[dl]))
                        no_intervention_all_Files = np.concatenate((no_intervention_all_Files, no_intervention))
                k=k+1
            for i in range(k,len(folder_list)):
                time_xyz_antennen_Signal_Komplex,no_intervention,doku_interpolation_tm = (ltsqawd.load_than_save_qualisys_and_wireshark_data.load_than_save_qualisys_and_wireshark_data_to_learning(self=self,path=self.path,File=folder_list[i],which_klass=klass_list[i],number_Signal=i))
                doku_interpolation.append(str(folder_list[k]))
                for dl in range(0, len(doku_interpolation_tm)):
                    doku_interpolation.append(str(doku_interpolation_tm[dl]))
                if(len(time_xyz_antennen_Signal_Komplex)!=0):
                    time_xyz_antennen_Signal_Komplex_all_Files = np.concatenate((time_xyz_antennen_Signal_Komplex_all_Files,time_xyz_antennen_Signal_Komplex))
                    if (len(no_intervention_all_Files) != 0):
                        no_intervention_all_Files = np.concatenate((no_intervention_all_Files,no_intervention))
                else:
                    no_intervention_all_Files = np.concatenate((no_intervention_all_Files, no_intervention))
                    print("File: ",folder_list[k],"is empty (All Data are cut out!)")

            if(len(no_intervention_all_Files)!=0):
                time_xyz_antennen_Signal_Komplex_all_Files_and_no_intervention_all_Files = np.concatenate((time_xyz_antennen_Signal_Komplex_all_Files,no_intervention_all_Files))
            else:
                time_xyz_antennen_Signal_Komplex_all_Files_and_no_intervention_all_Files=time_xyz_antennen_Signal_Komplex_all_Files
            np.savetxt(('%s\\' % self.path + '%s.csv' % self.name_file), time_xyz_antennen_Signal_Komplex_all_Files_and_no_intervention_all_Files, delimiter=';',fmt='%10.6E')
            print("%d Files are found!"%len(folder_list))
            file = open(('%s\\' % self.path + '%s.txt' % self.name_file), 'w')
            for d in range(0, len(doku_interpolation)):
                file.write("\n"+str(doku_interpolation[d]))
            file.close()
        else:
            print("no Files found!")

    def load_merge_data_and_learn(self):
        time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(self.path+"\\"+self.name_file+".csv",delimiter=';')
        test_sequenz = np.loadtxt((self.path_test+"\\"+self.name_file+".csv"),delimiter=';')

        plt.subplot(3,1,1)
        plt.plot(time_xyz_antennen_Signal_Komplex_all_Files[:,2])
        plt.subplot(3,1,2)
        plt.plot(time_xyz_antennen_Signal_Komplex_all_Files[:,3])
        plt.subplot(3,1,3)
        plt.plot(time_xyz_antennen_Signal_Komplex_all_Files[:,4])


        X = time_xyz_antennen_Signal_Komplex_all_Files[:,5:]
        Y = time_xyz_antennen_Signal_Komplex_all_Files[:,2:5]

        plt.figure(14)
        plt.subplot(8,1,1)
        plt.plot(X[:,17:19])
        plt.subplot(8,1,2)
        plt.plot(X[:,1:3])
        plt.subplot(8,1,3)
        plt.plot(X[:,19:21])
        plt.subplot(8,1,4)
        plt.plot(X[:,3:5])
        plt.subplot(8,1,5)
        plt.plot(X[:,21:23])
        plt.subplot(8,1,6)
        plt.plot(X[:,7:9])
        plt.subplot(8,1,7)
        plt.plot(X[:,23:25])
        plt.subplot(8,1,8)
        plt.plot(X[:,9:11])

        plt.figure(15)
        plt.subplot(8,1,1)
        plt.plot(X[:,25:27])
        plt.subplot(8,1,2)
        plt.plot(X[:,11:13])
        plt.subplot(8,1,3)
        plt.plot(X[:,27:29])
        plt.subplot(8,1,4)
        plt.plot(X[:,13:15])
        plt.subplot(8,1,5)
        plt.plot(X[:,29:31])
        plt.subplot(8,1,6)
        plt.plot(X[:,15:17])
        plt.subplot(8,1,7)
        plt.plot(X[:,31:33])
        plt.subplot(8,1,8)
        plt.plot(X[:,17:19])

        X_test_sequenz = test_sequenz[:,4:]
        Y_test_sequenz = test_sequenz[:,1:4]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,shuffle=False)

        plt.figure(3)
        plt.subplot(2,1,1)
        plt.plot(X_train)
        plt.subplot(2,1,2)
        plt.plot(Y_train)

        #clf = RandomForestRegressor(n_estimators=200,min_samples_split=3,min_samples_leaf=2)
        #clf = OneVsOneClassifier(SGDClassifier(random_state=20,max_iter=8000))
        #clf = make_pipeline(StandardScaler(),MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(80,80), random_state=50, max_iter=2000))
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20+j*10,15+i), random_state=0, max_iter=2000)
        #clf = make_pipeline(MinMaxScaler(),OneVsOneClassifier(MultinomialNB()))


        clf = make_pipeline(Normalizer(), MLPRegressor(learning_rate_init=0.007,verbose=True,activation='relu',hidden_layer_sizes=[240],solver='adam',random_state=9,max_iter=50))
        #,learning_rate_init=0.0001
        #clf = make_pipeline(MinMaxScaler(),MultiOutputRegressor(SVR(kernel='rbf',verbose=True)))
        #clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
        #regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train,Y_train)

        #clf = make_pipeline(StandardScaler(),MultiOutputRegressor(LinearSVR()))
        #clf = make_pipeline(StandardScaler(),MultiOutputRegressor(DecisionTreeRegressor()))
        #clf = make_pipeline(MinMaxScaler(),MultiOutputRegressor(GradientBoostingRegressor()))

        #model = tf.keras.models.Sequential([
        #    tf.keras.layers.Conv1D(240,activation='relu',padding='same',input_shape=(32,1),kernel_size=4),
        #    tf.keras.layers.Flatten(),
        #    tf.keras.layers.Dense(250, activation='relu'),
        #    tf.keras.layers.Dense(3,activation='softmax')
        #])
        #model.compile(optimizer='adam',
        #              loss='categorical_crossentropy',
        #              metrics=['accuracy'])

        #models.append(('LR', LinearRegression()))
        #models.append(('NN',MLPRegressor(solver='lbfgs')))

        #model = RandomForestRegressor()
        #models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
        #models.append(('SVR', SVR(gamma='auto'))) # kernel = linear
        ###models.append(('KNN',KNeighborsRegressor()))
        ###tscv = TimeSeriesSplit(n_splits=10)
        ###param_search = {
        ###    'n_estimators': [20, 50, 100],
        ###    'max_features': ['auto', 'sqrt', 'log2'],
        ###    'max_depth' : [i for i in range(5,15)]
        ###}
        ###
        ###clf = GridSearchCV(estimator=models,cv=tscv,param_grid=param_search)


        #cifar10 = tf.keras.datasets.cifar10
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()

        #x_train, x_test = x_train / 255.0, x_test / 255.0



        #X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        #X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)
        #
        #in_dim = (X_train.shape[1], X_train.shape[2])
        #out_dim = Y_train.shape[1]
        #print(in_dim)

        #cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoint.hdf5')

        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir=r"C:\Users\dauserml\AppData\Local\Programs\Python\Python37\Scripts\logs")
        #model.fit(X_train, Y_train, epochs=15,
        #           validation_data=(X_test, Y_test),
        #           callbacks=[cp_callback, tb_callback])

        clf.fit(X_train, Y_train)

        #cross_val_score(clf, X_train,Y_train,cv=3)

        #model.fit(X_train,Y_train,validation_data=(X_test,Y_test),callbacks=[cp_callback])

        #print("Training Loss: ",model.named_steps['mlpregressor'].loss_)

        #print(clf.loss_curve_)

        #model = Sequential()
        #model.add(LSTM(130, input_shape=in_dim, activation="relu"))
        #model.add(Dense(out_dim))
        #model.compile(loss="mse", optimizer="adam")
        #model.summary()

        #model.fit(X_train,Y_train,epochs=1000, batch_size=100, verbose=1)

        #
        X_test_sequenz =X_train
        Y_test_sequenz =Y_train

        Y_prediction_test_sequenz = clf.predict(X_test_sequenz)
        #print("mse_gesamt test: ",mean_squared_error(Y_test,Y_prediction_test_sequenz))
        #print("mse test: ",mean_squared_error(Y_test_sequenz,Y_prediction_test_sequenz, multioutput='raw_values'))

        #torch.save(model, ("C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\model_1"))

        plt.figure(4)
        plt.subplot(2,1,1)
        plt.plot(Y_prediction_test_sequenz[:,0])
        plt.plot(Y_test_sequenz[:,0])
        plt.legend(["Prediction","Orginal"])
        error = Y_test_sequenz-Y_prediction_test_sequenz
        plt.subplot(2,1,2)
        plt.plot(error[:,0])

        plt.figure(5)
        plt.subplot(2,1,1)
        plt.plot(Y_prediction_test_sequenz[:,1])
        plt.plot(Y_test_sequenz[:,1])
        plt.legend(["Prediction","Orginal"])
        plt.subplot(2,1,2)
        plt.plot(error[:,1])

        plt.figure(6)
        plt.subplot(2,1,1)
        plt.plot(Y_prediction_test_sequenz[:,2])
        plt.plot(Y_test_sequenz[:,2])
        plt.legend(["Prediction","Orginal"])
        plt.subplot(2,1,2)
        plt.plot(error[:,2])

        plt.show()

if __name__ == '__main__':
    # load_qualisys_daten from Script
    lsmdqw = load_and_save_merge_data_qualisys_and_wireshark(config_from_lqd_Script.path,config_from_lqd_Script.path_test,
                                                             config_from_lqd_Script.name_file,config_from_lqd_Script.komplex_orentation_faktor,
                                                             config_from_lqd_Script.cut,config_from_lqd_Script.x_cut,config_from_lqd_Script.y_cut,
                                                             config_from_lqd_Script.z_cut,config_from_lqd_Script.cut_x_y_z_position_higher_then_antenna_null,
                                                             config_from_lqd_Script.interpolation,config_from_lqd_Script.interpolation_value,
                                                             config_from_lqd_Script.threshold_antenna,config_from_lqd_Script.threshold_value)
    #
    lsmdqw.covert_and_save_merge_data_qualisys_and_wireshark()
    lsmdqw.load_merge_data_and_learn()