import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import time
from keras import Sequential
from keras.layers import Dense, LSTM,Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

scaler = StandardScaler()

class makeSignal():
    def learn_Signal_one(self):
        Signal_1 = pd.read_csv("C:\\Users\\dauserml\\Desktop\\goalref_simulation-master\\gui_multipleTabs\\tables\\holzRegal_119KHz_1.csv", header=None,sep='\n')

        Signal_1 = Signal_1[0].str.split(';', expand=True);
        Signal_1 = (Signal_1.values[18:len(Signal_1.values), ])
        return Signal_1

    def Signal_prediction(self):
        Signal_10 = pd.read_csv("C:\\Users\\dauserml\\Desktop\\goalref_simulation-master\\gui_multipleTabs\\tables\\holzRegal_119KHz_1.csv", header=None,sep='\n')
        Signal_prediction = Signal_10[0].str.split(';', expand=True); Signal_10 = (Signal_10.values[18:len(Signal_10.values), ])
        return Signal_prediction

    def Signal_convert(self,Signal_one):
        fig = plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.plot(Signal_one[:, 0]);
        plt.subplot(3, 1, 2)
        plt.plot(Signal_one[:, 1]);
        plt.subplot(3, 1, 3)
        plt.plot(Signal_one[:, 2]);

        Outputs_position = np.array(Signal_one[:, 0:3])
        Outputs_orientaion = np.array(Signal_one[:, 6:9])
        Outputs = np.hstack([Outputs_position, Outputs_orientaion])
        Inputs = np.array(Signal_one[:, 9:25])

        Inputs = scaler.fit_transform(Inputs)
        Outputs = scaler.fit_transform(Outputs)

        X = Inputs
        Y = Outputs

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test, Inputs, Outputs;

class make_dir():
    def dir(self):
        dat = time.time()
        dat = time.localtime(dat)
        hidden =""

        date = str("%s_" %dat.tm_year + "%s_" %dat.tm_mon + "%s_" %dat.tm_mday + "Uhrzeit_%s_"%dat.tm_hour + "%s"%dat.tm_min)
        print(date)
        path = ("C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\save_Model\\%s"%date)
        os.makedirs(path)
        return date

class NN:
    def train_NN(self,Numb_Iter,plot,hidden_layers,hidden_option):
        date = make_dir.dir(self=0)
        for i in range(0, Numb_Iter):
            #regressor = MLPRegressor(hidden_layers, solver='lbfgs', max_iter=8000, learning_rate_init=0.001,
            #                         activation='tanh', early_stopping=True)
            # hidden_layer_sizes=[50,50,50],solver='lbfgs',max_iter=4000,learning_rate_init=0.001,activation='tanh',early_stopping=True == gut
            regressor = DecisionTreeRegressor(random_state=0)##gute Ergebniss 6*10^-5
            #regressor = MultiOutputRegressor(Ridge(random_state=123)) geht aber nicht so gut wie DecisionTreeRegressor
            #regressor = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)) # mse 0.000166  r2_score= 0.9832
            te = cross_val_score(regressor, X_train, y_train, cv=20)
            print(np.mean(te))
            regressor.fit(X_train, y_train)
            Y_Neuronal = regressor.predict(Inputs)
            print('r2_score=', r2_score(Outputs,Y_Neuronal))
            hidden = ""
            for j in range(0, len(hidden_layers)):
                hidden = ("%s" % hidden + "%s_" % hidden_layers[j])
            torch.save(regressor,
                       "C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\save_Model\\%s" % date + "\\Netz_%d__" % i + "%s" % hidden)
            if (hidden_option == 1):
                hidden_layers = np.array(hidden_layers)
                hidden_lay = (hidden_layers * 2)
                hidden_layers = hidden_lay

            Y_Neuronal_1 = scaler.inverse_transform(Y_Neuronal)
            Outputs_1 = scaler.inverse_transform(Outputs)

            e = mean_squared_error(Y_Neuronal_1[:, 0:6], Outputs_1[:, 0:6])
            st = np.std(Y_Neuronal_1[:, 0:6] - Outputs_1[:, 0:6])
            print('Der MSE gesamt von',i,' beträgt: ', e, ' Die Standardabweichung: ', st)

        print('Der MSE 1 beträgt: ', mean_squared_error(Y_Neuronal_1[:, 0], Outputs_1[:, 0]))
        print('Der MSE 2 beträgt: ', mean_squared_error(Y_Neuronal_1[:, 1], Outputs_1[:, 1]))
        print('Der MSE 3 beträgt: ', mean_squared_error(Y_Neuronal_1[:, 2], Outputs_1[:, 2]))

        print('Der MSE 4 beträgt: ', mean_squared_error(Y_Neuronal_1[:, 3], Outputs_1[:, 3]))
        print('Der MSE 5 beträgt: ', mean_squared_error(Y_Neuronal_1[:, 4], Outputs_1[:, 4]))
        print('Der MSE 6 beträgt: ', mean_squared_error(Y_Neuronal_1[:, 5], Outputs_1[:, 5]))

        if(plot==1):
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            fig = plt.figure(2)
            plt.subplot(3, 1, 1)
            plt.plot(Y_Neuronal_1[:, 0])
            plt.plot(Outputs_1[:, 0, ])
            plt.title("X-Achse")
            plt.subplot(3, 1, 2)
            plt.plot(Y_Neuronal_1[:, 1])
            plt.plot(Outputs_1[:, 1, ])
            plt.title("Y-Achse")
            plt.subplot(3, 1, 3)
            plt.plot(Y_Neuronal_1[:, 2])
            plt.plot(Outputs_1[:, 2, ])
            plt.title("Z-Achse")
            fig = plt.figure(3)
            plt.subplot(3, 1, 1)
            plt.plot(Y_Neuronal_1[:, 3])
            plt.plot(Outputs_1[:, 3, ])
            plt.title("X-Winkel")
            plt.subplot(3, 1, 2)
            plt.plot(Y_Neuronal_1[:, 4])
            plt.plot(Outputs_1[:, 4, ])
            plt.title("Y-Winkel")
            plt.subplot(3, 1, 3)
            plt.plot(Y_Neuronal_1[:, 5])
            plt.plot(Outputs_1[:, 5, ])
            plt.title("Z-Winkel")
            plt.show()
        return regressor

    def load_NN(self,plot):
        regressor_2 = torch.load("C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\save_Model\\2020_8_18_Uhrzeit_12_43\\Netz_0__120_120_120_")
        Y_Neuronal_2 = regressor_2.predict(Inputs)
        Y_Neuronal_2 = scaler.inverse_transform(Y_Neuronal_2)
        Outputs_1 = scaler.inverse_transform(Outputs)

        e_2 = mean_squared_error(Y_Neuronal_2[:, 0:6], Outputs_1[:, 0:6])
        st_2 = np.std(Y_Neuronal_2[:, 0:6] - Outputs_1[:, 0:6])
        print('The load Network have a MSE: ', e_2, ' The Standardabweichung: ', st_2)
        if(plot==1):
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            fig = plt.figure(4)
            plt.subplot(3, 1, 1)
            plt.plot(Y_Neuronal_2[:, 0])
            plt.plot(Outputs_1[:, 0, ])
            plt.title("X-Achse")
            plt.subplot(3, 1, 2)
            plt.plot(Y_Neuronal_2[:, 1])
            plt.plot(Outputs_1[:, 1, ])
            plt.title("Y-Achse")
            plt.subplot(3, 1, 3)
            plt.plot(Y_Neuronal_2[:, 2])
            plt.plot(Outputs_1[:, 2, ])
            plt.title("Z-Achse")
            fig = plt.figure(5)
            plt.subplot(3, 1, 1)
            plt.plot(Y_Neuronal_2[:, 3])
            plt.plot(Outputs_1[:, 3, ])
            plt.title("X-Winkel")
            plt.subplot(3, 1, 2)
            plt.plot(Y_Neuronal_2[:, 4])
            plt.plot(Outputs_1[:, 4, ])
            plt.title("Y-Winkel")
            plt.subplot(3, 1, 3)
            plt.plot(Y_Neuronal_2[:, 5])
            plt.plot(Outputs_1[:, 5, ])
            plt.title("Z-Winkel")
            plt.show()
        return regressor_2

Signal_one= makeSignal.learn_Signal_one(self=0) # five Look-up-tables merge
#Signal_two= makeSignal.learn_Signal_two(self=0) # five Look-up-tables merge
#mse_Signale_look_up= mean_squared_error(Signal_one[:, 5:18], Signal_two[:, 5:18])# MSE between one and two Look-up-table
#print('MSE between Look-up-table one and two',mse_Signale_look_up) # 5 Look-up-tables merge

X_train, X_test, y_train, y_test,Inputs, Outputs=makeSignal.Signal_convert(self=0,Signal_one=Signal_one)#,Signal_two=Signal_two)

hidden_layers=[2,2,2]

regressor = NN.train_NN(self=0,Numb_Iter=1,plot=0,hidden_layers=hidden_layers,hidden_option=1)  #X-> Number, how often train the Network #plot = 0 -> no Figure plot
                                                                                                #hidden_option =1 -> x^2 quadriert Hidden_layers
regressor2= NN.load_NN(self=0,plot=1) #plot = 0 -> no Figure plot

